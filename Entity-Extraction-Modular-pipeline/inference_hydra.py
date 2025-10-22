import os
import sys
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv


#loggers
import wandb
from loguru import logger
import hydra
from omegaconf import OmegaConf, DictConfig

import torch
from nervaluate import Evaluator #finegrained evaluation 
from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report
# custom module
from steps.load_ner_dataset_hf import load_ner_dataset  
from steps.tokenize_preprocess import tokenize_and_align, cast_to_class_labels
from utils.hf_utils import count_entity_labels  #custom module using seqeval behind the scenes
#inference helper functions
from utils.wandb_utils import init_wandb_run
from utils.helper_functions import set_seed
from utils.inference_utils import (inference_dataloader, 
                                    init_nervaluate, 
                                    extract_and_flip_evaluator_result,
                                    create_wandb_table_seqeval,
                                    create_wandb_table_nervaluate,
                                    display_entity_tables,
                                    hf_metrics_inf,
                                    get_logger_inf,
                                    get_model_inf,
                                    get_model_name,
                                    get_model_dir_wandb,
                                    ) 




@logger.catch
@hydra.main(config_path="config", config_name="inference_config", version_base=None)
def run_inference(cfg:DictConfig):

    #seed reproducibility seed
    set_seed(cfg.seed) 
    #load environment variables
    load_dotenv()
    #init_logger
    get_logger_inf() 
    dataset_name = cfg.inference_dataset_path.split("/")[-1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb_run = None
    

    if not cfg.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        logger.info(f"Logging to Wandb is disabled for this run")

    

    else:
      #init wandb if set to true in config
        wandb_token = os.environ.get("WANDB_TOKEN")
        
        wandb.login(key=wandb_token)
        temp_run_name = f"Inference-{dataset_name}-{wandb.util.generate_id()}"
        
        wandb_run = init_wandb_run(mode="Inference", 
                                run_name=temp_run_name,
                                # project=cfg.logging.wandb.run.project,
                                # entity=cfg.logging.wandb.run.entity
                                    )
        logger.info(f"Logging to Wandb is enabled for this run. Run logs and metadata will be logged to: {cfg.logging.wandb.run.project}")

      

    model_path, model_name = get_model_inf(cfg, wandb_run)

    #model_name = get_model_name(model_path)
    logger.info(f"Model used for this inference is: {model_name}")

    final_run_name = f"Inference-{model_name}-Model_{dataset_name}-Dataset-{wandb.util.generate_id()}"
    wandb_run.name = final_run_name
    wandb_run.tags = [f"model:{model_name}",f"dataset:{dataset_name}", f"mode:inference"]
    wandb_run.save()
    
    nervaluate_artifact_aliases = [dataset_name, "nervaluate-metrics", model_name]
    seqeval_artifact_aliases = [dataset_name, "seqeval-metrics", model_name]

    #check dataset format 
    logger.info(f"Loading Inference dataset from path: {cfg.inference_dataset_path}")
    test_dataset = inference_dataloader(cfg.inference_dataset_path)
    logger.success("Dataset loaded successfully....Checking dataset structure-------->")
    logger.info(f"Test dataset after checking dataset structure: {test_dataset}") 

    #get label stats in inference dataset
    test_entity_count_iob, _ = count_entity_labels(test_dataset, cfg.label_col)
    logger.info(f"Inference dataset Entity distribution count: \n{test_entity_count_iob}")  
    logger.info(f"Loading model from path---->: {model_path}")

    #init tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)

    #get model label mapping
    label_list = list(model.config.id2label.values())
    logger.success(f"Model Loaded successfully. This model has been trained with the following entities: {label_list}")  

    #Transform datasets to ensure dataset labels are aligned to model labels
    logger.info("Casting Dataset from strings to integers")
    logger.info(f"Dataset sample before casting is applied: {test_dataset[0]}")
    test_dataset = cast_to_class_labels(test_dataset, 
                                        text_col=cfg.text_col, 
                                        label_col=cfg.label_col, 
                                        unique_tags=label_list
                                          )
    logger.success(f"Dataset casting completed. Dataset sample after casting is applied: {test_dataset[0]}")
    logger.info(f"Mapping for labels str: int is ----->: {model.config.id2label}")
    
    #tokenize and align dataset
    data_collator = DataCollatorForTokenClassification(tokenizer)
    tokenize_fn = lambda batch: tokenize_and_align(batch, tokenizer=tokenizer, text_col=cfg.text_col, tag_col=cfg.label_col)
    tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=test_dataset.column_names)

    # Metric function (seqeval)
    metric = hf_metrics_inf(label_list)

    save_dir = cfg.output_dir if cfg.output_dir else "./"

    test_args = TrainingArguments(
        output_dir=save_dir,
        do_train=False,
        do_predict=True,
        group_by_length=True,
        per_device_eval_batch_size=16,
        dataloader_num_workers=0,
        dataloader_drop_last=False,
        logging_dir=cfg.output_dir,
        run_name = final_run_name,
        report_to = "wandb" if cfg.use_wandb else "none"
    )

    trainer = Trainer(
        model=model,
        args=test_args,
        data_collator=data_collator,
        compute_metrics=metric,
        processing_class=tokenizer,
    )
    logger.info("Init Hf Trainer for prediction")
    logger.info("Running predictions on test dataset...")
    predictions, labels, metrics_seqeval = trainer.predict(test_dataset=tokenized_test,
                                                          metric_key_prefix="inference")
    logger.info(f"\nSeqeval metrics for this run: \n{metrics_seqeval}")


    # nervaluate evaluation
    pred_label_ids = np.argmax(predictions, axis=-1)

    true_labels, pred_labels = [], []
    for true_label, pred_label in zip(labels, pred_label_ids):
        true_temp, pred_temp = [], []
        for t, p in zip(true_label, pred_label):
            if t != -100:  # filter ignored tokens
                true_temp.append(label_list[t])
                pred_temp.append(label_list[p])
        true_labels.append(true_temp)
        pred_labels.append(pred_temp)

    logger.info("Running nervaluate metrics...")

    evaluator = init_nervaluate(true_labels, pred_labels)
    eval_results = evaluator.evaluate()
    overall_results, overall_results_per_entities = eval_results["overall"], eval_results["entities"]

    logger.info(f'\n Nervaluate\'s Finegrained metrics summary report(all): \n{evaluator.summary_report(mode="entities")}')
    logger.info(f'\n Nervaluate\'s Finegrained metrics summary report(Entity-Level): \n{evaluator.summary_report(mode="entities", scenario="ent_type")}')
    logger.info(f'\n Nervaluate\'s Finegrained metrics summary report(with partial matches count): \n{evaluator.summary_report(mode="entities", scenario="partial")}')

    #flip overall results and save to a dataframe
    extracted_results, overall_results_df = extract_and_flip_evaluator_result(overall_results)
    per_entity_results_df = display_entity_tables(overall_results_per_entities)


        #Log relevant variables to wandb
    if cfg.use_wandb and wandb_run is not None:
        wandb_run.log({"Device for run": device,
                        "Inference Dataset used": dataset_name,
                        "Model used for inference": model_name,
                        "Model Tags list": label_list,
                        "Inference dataset Entity distribution count": test_entity_count_iob,
                        })
        #log metrics as table to Wandb
        #seqeval classification report
        logger.info("Logging generated metrics to wandb")

        seqeval_table = create_wandb_table_seqeval(true_labels=true_labels, 
                                                pred_labels=pred_labels, 
                                                class_names = ["CellLine", "CellType", "Tissue"])
        report_columns = ["Entity", "Precision", "Recall", "F1", "Support"]
        wandb_run.log({
            "Seqeval Classification Report": wandb.Table(data=seqeval_table, columns=report_columns)
            })
        seqeval_artifact = wandb.Artifact(
                name=f"{model_name}-{dataset_name}_seqeval_nervaluate_metrics",
                type="inference",
                description=f"Seqeval Classification for Run",
                metadata={"model-used":model_name, "dataset-used":dataset_name}
                )
        wandb_run.log_artifact(seqeval_artifact, aliases=seqeval_artifact_aliases)
            
        #nervaluate's metrics
        for entity, df in per_entity_results_df.items():
            wandb_table = create_wandb_table_nervaluate(df)
            wandb_run.log({f"{entity}-nervaluate_metrics": wandb_table})

            #add to artifact
            nervaluate_artifact = wandb.Artifact(
                name=f"{entity.lower()}_{dataset_name}_{model_name}-nervaluate_metrics",
                type="inference",
                description=f"Nervaluate metrics for {entity}",
                metadata={"entity": entity, "model-used":model_name, "dataset-used":dataset_name}
                )
            nervaluate_artifact.add(wandb_table, name="metrics_table")
            wandb_run.log_artifact(nervaluate_artifact, aliases= nervaluate_artifact_aliases + [entity])

        logger.success("Entity results logged successfully")
        wandb_run.finish() if cfg.use_wandb and wandb_run is not None else None



if __name__ == "__main__": run_inference()








        
