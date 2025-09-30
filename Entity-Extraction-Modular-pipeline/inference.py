"""
This script runs inference on a dataset using a finetuned transformer model. 
It assummes dataset have columns: ["tokens", "tags"] and "tags" are in string representation e.g ["B-CellLine", "I-CellLine", "B-CellType", "O"]

Script Args: 
Dataset path can be a local path or name of a dataset in huggingface. 
Model path: Path to a transformer model either local dir/HF hub

"""

import os
import sys
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser

#loggers
import wandb
from loguru import logger


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
from utils.inference_utils import (inference_dataloader, 
                                    init_nervaluate, 
                                    extract_and_flip_evaluator_result,
                                    create_wandb_table_seqeval,
                                    create_wandb_table_nervaluate,
                                    display_entity_tables,
                                    hf_metrics_inf
                                    ) 

#init argument parser
parser = ArgumentParser()
parser.add_argument("--dataset_path", help="Name or path to dataset")
parser.add_argument("--model", help="Path or hfname of model to use for inference")
parser.add_argument("--output_dir", help="Path to save inference metrics to", default="./")
# parser.add_argument("--use_wandb", default=True, type=bool,
#                     help="Whether to activate logging to Weight and Bias")
parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
parser.set_defaults(use_wandb=True)

args = parser.parse_args()

#init logger
logger.remove(0)
logger.add(sys.stderr, 
           format=("[<green>{time:MM DD, YYYY -> HH:mm:ss}]</green>>>" 
                   "<blue>{name}:{line}</blue>"
                    " | <level>{level}</level> |"
                     " <cyan>{message}</cyan>"
                   
                   ),
)

#wandb login
if args.use_wandb:
    logger.info("Logging to wand is enabled for this run ")
    WANDB_KEY = os.environ.get("WANDB_TOKEN")
    wandb.login(key=WANDB_KEY)
    #wandb.login(key="e04178eb00a14485eae9448bb2f116176f294391")
    dataset_name = args.dataset_path.split("/")[-1]
    model_name = args.model.split("/")[-1]
    wandb_params = {"dataset_name": dataset_name, 
                    "model_name": model_name, 
                    }
    wandb_run = init_wandb_run(mode="Inference", 
                               run_name=f"Inference-run_{model_name}-{dataset_name}",
                               project="otar-3088",
                               entity="ebi_literature"
                        
                                            )

#init device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Current device in use: {device}")


# #load dataset
# test_dataset = load_ner_dataset(args.dataset_path)
# logger.info("Loading dataset----------->")
# logger.success("Dataset loaded successfully")
# logger.info(f"Test dataset loaded from source: {test_dataset}")
#check dataset format 
test_dataset = inference_dataloader(args.dataset_path)
logger.info(f"Test dataset after checking dataset structure: {test_dataset}")


test_entity_count_iob, _ = count_entity_labels(test_dataset, "tags")
logger.info(f"Inference dataset Entity distribution count: \n{test_entity_count_iob}")

logger.info(f"Loading model from path---->: {args.model}")
model = AutoModelForTokenClassification.from_pretrained(args.model).to(device)
label_list = list(model.config.id2label.values())
logger.success(f"Model Loaded successfully. This model has been trained with the following entities: {label_list}")

#init tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)

## Transform datasets to ensure dataset labels are aligned to model labels
logger.info("Casting Dataset from strings to integers")
logger.info(f"Dataset sample before casting is applied: {test_dataset[0]}")
test_dataset = cast_to_class_labels(
    test_dataset, text_col="tokens", label_col="tags", unique_tags=label_list
)
logger.success(f"Dataset casting completed. Dataset sample after casting is applied: {test_dataset[0]}")
logger.info(f"Mapping for labels str: int is ----->: {model.config.id2label}")
#tokenize and align dataset

data_collator = DataCollatorForTokenClassification(tokenizer)
tokenize_fn = lambda batch: tokenize_and_align(batch, tokenizer=tokenizer, text_col="tokens", tag_col="tags")
tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=test_dataset.column_names)

# Metric function (seqeval)
metric = hf_metrics_inf(label_list)


training_args = TrainingArguments(
    output_dir=args.output_dir,
    do_train=False,
    do_predict=True,
    group_by_length=True,
    per_device_eval_batch_size=16,
    dataloader_drop_last=False,
    logging_dir=args.output_dir,
    report_to = "wandb" if args.use_wandb else "none"
)

trainer = Trainer(
    model=model,
    args=training_args,
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
#entity, overall_results_per_entity_df = extract_and_flip_evaluator_result(overall_results_per_entities)
per_entity_results_df = display_entity_tables(overall_results_per_entities)



#Log relevant variables to wandb
if args.use_wandb:
    wandb_run.log({"Device for run": device,
                    "Inference Dataset path": args.dataset_path,
                    "Model for inference": args.model,
                    "Model Tags list": label_list,
                    "Inference dataset Entity distribution count": test_entity_count_iob,
                    "Predictions": pred_labels,
                    "SeqEval Classification report": metrics_seqeval,
                    "Nervaluate's Finegrained metrics summary report(all)": evaluator.summary_report(mode="entities"),
                    "Nervaluate's Finegrained metrics summary report(Entity-Level)": evaluator.summary_report(mode="entities", scenario="ent_type"),
                    "Nervaluate's Finegrained metrics summary report(with partial count))": evaluator.summary_report(mode="entities", scenario="partial"),
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
    #nervaluate's metrics
    for entity, df in per_entity_results_df.items():
        wandb_table = create_wandb_table_nervaluate(df)
        wandb_run.log({f"{entity}-nervaluate_metrics": wandb_table})

        #add to artifact
        run_artifact = wandb.Artifact(
            name=f"{entity.lower()}_nervaluate_metrics",
            type="inference",
            description=f"Nervaluate metrics for {entity}",
            metadata={"entity": entity, "model-used":model_name, "dataset-used":dataset_name}
            )
        run_artifact.add(wandb_table, name="metrics_table")
        wandb_run.log_artifact(run_artifact, aliases=["cellate", "nervaluate-metrics"])

    logger.success("Entity results logged successfully")



"""
Script TO-DO: 

1. Convert to a hydra-compatible script with flexibility on arguments and function parameters. 
2. allow to pass on "casting dataset" if dataset is already one-hot encoded. 
3. Create a func for nervaluate that accepts a tag_list as param to be set.  --> Done (added to inference_utils script)
4. Clean-up Wandb logging logic

"""