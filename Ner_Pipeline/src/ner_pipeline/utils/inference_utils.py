
import os
import sys
from typing import Union, List
import pandas as pd
import numpy as np
from IPython.display import display, HTML

import wandb
from loguru import logger
from datasets import Dataset, DatasetDict

import evaluate
from nervaluate import Evaluator
from seqeval.metrics import classification_report
from steps.load_ner_dataset_hf import load_ner_dataset



def inference_dataloader(dataset_path):
  #load dataset
  test_dataset = load_ner_dataset(dataset_path)
  logger.info("Loading dataset----------->")
  logger.success("Dataset loaded successfully")
  logger.info(f"Test dataset loaded from source: {test_dataset}")

  #check dataset format 
  test_dataset = check_dataset_struc(test_dataset)
  logger.info(f"Test dataset after checking dataset structure: {test_dataset}")

  return test_dataset


def check_dataset_struc(dataset:Union[Dataset, DatasetDict], 
                          split:str="train") -> Dataset:
    """
    Checks dataset structure and composition in the case of presence of multiple splits when loading from huggingface. 
    Args:
        dataset: Dataset or DatasetDict object
        split: Split to use for inference. Options: [train, validation, test]
    Returns:
        datasets.Dataset(obj) 
    """
    if isinstance(dataset, DatasetDict) and split in dataset:
        return dataset[split]
    elif isinstance(dataset, Dataset):
        return dataset
    else:
        raise ValueError("Dataset format not recognized.")


def init_nervaluate(true_labels:List[List[str]], 
                    pred_labels:List[List[str]], 
                    ner_labels:List=["CellLine", "CellType", "Tissue"]
                    ) -> Evaluator: 
    """
    Args: 
        true_labels: Ground Truth Labels
        pred_labels: Model's predictions
        ner_labels: List of unique ner tag names without "O" label. 
    Returns:
        nervaluate.Evaluator Object
    """

    return Evaluator(true_labels, pred_labels, tags=ner_labels, loader='list')


def extract_and_flip_evaluator_result(results):
    """
    Extracts results from a nervaluator object, 
    then passes to a dataframe and returns a transposed format for better visualisation

    """
    extracted_results = {k: v.__dict__ for k, v in results.items()}
    results_df = pd.DataFrame(extracted_results).T
    results_df = convert_2_percent(results_df)

    return extracted_results, results_df



def display_entity_tables(results):
    """
    Display nervaluate's results per entity type as a Pandas dataframe
    and return a dictionary of DataFrames for logging to wandb
    """
    entity_dfs = {}
    for entity, evals in results.items():
        _, df = extract_and_flip_evaluator_result(evals)
        logger.info(f"\nEntity: {entity}\n{df.to_string(index=True)}")
        entity_dfs[entity] = df
    return entity_dfs


def convert_2_percent(df):
  for col in df.columns:
    if col.startswith(("precision", "recall", "f1")):
        df[col] = df[col].apply(lambda x: f"{x*100:.2f}%")
  return df


def create_wandb_table_seqeval(true_labels: List,
                               pred_labels: List, 
                               class_names: List[str]) -> List[str]:

    report = classification_report(true_labels, pred_labels, digits=3)
    report_table = []
    report = report.splitlines()
    for line in report[2:(len(class_names)+2)]:
        report_table.append(line.split())

    return report_table


def create_wandb_table_nervaluate(df: pd.DataFrame):
  
    entity_wandb_table = wandb.Table(
        columns=["Evaluation Type"] + list(df.columns),
        data=[[idx] + row.tolist() for idx, row in df.iterrows()]
        )
  
    return entity_wandb_table



def hf_metrics_inf(label_list):
  metric = evaluate.load("seqeval")
  def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=2)
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
  return compute_metrics



def get_model_inf(cfg, wandb_run=None):
    if cfg.fetch_from_wandb and cfg.fetch_from_hf:
        raise ValueError("Only one of `fetch_from_hf` or `fetch_from_wandb` can be set to True at a time")
    if getattr(cfg, "fetch_from_wandb", False) and cfg.use_wandb and wandb_run is not None:
        model_artifact = wandb_run.use_artifact(cfg.wandb_artifact_dir, type="model")
        model_name = get_model_name(cfg.wandb_artifact_dir)
        model_name = model_name.split(":")[0]
        model_dir = model_artifact.download()
        model_dir = get_model_dir_wandb(model_dir)
    elif getattr(cfg, "fetch_from_hf", False):
        model_dir = cfg.hf_model_path
        model_name = get_model_name(model_dir)
    else:
        raise ValueError("This script loads model from huggingface or wandb. Please provide a valid wanbd artifact directory path or a HF Hub path")

    return model_dir, model_name




def get_model_dir_wandb(artifact_dir:str):
    subfolders = [f.path for f in os.scandir(artifact_dir) if f.is_dir()]
    if len(subfolders) == 1:
        model_dir = subfolders[0]
    else:
        # fallback to default name
        model_dir = os.path.join(artifact_dir, "best_model_checkpoint_path_for_this_run")
    return model_dir


def get_model_name(path:str):

  return os.path.split(path)[-1]



def get_logger_inf():
    logger.remove(0)
    logger.add(sys.stderr, 
           format=("[<green>{time:MM DD, YYYY -> HH:mm:ss}]</green>>>" 
                   "<blue>{name}:{line}</blue>"
                    " | <level>{level}</level> |"
                     " <cyan>{message}</cyan>"
                   
                   ),
              )
