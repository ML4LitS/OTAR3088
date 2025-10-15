import math
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
from ast import literal_eval

from omegaconf import DictConfig
from loguru import logger
from datasets import Dataset
from seqeval.metrics import classification_report
import evaluate

import torch
from transformers import (Trainer, 
                          TrainerCallback, 
                          AutoModelForTokenClassification, 
                          DataCollatorForTokenClassification, 
                          AutoTokenizer)

from steps.tokenize_preprocess import tokenize_and_align



def prepare_metrics_hf(label_list):
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
    report = classification_report(true_labels, true_predictions)
    logger.info(f"Eval Classification Report:\n {report}")
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
  return compute_metrics




def split_dataset(example, test_size=0.2):
  """
  Splits a dataset into train and validation sets
  Args:
    example: A huggingface dataset class
    test_size: Ratio to split dataset by
  returns:
  A huggingface dataset with train and validation splits

  """
  example = example.train_test_split(test_size=test_size)
  example["validation"] = example["test"]
  example.pop("test")
  return example



def update_counters(labels: List,
                   label_counter_iob: Counter,
                   label_counter_wo_iob: Counter) -> Tuple[Counter, Counter]:
  """
  Update counters for labels with and without IOB tags
  """
  label_counter_iob.update(labels)
  entity_labels_wo_iob = [label.split("-")[-1] if "-" in label else label for label in labels]
  label_counter_wo_iob.update(entity_labels_wo_iob)
  return label_counter_iob, label_counter_wo_iob


def count_entity_labels(dataset:Dataset, label_col:str) -> Counter:
  """
  Count instances of labels per row of Dataset
  Expects list of labels per row
  Returns: Counters of labels with and without IOB tags
  """
  label_counter_iob = Counter()
  label_counter_wo_iob = Counter()

  for labels in dataset[label_col]:
    if isinstance(labels, list):
      label_counter_iob, label_counter_wo_iob = update_counters(
         labels,
         label_counter_iob,
         label_counter_wo_iob
         )
    else:
      try:
        labels = literal_eval(labels)
        label_counter_iob, label_counter_wo_iob = update_counters(
           labels,
           label_counter_iob,
           label_counter_wo_iob
           )
      except:
        raise ValueError(f"Expected list of labels per example, got {type(labels)}")

  return label_counter_iob, label_counter_wo_iob



def get_label2id_id2label(label_list:Dict) -> Tuple[Dict, Dict]:

  label2id = {label:i for i,label in enumerate(label_list)}
  id2label = {i:label for label,i in label2id.items()}

  return label2id, id2label


def init_tokenizer_data_collator(hf_checkpoint_name):
  "Initialises tokenizer, data collator and applies tokenization function to dataset"
  tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint_name)
  data_collator  = DataCollatorForTokenClassification(tokenizer=tokenizer)
  tokenize_fn = lambda batch: tokenize_and_align(batch, tokenizer=tokenizer)

  return tokenizer, data_collator, tokenize_fn
 

def get_experiment_subfolder(cfg: DictConfig) -> str:
    """
    Construct a unique experiment subfolder path based on model type, 
    training strategy, and experiment-specific parameters.

    """
    training_strategy = cfg.training_strategy.lower()
    model_name = cfg.model.name
    data_name = cfg.data.name
    version_name = cfg.data.version_name

    # Base folder: model/strategy
    base_folder = f"{model_name}/{data_name}_{version_name}/{training_strategy}"

    if training_strategy in ["reinit_only", "reinit_llrd"]:
        experiment_subfolder = f"{base_folder}/{cfg.reinit_k_layers}K"

        #Append classifier reinitialisation flag if activated
        if getattr(cfg, "reinit_classifier", False):
            experiment_subfolder += "_with_reinit_classifier"
        else:
            experiment_subfolder += "_no_reinit_classifier"

        # Append LLRD value, if LLRD value not == 1.0, only true if training_strategy==reinit_llrd
        llrd_value = cfg.llrd
        if abs(llrd_value - 1.0) > 1e-6:
            experiment_subfolder += f"_llrd{llrd_value}"

    else:
        #For base 
        experiment_subfolder = base_folder

    return experiment_subfolder


def get_logging_params(cfg: DictConfig):
    """
    Builds dynamic logging and tracking configurations for different training strategies.
    Ensures consistency between Loguru and Weights & Biases logging directories.
    """
    strategy = cfg.training_strategy.lower()
    model_name = cfg.model.name
    lr = cfg.lr
    data_name, data_version = cfg.data.name, cfg.data.version_name
    base_log_dir = f"logs/{model_name}/{data_name}_{data_version}/{strategy}"
    base_log_filename = f"{strategy}-{data_name}_{data_version}-LR_{lr}"
    wandb_tags = [strategy, model_name, data_name, "ner", "hydra"]
    wandb_run_name = f"{model_name}-{data_name}_{data_version}-{strategy}"
    log_dir = base_log_dir
    log_filename = f"{base_log_filename}_model.log"

    if strategy == "base":
        log_filename = f"{base_log_filename}_model.log"

    elif strategy in ["reinit_only", "reinit_llrd"]:
        wandb_run_name += f"-{cfg.reinit_k_layers}K-Layers"
        wandb_tags.append(f"{cfg.reinit_k_layers}K-Layers")

        if getattr(cfg, "reinit_classifier", cfg.reinit_classifier):
            wandb_run_name += "-with_reinit_classifier"
            log_dir = f"{log_dir}/with_reinit_classifier"
            wandb_tags.append("with_reinit_classifier")
        else:
            wandb_run_name += "-no_reinit_classifier"
            log_dir = f"{log_dir}/without_reinit_classifier"
            wandb_tags.append("without_reinit_classifier")

        if strategy == "reinit_llrd" and hasattr(cfg, "llrd") and abs(cfg.llrd - 1.0) > 1e-6:
            log_filename = f"{base_log_filename}_{cfg.reinit_k_layers}K_LLRD-{cfg.llrd}_model.log"
            wandb_run_name += f"-LLRD-{cfg.llrd}"
            wandb_tags.append(f"LLRD-{cfg.llrd}")
        else: #reinit-only
            log_filename = f"{base_log_filename}_{cfg.reinit_k_layers}K_model.log"

    elif strategy == "llrd_only":
        if hasattr(cfg, "llrd") and abs(cfg.llrd - 1.0) > 1e-6:
            log_filename = f"{base_log_filename}_llrd-{cfg.llrd}_model.log"
            wandb_run_name += f"-LLRD-{cfg.llrd}"
            wandb_tags.append(f"llrd-{cfg.llrd}")


    cfg.logging.loguru.log_dir = log_dir
    cfg.logging.loguru.log_filename = log_filename
    cfg.logging.wandb.run.dir = log_dir
    cfg.logging.wandb.run.tags = wandb_tags
    #cfg.logging.wandb.run.name = wandb_run_name

    logger.info(f"Log files for this run are saved to: {cfg.logging.loguru.log_dir}/{cfg.logging.loguru.log_filename}")
    return cfg, wandb_run_name

