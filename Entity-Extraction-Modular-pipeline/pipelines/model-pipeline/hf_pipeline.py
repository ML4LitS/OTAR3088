from collections import Counter
from functools import partial
from typing import Union, List, Dict, Tuple
from omegaconf import DictConfig
from loguru import logger

from steps.load_ner_dataset_hf import load_ner_dataset
from steps.tokenize_preprocess import tokenize_and_align
from utils.helper_functions import prepare_metrics_hf


import torch
from datasets import Dataset, DatasetDict, Sequence, Value, ClassLabel
from transformers import (AutoTokenizer, 
                          AutoModelForTokenClassification, 
                          DataCollatorForTokenClassification,
                          TrainingArguments,
                          Trainer) 






def count_entity_labels(dataset:Dataset, label_col:str) -> Counter:

  label_counter_iob = Counter()
  label_counter_wo_iob = Counter()

  for labels in dataset[label_col]:
    if isinstance(labels, list):
      label_counter_iob.update(labels)
      entity_labels_wo_iob = [label.split("-")[-1] if "-" in label else label for label in labels]
      label_counter_wo_iob.update(entity_labels_wo_iob)
    else:
      raise ValueError(f"Expected list of labels per example, got {type(labels)}")
    
  return label_counter_iob, label_counter_wo_iob



def data_loader(cfg:DictConfig) -> Union[Dataset, DatasetDict]: 
  
  train_dataset = load_ner_dataset(cfg.train_file, 
                                 file_type=cfg.file_format
                                 )
  test_dataset = load_ner_dataset(cfg.test_file, 
                                 file_type=cfg.file_format
                                 )
  val_dataset = load_ner_dataset(cfg.val_file, 
                                 file_type=cfg.file_format
                                 )
  return train_dataset, val_dataset, test_dataset



def cast_to_class_labels(dataset:Dataset, label_col:str, text_col:str, label_list:List):
    """
    Convert str labels to one-hot encoded labels.
    Args:
    dataset: huggingface Dataset object
    label_col: label column name containing ner tags in dataset
    text_col: text column name containing tokens in dataset
    label_list: List containing only unique label names(assumes IOB format) in dataset :example["B-CellLine", "B-CellType", "I-cellLine", "O", .....]

    Returns:
        A dataset object with label columns converted to integers and actual class names stored as a feature 
    
    
    """
    features = dataset.features.copy()
    features[text_col] = Sequence(Value("string"))
    features[label_col] = Sequence(ClassLabel(names=label_list, 
                                              num_classes=len(label_list)
                                              ))
    return dataset.cast(features)


def get_label2id_id2label(label_list:List) -> Tuple[Dict, Dict]:
  """
  Extracts Label2id and Id2Label mapping.

  Args:
  label_list(list): List containing only unique label names(assumes IOB format) in dataset :example["B-CellLine", "B-CellType", "I-cellLine", "O", .....]

  
  Returns:
    label2id: Dictionary of label to int mapping
    id2label: Dictionary of int to label mapping 
    Tuple[Dict, Dict] 
  
  """

  label2id = {label:i for i,label in enumerate(label_list)}
  id2label = {i:label for label,i in label2id.items()}

  return label2id, id2label



def get_model(hf_checkpoint_name:str, 
              num_labels:List, 
              label2id:Dict, 
              id2label:Dict, 
              device:str):
  
  """
  Create an instance of a token classification model. 

  Args:
  hf_checkpoint_name: model name(without url) as referenced on Huggingface hub (e.g google-bert/bert-base-uncased)
  num_labels: number of labels to predicted(Generally equal to number of unique labels in dataset). Expected to match length of label list
  label2id: Dictionary of label to int mapping
  id2label: Dictionary of int to label mapping 
  device: Type of device to train on . E.g "cpu" or "cuda"

  Returns:
    model(obj): instance of a token classification model with classifier head initiased to size of label list
  
  """
  model = AutoModelForTokenClassification.from_pretrained(hf_checkpoint_name,
                                                          num_labels=num_labels,
                                                          label2id=label2id,
                                                          id2label=id2label
                                                          )
  model.to(device)
  return model


def init_tokenizer_data_collator(hf_checkpoint_name:str):
   """
   Creates an instance of a tokenizer and DataCollator(used for dynamic padding of inputs).

   Args:
    hf_checkpoint_name(str) - same as model name(without url) as referenced on Huggingface hub (e.g google-bert/bert-base-uncased)

    Returns:
        tokenizer(obj): An instance of a tokenizer class
        data_collator: An instance of a data collator for token classficiation class
   """

   tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint_name)
   data_collator  = DataCollatorForTokenClassification(tokenizer=tokenizer)

   return tokenizer, data_collator

def hf_trainer(cfg:DictConfig, 
               wandb_run, 
               run_artifact,
               output_dir:str, 
               device:str):
    
    hf_checkpoint_name = cfg.model.model_name_or_path
    logger.info(f"Model checkpoint used for this run is: {hf_checkpoint_name}")
    wandb_run.log({"Model checkpoint used for this run is": hf_checkpoint_name})

    #get train, val, test
    train_dataset, val_dataset, test_dataset = data_loader(cfg.data)
    logger.success("Datasets loaded")
    logger.info(f"Train Dataset: {train_dataset}")

    label_col, text_col = cfg.model.label_col, cfg.model.text_col
    

    train_entity_count_iob, train_entity_count_wo_iob = count_entity_labels(train_dataset, "labels")
    val_entity_count_iob, val_entity_count_wo_iob = count_entity_labels(val_dataset, "labels")
    test_entity_count_iob, test_entity_count_wo_iob = count_entity_labels(test_dataset, "labels")


    train_labels, val_labels, test_labels = train_entity_count_iob.keys(), val_entity_count_iob.keys(), test_entity_count_iob.keys()


    unique_tags = list(set(train_labels | val_labels))
    label2id, id2label = get_label2id_id2label(unique_tags)

    wandb_run.log({
        "Text column in dataset": text_col,
        "Labels column in dataset": label_col,
        "Unique labels in dataset": unique_tags,
        "Labels count in train dataset": train_entity_count_wo_iob,
        "Labels count in val dataset": val_entity_count_wo_iob,
        "Num classes to predict": len(unique_tags)
    })
    #one-hot encode labels to integer values
    logger.info("Processing datasets using `cast_to_label` func")
    train_dataset = cast_to_class_labels(train_dataset, "labels", "words", unique_tags)
    val_dataset = cast_to_class_labels(val_dataset, "labels", "words", unique_tags)
    test_dataset = cast_to_class_labels(test_dataset, "labels", "words", unique_tags)
    logger.success("Datasets successfully processed")
    
    #init tokenizer and data collator
    logger.info("Initialising tokenizer and data collator")
    tokenizer, data_collator = init_tokenizer_data_collator(hf_checkpoint_name)

    #tokenize dataset
    # tokenized_train = train_dataset.map(tokenize_and_align, fn_kwargs={"tokenizer": tokenizer, "device": device}, batched=True, remove_columns=train_dataset.column_names)
    # tokenized_val = val_dataset.map(tokenize_and_align, fn_kwargs={"tokenizer": tokenizer, "device": device}, batched=True, remove_columns=val_dataset.column_names)
    # tokenized_test = test_dataset.map(tokenize_and_align, fn_kwargs={"tokenizer": tokenizer, "device": device}, batched=True, remove_columns=test_dataset.column_names)

    tokenize_fn = lambda batch: tokenize_and_align(batch, tokenizer=tokenizer)
    logger.info("Tokenizing datasets using `tokenize and align` func")
    tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=val_dataset.column_names)
    tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=test_dataset.column_names)

    logger.info(f"Tokenized train sample: {next(iter(tokenized_train))}")


    #init training args
    logger.info("Initialising training args")
    args = TrainingArguments(output_dir=output_dir, 
                             logging_dir=f"{output_dir}/{cfg.model.name}-{cfg.data.name}_model-trainer_logs",
                             **cfg.model.args)  
    logger.info(f"Current training argument init: {args}")
    
    logger.info("Creating model instance.....")
    model = get_model(hf_checkpoint_name, num_labels=len(unique_tags), label2id=label2id, id2label=id2label, device=device)
    logger.info(f"Model initialised as : {model}")

    #init compute metrics
    compute_metrics = prepare_metrics_hf(unique_tags)
    trainer = Trainer(model,
                           args,
                           train_dataset=tokenized_train,
                           eval_dataset=tokenized_val,
                           processing_class=tokenizer,
                           data_collator=data_collator,
                           compute_metrics=compute_metrics
                           )
    
    logger.info("Starting training.......")
    trainer.train()
    logger.success(f"Training completed. Model weights saved to: {output_dir}")
    #get best model checkpoint
    best_ckpt_path = trainer.state.best_model_checkpoint
    logger.info(f"Best model checkpoint for this run: {best_ckpt_path}")

    logger.info("Starting inference on test set.......")    
    test_results = trainer.predict(tokenized_test)
    logger.success("Inference on test set completed")
    wandb_run.log_artifact(run_artifact)
    wandb_run.log({"Test-results from run": test_results,
                   "Best model checkpoint for run": best_ckpt_path})
                                    

    #next steps- To-DO
    
    """
    1. Make test dataset optional
    2. Can configure dataset to load directly from Wandb artifact if already saved there
    """

