import math
from typing import List, Dict, Tuple, Optional
import numpy as np

import torch
import evaluate

from omegaconf import DictConfig
from loguru import logger

from datasets import Dataset, DatasetDict
from transformers import (Trainer, 
                          TrainerCallback, 
                          TrainingArguments,
                          AutoModelForTokenClassification, 
                          DataCollatorForTokenClassification, 
                          AutoTokenizer)

from steps.load_ner_dataset_hf import data_loader
from steps.tokenize_preprocess import cast_to_class_labels, tokenize_and_align
from utils.hf_utils import (
                            count_entity_labels,
                            get_label2id_id2label
                                
                            )


def prepare_datasets(cfg:DictConfig, wandb_run) -> Tuple[Dataset, Dataset, List, Dict, Dict]:

  train_dataset, val_dataset = data_loader(cfg.data)
  text_col, label_col = train_dataset.column_names
  train_entity_count_iob, train_entity_count_wo_iob = count_entity_labels(train_dataset, label_col)
  val_entity_count_iob, val_entity_count_wo_iob = count_entity_labels(val_dataset, label_col)

  unique_tags = list(set(train_entity_count_iob.keys()) | set(val_entity_count_iob.keys()))
  label2id, id2label = get_label2id_id2label(unique_tags)

  train_dataset = cast_to_class_labels(train_dataset, label_col, text_col, unique_tags)
  val_dataset = cast_to_class_labels(val_dataset, label_col, text_col, unique_tags)

  if cfg.use_wandb and wandb_run is not None:
    wandb_run.log({
        "Text column in dataset": text_col,
        "Labels column in dataset": label_col,
        "Unique labels in dataset": list(unique_tags),
        "Labels count in train dataset": dict(train_entity_count_wo_iob),
        "Labels count in val dataset": dict(val_entity_count_wo_iob),
        "Train Label counts in IOB": dict(train_entity_count_iob), 
        "Validation Label counts in IOB": dict(train_entity_count_iob)
    })


  return train_dataset, val_dataset, unique_tags, label2id, id2label



def prepare_training_args(cfg:DictConfig, output_dir:str):
  hf_checkpoint_name = cfg.model.model_name_or_path
  checkpoint_size = hf_checkpoint_name.split("/")[0]
  return TrainingArguments(
      output_dir=f"{output_dir}/{checkpoint_size}",
      logging_dir=f"{output_dir}/{checkpoint_size}/{cfg.model.name}-{cfg.data.name}_logs",
      **cfg.model.args
  )



def get_model(checkpoint:str,
              num_labels:List,
              label2id:Dict,
              id2label:Dict,
              device:str):
  model = AutoModelForTokenClassification.from_pretrained(checkpoint,
                                                          num_labels=num_labels,
                                                          label2id=label2id,
                                                          id2label=id2label
                                                          )
  model.to(device)
  logger.info(f"Model initialised with {count_trainable_params(model)} trainable params")
  return model





def get_training_steps(args, train_dataset):

  """
  Calculate total training steps based on dataset size, batch size, gradient accumulation steps and number of epochs
  Args:
    args: TrainingArguments object
    train_dataset: Huggingface dataset object
  Returns:
    Total number of training steps as integer
  """
  # Effective batch size accounts for accumulation and number of devices
  effective_batch_size = (
      args.per_device_train_batch_size *
      args.gradient_accumulation_steps *
      max(1, torch.cuda.device_count())  # default 1 if no GPU
  )
  #get total num steps per epoch
  num_update_steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)

  #get total training steps
  num_training_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
  
  return num_training_steps



def custom_warmup_steps(num_training_steps, warmup_ratio=0.1):
    """Return warmup steps given a ratio (e.g., 0.1 = 10%)."""
    return int(num_training_steps * warmup_ratio)





def get_model_backbone(model):
    """
    Return (backbone_name, backbone) for common architectures.
    """
    if hasattr(model, "bert"):
        return "bert", model.bert
    if hasattr(model, "roberta"):
        return "roberta", model.roberta
    if hasattr(model, "distilbert"):
        return "distilbert", model.distilbert
    raise ValueError("Unsupported Bert backbone. Inspect model to find name of backbone.")



def get_encoder_layers(model):
  "Finds encoder layer for BERT model variants"
  model_name, backbone = get_model_backbone(model)
  if model_name == "bert":
    encoder_layer = backbone.encoder.layer
  elif model_name == "roberta":
    encoder_layer = backbone.encoder.layer
  elif model_name == "distilbert":
    encoder_layer = backbone.transformer.layer
  return encoder_layer



def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





class CustomTrainer(Trainer):
    def __init__(self, *args, id2label=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_predictions = []
        self.epoch_labels = []
        self.epoch_loss = []
        self.id2label = id2label

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels is not None and logits is not None:
            preds = logits.argmax(dim=-1)

            # Store predictions and labels in seqeval format
            for pred_seq, label_seq in zip(preds, labels):
                pred_labels = [self.id2label[p.item()] for p, l in zip(pred_seq, label_seq) if l != -100]
                true_labels = [self.id2label[l.item()] for p, l in zip(pred_seq, label_seq) if l != -100]

                self.epoch_predictions.append(pred_labels)
                self.epoch_labels.append(true_labels)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        self.epoch_loss.append(loss.item())

        return (loss, outputs) if return_outputs else loss







class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        metric = evaluate.load("seqeval")
        preds = self._trainer.epoch_predictions
        labels = self._trainer.epoch_labels
        losses = self._trainer.epoch_loss

        if preds and labels:

            train_results = metric.compute(predictions=preds, references=labels)
            mean_loss = np.mean(losses)

            logger.info("\n======== Training Metrics on Epoch End ========")
            logger.info(f"Train Loss: {mean_loss:.4f}")
            logger.info(f"Training Results:\n {train_results}")
            logger.info("=============================================")

        # Reset storage for next epoch
        self._trainer.epoch_predictions = []
        self._trainer.epoch_labels = []
        self._trainer.epoch_loss = []