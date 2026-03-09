from typing import (List, Dict,
                    Tuple, Union)
from loguru import logger

import evaluate
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import (Trainer, 
                          TrainerCallback, 
                          PreTrainedTokenizerBase, 
                          PreTrainedTokenizerFast,
                          AutoModelForTokenClassification, 
                          DataCollatorForTokenClassification, 
                          AutoTokenizer)


from ...shared.factory import count_trainable_params                         


def build_ner_model(checkpoint:str,
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


def build_ner_model_init(checkpoint: str, num_labels: int, label2id: dict, id2label: dict, device: str = "cuda"):
    """
    Returns a model initialisation function compatible with Hugging Face Trainer.
    
    Usage:
        trainer = Trainer(
            model_init=model_init_func,
            ...
        )
    """
    def init():
        model = AutoModelForTokenClassification.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        ).to(device)

        # Optional logging
        trainable_params = count_trainable_params(model)
        logger.info(f"Model initialised with {trainable_params:,} trainable parameters.")

        return model
    
    return init

def init_tokenizer_data_collator(hf_checkpoint_name: str) -> Tuple[Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast], 
                                                                    DataCollatorForTokenClassification]:
  "Initialises tokenizer, data collator and applies tokenization function to dataset"
  tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint_name)
  data_collator  = DataCollatorForTokenClassification(tokenizer=tokenizer)

  return tokenizer, data_collator



def build_boundary_weights(label2id):
    """
    Assign higher weight to B- tags, especially rare entities.
    Adjust scaling factors as needed.
    """
    weights = torch.ones(len(label2id))

    for label, idx in label2id.items():
        if label == "O":
            weights[idx] = 1.0
        elif label.startswith("B-"):
            if label in ["Tissue", "CellType"]:
                weights[idx] = 4.0
            else:
                weights[idx] = 3.0
        elif label.startswith("I-"):
            weights[idx] = 2.0

    return weights


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



class WeightedCustomTrainer(Trainer):
    def __init__(self, *args, id2label=None, label2id=None, swa_start_ratio=0.75, **kwargs):
        super().__init__(*args, **kwargs)

        self.id2label = id2label
        self.label2id = label2id

        self.epoch_predictions = []
        self.epoch_labels = []
        self.epoch_loss = []

        # ----- Boundary-aware class weights -----
        class_weights = build_boundary_weights(label2id)
        self.class_weights = class_weights.to(self.args.device)

        self.loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=-100
        )

        # ----- SWA configuration -----
        self.swa_model = None
        self.swa_scheduler = None
        self.swa_start_ratio = swa_start_ratio
        self.swa_start_epoch = int(self.args.num_train_epochs * swa_start_ratio)

    # --------------------------------------------------
    # Weighted CrossEntropy
    # --------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = None
        if labels is not None:
            # Flatten
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)

            loss = self.loss_fct(logits, labels)

        # ----- Store seqeval-formatted predictions -----
        if labels is not None:
            preds = torch.argmax(outputs.logits, dim=-1)

            for pred_seq, label_seq in zip(preds, inputs["labels"]):
                pred_labels = [
                    self.id2label[p.item()]
                    for p, l in zip(pred_seq, label_seq)
                    if l != -100
                ]
                true_labels = [
                    self.id2label[l.item()]
                    for p, l in zip(pred_seq, label_seq)
                    if l != -100
                ]

                self.epoch_predictions.append(pred_labels)
                self.epoch_labels.append(true_labels)

        if loss is not None:
            self.epoch_loss.append(loss.item())

        return (loss, outputs) if return_outputs else loss

    # --------------------------------------------------
    # Create optimizer + SWA
    # --------------------------------------------------
    def create_optimizer(self):
        super().create_optimizer()
        if self.optimizer is None:
            return

        # Initialize SWA model after optimizer exists
        self.swa_model = AveragedModel(self.model)

        self.swa_scheduler = SWALR(
            self.optimizer,
            swa_lr=5e-6,
            anneal_strategy="cos",
            anneal_epochs=1
        )

    # --------------------------------------------------
    # Training step with SWA update
    # --------------------------------------------------
    def training_step(self, model, inputs,  *args, **kwargs):
        loss = super().training_step(model, inputs,  *args, **kwargs)

        # Update SWA after threshold epoch
        if self.state.epoch is not None and self.state.epoch >= self.swa_start_epoch:
            self.swa_model.update_parameters(model)
            self.swa_scheduler.step()

        return loss

    # --------------------------------------------------
    # After training, swap to SWA weights
    # --------------------------------------------------
    def train(self, *args, **kwargs):
        output = super().train(*args, **kwargs)

        if self.swa_model is not None:
            print("Updating BatchNorm statistics for SWA model...")
            update_bn(self.get_train_dataloader(), self.swa_model)

            print("Loading SWA averaged weights...")
            self.model.load_state_dict(self.swa_model.module.state_dict())

        return output