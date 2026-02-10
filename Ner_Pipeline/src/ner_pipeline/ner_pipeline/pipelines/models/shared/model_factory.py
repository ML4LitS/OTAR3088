


from typing import Dict, Tuple, List

from omegaconf import DictConfig
from loguru import logger

import numpy as np
import evaluate
import nervaluate

from transformers import (Trainer, 
                          TrainerCallback, 
                          TrainingArguments,
                          AutoModelForTokenClassification, 
                          DataCollatorForTokenClassification, 
                          AutoTokenizer)



def prepare_training_args(cfg:DictConfig, output_dir:str):
  hf_checkpoint_name = cfg.model.model_name_or_path
  #checkpoint_size = hf_checkpoint_name.split("/")[0]
  checkpoint_name = get_checkpoint_name(hf_checkpoint_name)
  report_to = "wandb" if cfg.use_wandb else "none"
  return TrainingArguments(
      output_dir=f"{output_dir}/{checkpoint_name}",
      logging_dir=f"{output_dir}/{checkpoint_name}/logs",
      report_to = report_to,
      **cfg.model.args
  )



def get_label2id_id2label(label_list:Dict) -> Tuple[Dict, Dict]:

  label2id = {label:i for i,label in enumerate(label_list)}
  id2label = {i:label for label,i in label2id.items()}

  return label2id, id2label


def init_tokenizer_data_collator(hf_checkpoint_name):
  "Initialises tokenizer, data collator"
  tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint_name)
  data_collator  = DataCollatorForTokenClassification(tokenizer=tokenizer)

  return tokenizer, data_collator


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


def model_init(checkpoint: str, num_labels: int, label2id: dict, id2label: dict, device: str = "cuda"):
    """
    Returns a model initialization function compatible with Hugging Face Trainer.
    
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


# class WeightedLossTrainer(CustomTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.get("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits

#         # flatten
#         loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device), ignore_index=-100)
#         loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

#         return (loss, outputs) if return_outputs else loss


# def compute_class_weights(train_dataset, model):
#   label_counts = Counter()
#   for ex in train_dataset:
#       for l in ex["labels"]:
#           if l != -100: label_counts[l] += 1

#   total = sum(label_counts.values())
#   num_labels = model.config.num_labels
#   class_weights = torch.ones(num_labels)
#   for i in range(num_labels):
#       # use inverse frequency, small smoothing
#       freq = label_counts.get(i, 1)
#       class_weights[i] = total / (num_labels * freq)
#   # clamp extreme values
#   class_weights = torch.clamp(class_weights, 0.1, 10.0)
#   return class_weights