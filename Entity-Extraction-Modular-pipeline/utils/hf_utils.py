
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
from ast import literal_eval

from loguru import logger
from datasets import Dataset
from seqeval.metrics import classification_report
import evaluate

from transformers import (Trainer, 
                          TrainerCallback, 
                          AutoModelForTokenClassification, 
                          DataCollatorForTokenClassification, 
                          AutoTokenizer)



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

def get_model(checkpoint:str,
              num_labels:List,
              label2id:Dict,
              id2label:Dict,
              device):
  model = AutoModelForTokenClassification.from_pretrained(checkpoint,
                                                          num_labels=num_labels,
                                                          label2id=label2id,
                                                          id2label=id2label
                                                          )
  model.to(device)
  return model


def init_tokenizer_data_collator(hf_checkpoint_name):
   tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint_name)
   data_collator  = DataCollatorForTokenClassification(tokenizer=tokenizer)

   return tokenizer, data_collator


def differential_lr(base_lr, head_lr, model, num_layers):
  
  optimizer_grouped_parameters = [
    {
        'params': [p for p in model.bert.encoder.layer[:num_layers].parameters()], # Early layers
        'lr': base_lr
    },
    {
        'params': [p for p in model.bert.encoder.layer[num_layers:].parameters()], # Late layers
        'lr': base_lr * 2  
    },
    {
        'params': model.classifier.parameters(), # The classification head
        'lr': head_lr
    }
    ]
  return optimizer_grouped_parameters




def get_encoder_layers(model):
    """
    Return (backbone_name, layer_list) for common architectures.
    """
    if hasattr(model, "bert"):
        return "bert", model.bert.encoder.layer
    if hasattr(model, "roberta"):
        return "roberta", model.roberta.encoder.layer
    if hasattr(model, "distilbert"):
        return "distilbert", model.distilbert.transformer.layer
    raise ValueError("Unsupported Bert backbone. Inspect model to find encoder layers.")


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