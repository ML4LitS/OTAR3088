
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
from ast import literal_eval


from datasets import Dataset, DatasetDict
import evaluate






def prepare_metrics_hf(label_list):
  metric = evaluate.load("seqeval")
  print(metric)
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
                   label_counter_wo_iob: Counter) -> Tuple(Counter, Counter):
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