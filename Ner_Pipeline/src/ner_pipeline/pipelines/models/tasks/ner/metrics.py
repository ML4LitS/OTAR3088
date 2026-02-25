"""
to move all ner metrics logic here including seqeval and nervaluate f
"""
from loguru import logger

import numpy as np
import evaluate
from seqeval.metrics import classification_report


def seqeval_metrics(label_list):
  metric = evaluate.load("seqeval")
  def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
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