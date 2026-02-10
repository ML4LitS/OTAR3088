
from ast import literal_eval
from typing import (
                    List, Tuple,
                    Dict,
                        )
from collections import Counter

from dataclasses import dataclass
from omegaconf import DictConfig

from datasets import (
                      Dataset, DatasetDict, 
                      Sequence, Value, ClassLabel
                      )

from wandb.sdk.wandb_run import Run as WandbRun



def cast_to_class_labels(dataset:Dataset, label_col:str, text_col:str, unique_tags:List):
    features = dataset.features.copy()
    features[text_col] = Sequence(Value("string"))
    features[label_col] = Sequence(ClassLabel(names=unique_tags,
                                              num_classes=len(unique_tags)
                                              ))
    return dataset.cast(features, load_from_cache_file=False)



def split_dataset(example, test_size=0.2):
  """
  Splits a dataset into train and validation sets
  Args:
    example: A huggingface dataset class
    test_size: Ratio to split dataset by
  returns:
  A huggingface dataset with train and validation splits

  """
  example = example.train_test_split(test_size=test_size, seed=42)
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


@dataclass
class DatasetArtifact:
  """
  Container object holding prepared datasets and label metadata
    required for downstream model training.

    Attributes
    ----------
    train_dataset : Dataset
        Token-level annotated training dataset.
    eval_dataset : Dataset
        Token-level annotated validation dataset.
    unique_tags : List[str]
        List of unique entity labels present across train and validation splits.
    label2id : Dict[str, int]
        Mapping from string labels to integer IDs.
    id2label : Dict[int, str]
        Reverse mapping from integer IDs to string labels.
  """
  train_dataset: Dataset
  eval_dataset: Dataset
  unique_tags: List[str]
  label2id: Dict[str, int]
  id2label: Dict[int, str]


class PrepareDatasets:
  """
    Utility class responsible for loading, validating, filtering,
    and normalising NER datasets for model training.

    This class supports loading datasets from either HuggingFace Hub
    or local filesystem sources and prepares them into a standardised
    format suitable for HuggingFace Trainer-based pipelines.
  """

  def __init__(self, cfg: DictConfig, wandb_run: WandbRun = None):
    """
    Initialise the dataset preparation utility.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing dataset source information,
        preprocessing options, and runtime settings.
    wandb_run : wandb.sdk.wandb_run.Run, optional
        Active Weights & Biases run for logging dataset statistics.
    """
    self.cfg = cfg
    self.wandb_run = wandb_run
    self._validate_source_type()

  def _validate_source_type(self):
    allowed = {"hf", "local"}
    if not hasattr(self.cfg, "source_type"):
      raise ValueError("Source type missing from data config.\n"
                      "Source type is required for loading dataset using the appropraite method.\n"
                      "Use one of `local` or `hf` to specify.")
    source_type = self.cfg.source_type.lower()
    if source_type not in allowed:
      raise ValueError(f"Invalid source_type. Supported `source_type` are:  `{allowed}`")
    if source_type == "hf" and not hasattr(self.cfg, "hf_path"):
      raise ValueError("HuggingFace path is required when `source_type`==`hf`.\n "
                        "Example hf_path format: `OTAR3088/CeLLate1.0`"
                        )
    if source_type == "local" and not hasattr(self.cfg, "data_folder"):
      raise ValueError("Local path is required when `source_type`==`local`.\n"
                        "Format==/absolute/path/to/folder/in/local/")

  def _require_prepared(self):
    if not hasattr(self, "_dataset_artifact"):
        raise RuntimeError(
            "Datasets have not been prepared yet. "
            "Call `prepare()` before accessing dataset properties."
        )

      
  def _normalise_col_name(self, train_dataset: Dataset, eval_dataset: Dataset) -> Tuple[Dataset, Dataset]:
    """
    Normalises dataset column names to a standard schema.

    Renames dataset columns to `words` and `labels` to ensure
    compatibility with downstream NER processing utilities.
    N.B Assumes dataset has only 2 columns, expected to be Ner tokens 
        and tags columns

    Parameters
    ----------
    train_dataset : Dataset
        Training dataset with original column names.
    eval_dataset : Dataset
        Validation dataset with original column names.

    Returns
    -------
    Tuple[Dataset, Dataset]
        Datasets with normalized column names.
    """
    default_column_names = train_dataset.column_names
    rename_dict = {default_column_names[0]: "words",
                default_column_names[1]: "labels"}
    train_dataset = train_dataset.rename_columns(rename_dict)
    eval_dataset = eval_dataset.rename_columns(rename_dict)
    return train_dataset, eval_dataset

  @property
  def train_ent_iob(self):
    self._require_prepared()
    return self._train_ent_iob
  @property
  def eval_ent_iob(self):
    self._require_prepared()
    return self._eval_ent_iob
  
  @property
  def train_ent_wo_iob(self):
    self._require_prepared()
    return self._train_ent_wo_iob

  @property
  def eval_ent_wo_iob(self):
    self._require_prepared()
    return self._eval_ent_wo_iob


  @property
  def dataset_artifact(self):
    self._require_prepared()
    return self._dataset_artifact

  @property
  def train_dataset(self):
    return self._dataset_artifact.train_dataset

  @property
  def eval_dataset(self):
    return self._dataset_artifact.eval_dataset

  @property
  def unique_tags(self):
    return self._dataset_artifact.unique_tags

  @property
  def label2id(self):
    return self._dataset_artifact.label2id

  @property
  def id2label(self):
    return self._dataset_artifact.id2label


  def load(self) -> Tuple[Dataset, Dataset]:
    """
    Load raw datasets from the configured source and ensure
    train/validation splits are available.

    If no validation split exists, a validation split is automatically
    generated from the training data.

    Returns
    -------
    Tuple[Dataset, Dataset]
        Training and validation datasets.
    """
    if self.cfg.source_type.lower() == "hf":
      dataset_path = self.cfg.hf_path
      dataset = load_ner_dataset(dataset_path, source_type=self.cfg.source_type.lower()) 
    elif self.cfg.source_type.lower() == "local":
      dataset_path = to_absolute_path(self.cfg.data_folder)
      dataset = load_ner_dataset(dataset_path, source_type=self.cfg.source_type.lower(), file_type=self.cfg.file_type)
    
    data_split = list(dataset.keys())
    logger.info(f"This dataset has {len(data_split)} split(s)")
    logger.info(f"Splits found in dataset are: {data_split}")
    if len(data_split) <= 1:
      if getattr(self.cfg, "test_size", None):
        logger.warning(f"No validation set found in dataset. Auto-generating validation split using {self.cfg.test_size*100}% of training set")
      else:
        logger.warning(f"No validation set found in dataset and no split size specified. Auto-generating validation split using default 20% of training set")
      test_size = getattr(self.cfg, "test_size", 0.2)
      dataset = split_dataset(dataset[data_split[0]], test_size=test_size)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    train_dataset, eval_dataset = self._normalise_col_name(train_dataset, eval_dataset)
    return train_dataset, eval_dataset 


  def prepare(self) -> DatasetArtifact:
    """
    Execute the full dataset preparation pipeline.

    This includes loading datasets, filtering empty-label examples,
    computing label statistics, generating label mappings, normalising
    column names, and optionally logging dataset metadata to Weights & Biases.

    Returns
    -------
    DatasetKwargs
        Prepared datasets and associated label metadata ready for
        downstream model training.
    """
    set_seed(self.cfg.seed)
    train_dataset, eval_dataset = self.load()
    text_col, label_col = train_dataset.column_names
    train_ent_iob, train_ent_wo_iob = count_entity_labels(train_dataset, label_col)
    eval_ent_iob, eval_ent_wo_iob = count_entity_labels(eval_dataset, label_col)
    logger.info(f"Train entity counts before filtering (without IOB): \n{train_ent_wo_iob}")
    logger.info(f"Validation entity counts before filtering(without IOB): \n{eval_ent_wo_iob}")

    #filter rows where all_labels=='O' from dataset
    train_dataset = train_dataset.filter(lambda x: set(x[label_col]) != {"O"})
    eval_dataset = eval_dataset.filter(lambda x: set(x[label_col]) != {"O"})

    #fetch label count after filtering 
    self._train_ent_iob, self._train_ent_wo_iob = count_entity_labels(train_dataset, label_col)
    self._eval_ent_iob, self._eval_ent_wo_iob = count_entity_labels(eval_dataset, label_col)
    logger.info(f"Train entity counts after filtering (without IOB): \n{self._train_ent_wo_iob}")
    logger.info(f"Validation entity counts after filtering(without IOB): \n{self._eval_ent_wo_iob}")

    unique_tags = list(set(self._train_ent_iob.keys()) | set(self._eval_ent_iob.keys()))
    label2id, id2label = get_label2id_id2label(unique_tags)

    train_dataset = cast_to_class_labels(train_dataset, label_col, text_col, unique_tags)
    eval_dataset = cast_to_class_labels(eval_dataset, label_col, text_col, unique_tags)

    if getattr(self.cfg, "use_wandb", False) and self.wandb_run is not None:
        self.wandb_run.log({
            "Text column in dataset": text_col,
            "Labels column in dataset": label_col,
            "Unique labels in dataset": list(unique_tags),
            "Labels count in train dataset": dict(self._train_ent_wo_iob),
            "Labels count in val dataset": dict(self._eval_ent_wo_iob),
            "Train Label counts in IOB": dict(self._train_ent_iob), 
            "Validation Label counts in IOB": dict(self._eval_ent_iob)
        })
    
    self._dataset_artifact = DatasetArtifact(
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  unique_tags=unique_tags,
                  label2id=label2id,
                  id2label=id2label
              )
    return self._dataset_artifact

