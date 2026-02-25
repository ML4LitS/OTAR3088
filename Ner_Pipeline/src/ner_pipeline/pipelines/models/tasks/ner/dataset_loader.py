
from ast import literal_eval
from collections import Counter
from typing import (
                    List, Tuple,
                    Dict, Union
                        )
from pathlib import Path

from dataclasses import dataclass


from omegaconf import DictConfig
from hydra.utils import to_absolute_path

import pandas as pd
from datasets import (
                      Dataset, DatasetDict, 
                      Sequence, Value, ClassLabel,
                      load_dataset
                      )

from loguru import logger
from wandb.sdk.wandb_run import Run as WandbRun
from ner_pipeline.utils.common import set_seed
from ner_pipeline.utils.io.readers import read_conll

def cast_to_class_labels(dataset:Dataset, label_col:str, text_col:str, unique_tags:List):
    """
    Casts dataset columns to int, primarily for classification tasks
    (e.g token classification).

    This function updates the dataset features by setting the text column to a 
    sequence of strings and the label column to a sequence of ClassLabels based 
    on the provided unique tags. It then returns the dataset with these new 
    feature schemas applied.

    Args:
        dataset (Dataset): The Hugging Face dataset to be transformed.
        label_col (str): The name of the column containing the labels/tags.
        text_col (str): The name of the column containing the input text.
        unique_tags (List[str]): A list of unique strings representing the 
            class names for the ClassLabel feature.

    Returns:
        Dataset: A new dataset object with updated feature types.
    """    
    features = dataset.features.copy()
    features[text_col] = Sequence(Value("string"))
    features[label_col] = Sequence(ClassLabel(names=unique_tags,
                                              num_classes=len(unique_tags)
                                              ))
    return dataset.cast(features, load_from_cache_file=True)


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


class PrepareNerDatasets:
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
        self.source_type = self.cfg.data.source_type.lower()

    def _validate_source_type(self):
        allowed = {"hf", "local"}
        if not hasattr(self.cfg.data, "source_type"):
            raise ValueError("Source type missing from data config.\n"
                        "Source type is required for loading dataset using the appropraite method.\n"
                        "Use one of `local` or `hf` to specify.")
        source_type = self.cfg.data.source_type.lower()
        if source_type not in allowed:
            raise ValueError(f"Invalid source_type. Supported `source_type` are:  `{allowed}`")
        if source_type == "hf" and not hasattr(self.cfg.data, "hf_path"):
            raise ValueError("HuggingFace path is required when `source_type`==`hf`.\n "
                            "Example hf_path format: `OTAR3088/CeLLate1.0`"
                            )
        if source_type == "local" and not hasattr(self.cfg.data, "data_folder"):
            raise ValueError("Local path is required when `source_type`==`local`.\n"
                            "Format==/absolute/path/to/folder/in/local/")

    def _validate_hf_col_names(self, dataset):
        dataset_split = list(dataset.keys())
        if "train" not in dataset_split:
            raise ValueError("No training split found in dataset")
        known_val_headers = ["validation", "val", "dev", "eval"]
        if not any(header in dataset_split for header in known_val_headers):
            raise ValueError("No validation split found in dataset. Inspect your dataset and rename columns if possible.\n"
                        f"Common validation headers used in HF datasets are: {known_headers}")
        else:
            train_dataset = dataset["train"]
            if "validation" in dataset_split:
                eval_dataset = dataset["validation"]
            elif "val" in dataset_split:
                eval_dataset = dataset["val"]
            elif "dev" in dataset_split:
                eval_dataset = dataset["dev"]
            elif "eval" in dataset_split:
                eval_dataset = dataset["eval"]
        
            else:
                logger.warning("Known validation headers not found in dataset.\n"
                            "Using split name `test` as validation instead")
                eval_dataset = dataset["test"]
        return train_dataset, eval_dataset


    def _require_prepared(self):
        if not hasattr(self, "_dataset_artifact"):
            raise RuntimeError(
                "Datasets have not been prepared yet. "
                "Call `prepare()` before accessing dataset properties."
            )
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
        self._require_prepared()
        return self._dataset_artifact.train_dataset

    @property
    def eval_dataset(self):
        self._require_prepared()
        return self._dataset_artifact.eval_dataset

    @property
    def unique_tags(self):
        self._require_prepared()
        return self._dataset_artifact.unique_tags

    @property
    def label2id(self):
        self._require_prepared()
        return self._dataset_artifact.label2id

    @property
    def id2label(self):
        self._require_prepared()
        return self._dataset_artifact.id2label

      
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
        if len(default_column_names) > 2:
            raise ValueError("Expected exactly 2 columns(`tokens`, `tags`) for Ner Dataset"
                            "Inspect your dataset and drop unnecessary columns")
        rename_dict = {default_column_names[0]: "words",
                    default_column_names[1]: "labels"}
        train_dataset = train_dataset.rename_columns(rename_dict)
        eval_dataset = eval_dataset.rename_columns(rename_dict)
        return train_dataset, eval_dataset


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
        "method still needs a refactor to replace load_ner_dataset"
        if self.source_type == "hf":
            dataset_path = self.cfg.data.hf_path
            dataset = load_ner_dataset(dataset_path)
        elif self.cfg.source_type.lower() == "local":
            dataset_path = to_absolute_path(self.cfg.data.data_folder)
            dataset = load_ner_dataset(dataset_path, source_type=self.source_type,
                                    file_type=self.cfg.file_type)
        
        data_split = list(dataset.keys())
        logger.info(f"This dataset has {len(data_split)} split(s)")
        logger.info(f"Splits found in dataset are: {data_split}")
        if len(data_split) <= 1:
            if getattr(self.cfg.data, "test_size", None):
                logger.warning(f"No validation set found in dataset. Auto-generating validation split using {self.cfg.data.test_size*100}% of training set")
            else:
                logger.warning(f"No validation set found in dataset and no split size specified. Auto-generating validation split using default 20% of training set")
            test_size = getattr(self.cfg.data, "test_size", 0.2)
            dataset = split_dataset(dataset[data_split[0]], test_size=test_size)
            train_dataset, eval_dataset = dataset["train"], dataset["validation"]
        else:
            train_dataset, eval_dataset = self._validate_hf_col_names(dataset)
        
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
        if hasattr(self, "_dataset_artifact"):
            return self._dataset_artifact

        set_seed(self.cfg.seed)
        train_dataset, eval_dataset = self.load()
        text_col, label_col = "words", "labels"
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


class NerDatasetLoader:
    pass

#to refactor `load_ner_dataset` as `NerDatasetLoader`
def load_ner_dataset(
    file_path: Union[str, Path],
    source_type: str = "hf",  # ["hf", "local"] #loaded from config when added to pipeline
    file_type: Union[str, None] = None,  # ["conll", "csv", "tsv", "txt"]
    text_col: str = "words",
    label_col: str = "labels",
) -> Union[Dataset, DatasetDict]:
    """
    Loads NER data and returns a HuggingFace Dataset or DatasetDict object.

    Args:
        file_path: File path or HF dataset name
        source_type: One of ["hf", "local"]
        file_type: Required if source_type is "local". One of ["conll", "csv", "tsv", "txt"]
        text_col: Column name for tokens
        label_col: Column name for labels

    Returns:
        A HuggingFace Dataset or DatasetDict
    """
    if source_type not in {"hf", "local"}:
        raise ValueError("source_type must be one of ['hf', 'local']")

    if source_type == "hf":
        return _load_from_hf(file_path)

    elif source_type == "local":
        if file_type is None:
            raise ValueError("file_type must be specified when source_type is 'local'")

        file_type = file_type.lower()

        if file_type in {"conll", "txt"}:
            return _load_from_conll(file_path, text_col, label_col)

        if file_type in {"csv", "tsv"}:
            return _load_from_csv_tsv(file_path, text_col, label_col, file_type)

        if file_type in {"json", "jsonl"}:
            raise NotImplementedError(f"{file_type} format not yet supported.")

        raise ValueError(f"Unsupported file_type: {file_type}")


def _load_from_hf(file_path: str) -> DatasetDict:
    """
    Load a Hugging Face dataset (local or remote).
    """
    return load_dataset(file_path, trust_remote_code=True, download_mode="force_redownload")


def _load_from_conll(file_path: str, text_col: str, label_col: str) -> Dataset:
    """
    Load dataset from CoNLL format.
    """
    tokens, labels = read_conll(file_path)
    return Dataset.from_dict({text_col: tokens, label_col: labels})


def _load_from_csv_tsv(file_path: str, text_col: str, label_col: str, file_type: str) -> Dataset:
    """
    Load from CSV/TSV or auto-detect CoNLL format in those files.
    """
    sep = "\t" if file_type == "tsv" else ","

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    has_header = _has_text_label_header(lines[0], sep)
    data_lines = lines[1:] if has_header else lines

    if _looks_like_conll(data_lines, sep):
        return _load_from_conll(file_path, text_col, label_col)

    df = pd.read_csv(file_path, sep=sep, header=0 if has_header else None)

    if not has_header:
        df.columns = [text_col, label_col]

    if isinstance(df[text_col].iloc[0], str) and df[text_col].iloc[0].startswith("["):
        df[text_col] = df[text_col].apply(eval)
        df[label_col] = df[label_col].apply(eval)

    return Dataset.from_pandas(df)


def _has_text_label_header(header_line: str, sep: str) -> bool:
    known_headers = {"word", "token", "words", "tokens", "ner", "ner_tag", "ner_tags", "label", "labels"}
    return any(col.lower() in known_headers for col in header_line.strip().split(sep))


def _looks_like_conll(data_lines: list[str], sep: str, threshold: float = 0.9) -> bool:
    non_empty = [line for line in data_lines if line.strip()]
    two_col_lines = [line for line in non_empty if len(line.strip().split(sep)) == 2]
    return (len(two_col_lines) / max(len(non_empty), 1)) > threshold



