import pandas as pd
from typing import Union
from pathlib import Path

from omegaconf import DictConfig
from loguru import logger
from datasets import Dataset, DatasetDict, load_dataset


from utils.file_parsers import read_conll
from utils.hf_utils import split_dataset


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
    return load_dataset(file_path, trust_remote_code=True)


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




def data_loader(cfg:DictConfig) -> Union[Dataset, DatasetDict]:

  file_type = cfg.file_type
  source_type = cfg.source_type
  data_prepped =  cfg.data_prepped

  if source_type == "hf":
    dataset = load_ner_dataset(cfg.hf_path, source_type=source_type) 
  elif source_type == "local":
    if data_prepped:
        train_dataset = load_ner_dataset(cfg.train_file, source_type=source_type, file_type=file_type)
        # TODO - Consider what we would do should we wish to test data
        # TODO - This would include another cfg flag, after which if true would run these sections but w test
        # test_dataset = load_ner_dataset(cfg.hf_path, source_type=source_type)
        eval_dataset = load_ner_dataset(cfg.test_file, source_type=source_type, file_type=file_type)
        return train_dataset, eval_dataset
    else:
        dataset = load_ner_dataset(cfg.data_folder, source_type=source_type, file_type=file_type)

        data_split = list(dataset.keys())
        print(data_split)
        print(len(data_split))
        if len(data_split) <= 1:
            #raise ValueError(f"Dataset must have an eval set for training, but received {data_split[0]}. Use the inference pipeline if running inference")
            # print(f"No validation set found in dataset. Auto-generating validation split using split ratio 80:20 training set")
            logger.warning(f"No validation set found in dataset. Auto-generating validation split using {cfg.test_size*100}% of training set")
            dataset = split_dataset(dataset[data_split[0]], test_size=0.2)

        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        default_column_names = train_dataset.column_names

        rename_dict = {default_column_names[0]: "words",
                        default_column_names[1]: "labels"}

        train_dataset = train_dataset.rename_columns(rename_dict)
        eval_dataset = eval_dataset.rename_columns(rename_dict)

        return train_dataset, eval_dataset