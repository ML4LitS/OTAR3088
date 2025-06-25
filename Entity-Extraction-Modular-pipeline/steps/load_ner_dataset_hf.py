import pandas as pd
from typing import Tuple, List, Union

from datasets import Dataset, DatasetDict

from utils.file_parsers import read_conll




def load_ner_dataset(
        file_path:Union[str,dict],
        file_type:str = "conll", #options ["conlll", "csv", "txt", "tsv"]
        text_col:str = "words",
        label_col:str = "labels",

) -> Union[Dataset, DatasetDict]:
    """
    Loads and converts different file formats to huggingface datasets. 
    file_path(str): Path/name of file to be load
    file_type(str): file format. Options--> Conll, csv, tsv
    text_col: name to be used for the column containing list of tokens/words
    label_col(str): name to be used for the column containing list of ner tags/labels
    

    Returns: 
        Dataset | DatasetDict: Huggingface dataset dict object
    """
    file_type = file_type.lower()

    if file_type in ["conll", "txt"]:
        return _dataset_from_conll(file_path, text_col, label_col)
    
    elif file_type in ["csv", "tsv"]:
        return __dataset__from_csv_tsv(file_path, text_col, label_col, file_type)
    
    elif file_type in ["json", "jsonl"]:
        raise NotImplementedError(f"Missing functionality. {file_type} is yet to be supported. Future updates will include this. Raise a github issue if required")
    
    else:
        raise ValueError(f"Unsupported file type: {file_type}")



def _dataset_from_conll(file_path: str, text_col: str, label_col: str) -> Dataset:
    tokens, labels = read_conll(file_path)
    return Dataset.from_dict({text_col: tokens, 
                              label_col: labels})



def __dataset__from_csv_tsv(file_path: str, text_col: str, label_col: str, file_type: str) -> Dataset:
    sep = "\t" if file_type == "tsv" else ","

    # Open file to check if file has CONLL structure
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Detect if the first line is a header
    header_line = lines[0]
    has_header = any(col.lower() in ["word", "token", "ner", "ner_tag", "ner_tags", "label", "labels"]
                     for col in header_line.strip().split(sep))

    # Drop header if present 
    data_lines = lines[1:] if has_header else lines

    # check file for CONLL-style format
    non_empty = [line for line in data_lines if line.strip()]
    two_col_lines = [line for line in non_empty if len(line.strip().split(sep)) == 2]

    #check percentage of file having CONLL structure
    if len(two_col_lines) / max(len(non_empty), 1) > 0.9:
        # Treat like CoNLL/IOB token-per-line format
        return _dataset_from_conll(file_path, text_col, label_col)

    # Load as DataFrame: specify header depending on whether file has a header
    if has_header:
        df = pd.read_csv(file_path, sep=sep)
    else:
        df = pd.read_csv(file_path, sep=sep, header=None, names=[text_col, label_col])

    # If token/label columns are stringified lists, convert them
    if isinstance(df[text_col].iloc[0], str) and df[text_col].iloc[0].startswith("["):
        df[text_col] = df[text_col].apply(eval)
        df[label_col] = df[label_col].apply(eval)

    return Dataset.from_pandas(df)




    
