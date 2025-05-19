from typing import List, Tuple, Dict, Union
from pathlib import Path
import random
import glob

from utils.file_parsers import load_brat
from steps.tokenize_preprocess import tokenize_with_offsets, label_tokens_with_iob
from utils.file_writers import write_to_conll
from utils.helper_functions import clean_text

def process_single_file(file_id: str, input_dir: Path):
    """
    Process one BRAT-annotated document and return labeled tokenized sentences.
    """
    txt_path = input_dir / f"{file_id}.txt"
    dataset = load_brat(txt_path)
    if not dataset:
        return None
    text, entities = dataset[0]["text"], dataset[0]["entities"]
    text = clean_text(text)
    tokenized = tokenize_with_offsets(text)
    labeled = label_tokens_with_iob(tokenized, entities)
    return labeled

def _process_split(file_ids: List[str], input_dir: Path, output_dir: Path, split_name: str):
    """
    Process all files in a list, e.g train_splits: [1.txt, 2.txt,.......] and write them to disk.
    """
    for file_id in file_ids:
        labeled_sentences = process_single_file(file_id, input_dir)
        if labeled_sentences:
            write_to_conll(labeled_sentences, output_dir, split_name)

def data_splitter(path: str, train_ratio: float = 0.6, seed: int = 42) -> Dict[str, List[str]]:
    path = Path(path)
    all_files = list(path.glob("*.txt"))
    random.seed(seed)
    random.shuffle(all_files)
    base_filenames = [f.stem for f in all_files]

    total_files = len(base_filenames)
    train_size = int(total_files * train_ratio)
    val_test_size = (total_files - train_size) // 2

    return {
        "train": base_filenames[:train_size],
        "val": base_filenames[train_size:train_size + val_test_size],
        "test": base_filenames[train_size + val_test_size:]
    }

def process_dataset(split_dict: Dict[str, List[str]], input_dir: Union[str, Path], output_dir: Union[str, Path]):
    """
    Process all data splits (train, test, val) using BRAT-to-CoNLL pipeline.
    Assuming input is a dict with split names as keys and files as values. 
    Example: {"train":[train_files......],
    "test":[test_files......]
    "val":[val_files......]
    }
    Only used to preprocess cell-finder dataset in our pipeline
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for split_name, file_ids in split_dict.items():
        print(f"Processing {split_name} with {len(file_ids)} files...")
        _process_split(file_ids, input_dir, output_dir, split_name)
