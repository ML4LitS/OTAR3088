import requests
from requests.exceptions import HTTPError, RequestException
import functools
import tarfile
from pathlib import Path
from typing import Optional, Union
import re
import sys
import random
import torch
from datasets import Dataset, DatasetDict, load_metric
import numpy as np
import pandas as pd
from omegaconf import DictConfig


##missing typ int and docstrings for some func
##Todo

def catch_request_errors(func):
    """
    Wrapper function to catch request errors
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPError as http_err:
            print(f"HTTP error in {func.__name__}: {http_err}")
        except RequestException as req_err:
            print(f"Request error in {func.__name__}: {req_err}")
        return None
    return wrapper



           
def extract_tar_gz_local(file_path: Path, extract_to: Optional[Path] = None):
    """
    Extracts a .tar.gz archive to the specified directory.

    Args:
        file_path (Path): Path to the .tar.gz file.
        extract_to (Optional[Path]): Directory to extract into. Defaults to a folder with the same name as the archive.
    """
    if extract_to is None:
        extract_to = file_path.with_suffix('').with_suffix('')  # Remove both .gz and .tar
    extract_to.mkdir(parents=True, exist_ok=True)

    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted {file_path.name} → {extract_to}")


def clean_text(text:str) -> str:
    """
    This function cleans a text by filtering reference patterns in text, 
    extra whitespaces, escaped latex-style formatting appearing in text body instead of predefined latex tags

    Args: 
    text(str): The text to be cleaned
    
    Returns: 
    tex(str): The cleaned text 
    
    """
   
    # Remove LaTeX-style math and formatting tags #already filtered from soup content but some still appear
    text = re.sub(r"\{.*?\}", "", text)  # Matches and removes anything inside curly braces {}
    text = re.sub(r"\\[a-zA-Z]+", "", text)  # Matches and removes characters that appears with numbers
    
    # Remove reference tags like [34] or [1,2,3]
    text = re.sub(r"\[\s*(\d+\s*(,\s*\d+\s*)*)\]", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def push_to_hub(
    dataset: Union[Dataset, DatasetDict],
    repo_name: str,
    private_repo: bool = False,
    token: str = None,
    split_name: str = None,
):
    pass




def set_seed(seed: int):
    """Ensure reproducibility across Python, NumPy, and PyTorch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# def parallel_process(df, func):
#     data = [row for _,row in df.iterrows()]
    

#     # Initialize multiprocessing pool
#     with Pool(cpu_count()) as pool:
#         results = list(tqdm(pool.imap(func, data), total=len(data), desc="Processing rows"))

#     # Convert back to DataFrame
#     return pd.DataFrame(results)


def create_output_dir(base_path:str, 
                      name:str, 
                      *, 
                      is_model:bool=True, 
                      is_datasets:bool=False,
                      is_subfolder:bool=True):
  
  """
  Creates output directory for saving model outputs or datasets.
  Parameters:
      base_path (str): Base directory to create outputs in.
      name (str): Name of the output directory (e.g model or dataset name/version).
      is_model (bool): If True, creates under 'model_outputs' if a model type folder.
      is_datasets (bool): If True, creates under 'Datasets' if dataset type folder.
      include_subfolder (bool): Whether to create under a subfolder for a model/dataset type.

  Returns:
      Path: Path object of the created directory.

  """
  if not is_model and not is_datasets:
    raise ValueError("Either `is_model` or `is_datasets` must be True.")
  if is_model and is_datasets:
    raise ValueError("Only one of `is_model` or `is_datasets` can be True at a time.")
  
  subfolder = "model_outputs" if is_model else "Datasets" 
  

  try:
    base_path = Path(base_path)
  except TypeError:
      raise ValueError("base_path must be a valid str or path")
  
  try:
    output_dir = base_path / subfolder / name if is_subfolder else base_path / name
    output_dir.mkdir(parents=True, exist_ok=True)
    
  except Exception as e:
    raise RuntimeError(f"Failed to create output directory at {output_dir}") from e
  print(f"Output directory created at {output_dir}")
  return output_dir

def setup_loguru(config:Optional[DictConfig]):
    """Setup loguru to a central directory if specified in hydra config or 
    default to current directory(useful for inference only)
    """
    log_dir = Path(config.loguru.log_dir) if config and "loguru" in config else Path.cwd()
    print(log_dir)
    log_filename = config.loguru.log_filename if config and "loguru" in config else "run.log"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_filename

    logger.remove(0)
    logger.add(
        log_path,
        format="[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] | <level>{level}</level> | <cyan>{message}</cyan>",
        mode="w",

    level=config.loguru.level
    )
    logger.success(f"Loguru initialised at: {log_path}")




def prepare_metrics_hf(label_list):
  metric = load_metric("seqeval")
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