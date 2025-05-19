import requests
from requests.exceptions import HTTPError, RequestException
import functools
import tarfile
from pathlib import Path
from typing import Optional
import re


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
