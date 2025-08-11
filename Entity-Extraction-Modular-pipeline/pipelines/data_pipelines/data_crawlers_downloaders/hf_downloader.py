# """
# This script although intended for dataset download only, 
# it can also be used to download other files from a huggingface repository. 

# """

from hugging_hub import hf_hub_download
from typing import Optional

def download_hf_dataset_file(
    repo_id: str,
    filename: str,
    repo_type: str = "dataset", #Set to "dataset" or "space" if downloading from a dataset or space, None or "model" if downloading from a model. Default is dataset
    local_dir: str = ".",
    revision: str = "main",
    token: Optional[str] = None
) -> str:
    """
    Download a specific file from a Hugging Face repo.

    Args:
        repo_id (str): The dataset namespace and name (e.g., "user/dataset-name", or "organisation/dataset-name").
        filename (str): The name of the file to download (e.g., "data.conll").
        repo_type (str): Repository type; only "dataset" is supported here.
        local_dir (str): Local directory to save the downloaded file.
        revision (str): Branch or tag name (default: "main").
        token (str): Optional HF token for private datasets.

    Returns:
        str: Local path to the downloaded file.
    """
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        local_dir=local_dir,
        token=token #not needed for public dataset
    )
    return local_path
