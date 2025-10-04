import os
import subprocess
import shutil
from argparse import ArgumentParser
from loguru import logger

parser = ArgumentParser()
parser.add_argument("--repo_url", type=str, help="URL of the main git repository", required=True)
parser.add_argument("--subfolder_name", type=str, help="Name of the subfolder to fetch", required=True)
parser.add_argument("--branch", type=str, default="main", help="Branch to checkout")
parser.add_argument("--dest_dir", type=str, default="./", help="output directory to save clone to")
parser.add_argument("--is_unpack", action="store_true", help="Whether to unpack the subfolder contents directly into output directory")
parser.add_argument("--no_unpack", dest="is_unpack", action="store_false")
parser.set_defaults(is_unpack=True)
args = parser.parse_args()


def fetch_git_subfolder(repo_url: str, subfolder_name: str, branch: str, dest_dir: str, is_unpack: bool):
    """
    Fetch a specific subfolder from a git repository.

    Parameters:
        repo_url (str): URL of the git repository.
        subfolder_name (str): Name of the subfolder to fetch.
        branch (str): Branch to checkout.
        dest_dir (str): Directory to save the fetched subfolder.
        is_unpack (bool): Whether to unpack the subfolder contents directly into dest_dir.
    """
    temp_dir = "./temp_repo_clone"

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    # Clone the repository with no checkout
    subprocess.run(["git", "clone", "--no-checkout", repo_url, temp_dir], check=True)
    logger.info(f"Cloning into git repository {repo_url} temporarily at {temp_dir}")
    #cd into repo and init sparse checkout
    subprocess.run(["git", "-C", temp_dir, "sparse-checkout", "init", "--cone"], check=True)
    logger.info(f"initialising sparse checkout for specified subfolder: {subfolder_name}")
    #set sparse checkout to the desired subfolder
    subprocess.run(["git", "-C", temp_dir, "sparse-checkout", "set", subfolder_name], check=True)
    #checkout the specified branch
    logger.info(f"Checkout branch set as: {branch}")
    subprocess.run(["git", "-C", temp_dir, "checkout", branch], check=True)
    logger.success(f"Successfully cloned '{subfolder_name}' from '{repo_url}' on branch --> {branch}")

    src_path = os.path.join(temp_dir, subfolder_name)
    if is_unpack:
        logger.info(f"Unpacking contents of {subfolder_name} to {dest_dir}")
        # unpack contents of the subfolder to dest_dir
        for item in os.listdir(src_path):
            s = os.path.join(src_path, item)
            d = os.path.join(dest_dir, item)
            if os.path.exists(d):
                shutil.rmtree(d) if os.path.isdir(d) else os.remove(d)
            shutil.move(s, d)
        logger.success(f"Successfully unpacked contents of '{subfolder_name}' to '{dest_dir}'")
    else:
        # Keep the default strucuture of subfolder and move to dest_dir
        shutil.move(src_path, dest_dir)
        logger.success(f"Successfully cloned '{subfolder_name}' to local directory at:  '{dest_dir}'")
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    

if __name__ == "__main__":
    fetch_git_subfolder(
        repo_url=args.repo_url,
        subfolder_name=args.subfolder_name,
        branch=args.branch,
        dest_dir=args.dest_dir,
        is_unpack=args.is_unpack
    )