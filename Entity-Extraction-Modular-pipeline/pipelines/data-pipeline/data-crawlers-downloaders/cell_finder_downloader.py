import os
import requests
from requests import HTTPError, RequestException
from pathlib import Path
from bs4 import BeautifulSoup as bs
from typing import List, Tuple
from typing_extensions import Annotated, Optional
import tarfile
from utils.helper_functions import catch_request_errors
from argparse import ArgumentParser
import sys
from pathlib import Path




BASE_URL = "https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder/"



@catch_request_errors
def get_data_archive(url:str=BASE_URL) -> Annotated[List[str], "cellfinder-data-links"]:
    response = requests.get(url)
    response.raise_for_status()
    soup = bs(response.text, "html.parser")
    links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".tar.gz")]
    links = [os.path.basename(link) for link in links]
    return links


@catch_request_errors
def download_and_extract(data_links:List[str], output_dir:str, url:str=BASE_URL):
    """the following files are available for download form the cellfinder website: 
    ['cellfinder1_brat.tar.gz',
 'cellfinder1_brat_sections.tar.gz',
 'cellfinder1_xml.tar.gz']

 In this script, we extract by default only the first file.
 The function may be used to extract any other formats or all if of interest.
 Example usage: for single link: link = [get_archive(url)[0]]
 download_and_extract(link, path)
 For all files: links = get_archive(url)
 download_and_extract(links, path)

    """
    #create file save directory if it doesn't already exits
    output_dir.mkdir(parents=True, exist_ok=True)
    for link in data_links:
        archive_url = url + link
        archive_path = output_dir / link
        extract_dir = output_dir / link.replace(".tar.gz", "")

        try:
            print(f"Downloading {link} → {archive_path}")
            with requests.get(archive_url, stream=True) as r:
                r.raise_for_status()
                with open(archive_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"Extracting {link} → {extract_dir}")

            with tarfile.open(archive_path, "r:gz") as t:
                t.extractall(path=extract_dir)

        except Exception as e:
            print(f"Error downloading or extracting {link}: {e}")


def main():
    parser = ArgumentParser(description="Download and extract CellFinder BRAT/XML datasets.")
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True,
        help="Directory to save and extract downloaded files"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all available files"
    )
    parser.add_argument(
        "--index", type=int, nargs="+", default=[0],
        help="Index(es) of files to download. Use --all to download everything."
    )
    parser.add_argument(
        "--url", type=str, default=BASE_URL,
        help="Override the base URL (default: CellFinder URL)"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("Fetching available archives...")
    available_links = get_data_archive(args.url)

    if not available_links:
        print("No archives found. Exiting.")
        return

    if args.all:
        selected_links = available_links
    else:
        selected_links = []
        for i in args.index:
            if i < len(available_links):
                selected_links.append(available_links[i])
            else:
                print(f"Warning: index {i} is out of bounds. Skipping.")

    print(f"\nSelected files: {selected_links}")
    download_and_extract(selected_links, output_dir, args.url)


if __name__ == "__main__":
    main()