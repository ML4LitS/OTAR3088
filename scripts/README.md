# Data Extraction Scripts

## Overview

This repository contains scripts for extracting and processing articles from EuropePMC. The workflow includes fetching articles based on search queries, extracting sentences or paragraphs, and saving the processed data.

### Context
- **Search Queries**: Curated bioentity names gathered from IntAct and ChEMBL teams.  
- **Entities of Interest**: Cell Line and Cell Type.

## Scripts

### 1. **`extract_articles.py`**
This is the main script for fetching and processing articles.
- **Functionality**:
  - Extracts articles based on provided search terms.  
  - Search terms are constructed using entity names and synonyms, e.g., `("A9-PTX10 cell line" OR "1A9PTX10 cells")`, `("76N cell line" OR "epithelial cells")`.
  - Utilizes EuropePMC API with filters for Open Access and Full Text articles only.
  - Allows extraction at **sentence** or **paragraph** levels.
  - Logs the process and handles duplicates, missing values, and retracted publications.

- **Arguments**:
  - `--data_path`: Path to a CSV file with search terms (`term`, `synonymn` columns required).
  - `--output_dir`: Directory to save the extracted data (default: `./`).
  - `--format2extract`: Specify extraction format (`sent` for sentences, `par` for paragraphs).
  - `--save_name`: Name for the output CSV file.
  - `--log_dir`: Directory for saving logs (optional; defaults to `output_dir`).

### 2. **`split2sent_par.py`**
A utility script for processing article content.
- **Functionality**:
  - Fetches XML content from EuropePMC using the PMCID.
  - Processes articles to extract:
    - Sentences using **spaCy**.
    - Paragraphs, section titles, and figure captions using **BeautifulSoup**.
  - Supports concurrent processing for faster extraction.

- **Main Functions**:
  - `get_xml`: Fetches the XML content for a given PMCID.
  - `sentencise_in_parallel`: Extracts sentences from articles concurrently.
  - `get_df`: Processes paragraphs, sections, and figure captions concurrently.

## Usage
`python extract_articles.py --data_path <path_to_csv> \
                           --output_dir <output_directory> \
                           --format2extract <sent|par> \
                           --save_name <output_file_name>`


### Example
`python extract_articles.py --data_path data/search_terms.csv \
                           --output_dir results/ \
                           --format2extract sent \
                           --save_name cell_line_sentences`


### Expected Output from script
- Extracted data is saved as a CSV file in the specified output_dir.
- Logs are saved in the log_dir or output_dir.

## Notes
The scripts prioritize efficient querying using EuropePMC's API techniques.
Ensure the input CSV is correctly formatted with term and synonymn columns.

## License
This project is open-source and available under the *** License.

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>

