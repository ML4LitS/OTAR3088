# End-to-End Entity Extraction Modular Pipeline

This repository hosts an end-to-end modular and scalable entity extraction pipeline. The primary goal is to provide a flexible and modular framework that supports various data formats, and modeling frameworks.  It is built with a focus on plug-in modularity, version control, reproducibility, and easy evaluation.
This pipeline has been developed for the OTAR3088 project initiative focused on automated knowledge extraction from scientific and biomedical literature. The core objective is to extract and classify named entities using supervised machine learning and natural language processing techniques. 

## Table of Contents
- [Project Information](#project-information)
- [Current Functionalities](#current-functionalities)
- [Information for flair model pipeline](#information-for-flair-model-pipeline)
- [Information for Huggingface pipeline](#information-for-huggingface-pipeline)
- [Information on Configuration Management with Hydra](#information-on-configuration-management-with-hydra)
- [Experiment Tracking With Weight and Bias](#experiment-tracking-with-weight-and-bias)
- [Preprocessing BRAT data](#preprocessing-brat-data)
- [Getting started](#getting-started)
- [Contributing](#contributing)
- [License](#license)


## Project Information





## Current Functionalities
- **Integrated Model Support**: Integration for training and fine-tuning sequence labelling models using the [Flair](https://flairnlp.github.io/) framework and [huggingface transformers](https://huggingface.co/docs/transformers/en/index).

- **Data Input flexibility:**

   - CoNLL-formatted `.txt` or `.conll`

   - CSV/TSV (token-label style or standard dataframe with token/label columns)

   - BRAT `.txt` + `.ann` format (conversion supported)

- **Preprocessing Tools:**

    - Converts BRAT files to CoNLL format via script
    - Reads Brat-type files
    - Reads CoNLL type files
    - Parse `CoNLL/.tx`t, `.csv/.tsv` to huggingface `dataset dict`

- **Hydra-Based Configuration**: Utilises [Hydra](https://hydra.cc/docs/intro/) for flexible and organised configuration management, with structured settings for flexible data loading, model settings, logging, and tracking. 


- **Experiment Tracking:**

  - Integrated with [Weights & Biases (wandb)]()

  - Loguru for local logs

-** Evaluation:**

  - Precision, Recall, F1, and Accuracy via [seqeval]()





## Information for flair model pipeline

Flair models expect all input data (training, development/validation, and test sets) to be saved as individual `.txt` files within a designated folder. Each file should contain data in a format compatible with Flair's ColumnCorpus (typically CoNLL-like). Example folder structure is shown below: 
```
flair_dataset/
    ├── train.txt
    ├── val.txt
    └── test.txt
```
Each file should be formatted like: 
``` text
Word1    B-Gene
Word2    O
...
WordN    I-Protein
...
WordA    I-CellType
WordB    O

```


## Information for HuggingFace pipeline

The current huggingface finetunning wrapper supports loading data in `txt/conll` or `csv/tsv formats`. `csv/tsv` files are assumed to be in one of two structures:

- CoNLL-style: Two columns (token and label) separated by a delimiter, with blank lines indicating sentence boundaries(same as used for Flair model and shown above).

- Standard DataFrame format: Each row represents a sample, with one column containing a list of tokens and another containing a corresponding list of labels for the entire sentence/example.
  Example structure for such files:
  ``` text
  tokens,labels
    "['Neural','cells','can','be','derived','from','hESCs','either','by','direct','enrichment',.......]", "['B-CellType','I-CellType','O','O','O','O','B-CellType','O','O','O','O',.....]"
    ```

    It is currently assumed that train, test, and validation (or dev) splits are present in the input data directory. Future improvements will include        support for optional test sets and direct loading of datasets from the Hugging Face Hub.


## Information on Configuration Management with Hydra

The pipeline extensively uses Hydra for flexible and organised configuration management.

The primary configuration directory (`config/`) is structured into the following sub-directories:

- `data/`: This directory contains configurations for different datasets. You can define specific parameters(e.g., paths, format details, splits) for each dataset, for example:

 - `dataset_name.yaml`
   ``` yaml
   name: your_dataset_name
   version_name: your_dataset_name-v1 #optional 
   data_folder: "path/to/folder/${.version_name}" #used for model trainining only. Note that path to folder is relative to where the main config folder(config/) sits.

    #data preprocessing(optional)
    seed: 42
    train_ratio: 0.6
    input_dir: path/to/folder/raw_brat_folder #used for converting brat to conll in our pipeline
    
    #for hf
    file_format: "txt"
    train_file: "${.data_folder}/train.${.file_format}"
    test_file: "${.data_folder}/test.${.file_format}"
    val_file: "${.data_folder}/val.${.file_format}"
   
   ```

- `logging/`: This directory contains configurations for various logging backends.
  Example:
  `logging/wandb.yaml`
  ``` yaml
  wandb:
      project: your_project_name 
      dir: logs/${model.name} #custom logging name
      entity: your_wandb_username                    
      tags: ["${model.name}", "${data.name}", "ner", "hydra"]
  ```

- `models/`: This directory holds model-specific configurations (Flair, Hugging Face), including hyperparameters and architecture settings.

This structure allows for easy modification of experimental variables and promotes reproducibility. Defaults are set via the central config file `common_config.yaml`
 
 ### Overriding Configuration Parameters
Hydra allows you to override any configuration parameter directly from the command line. For example:

``` bash 
python main.py model=huggingface model.model_name_or_path=bert-base-cased data=my_custom_dataset
```

- `<model_name>`: Corresponds to a `.yaml` file in conf/models/ (e.g., flair, huggingface).

- `<dataset_name>`: Corresponds to a `.yaml` file in conf/data/ (e.g., my_dataset).

- `<logger_name`>: Corresponds to a `.yaml` file in conf/logging/ (e.g., wandb, loguru).

This setup allows hydra parse the current configuration to the script for running.

Example training a flair model:
``` bash
python main.py model=flair data=my_flair_data model.embeddings.model="bert-base-uncased"
```
See [Flair documentation](https://flairnlp.github.io/docs/tutorial-embeddings/transformer-embeddings) for other supported embeddings type. 


Example training a huggingface model:

``` bash
python main.py model=hf data=my_flair_data model.model_name_or_path="xlm-roberta-base" lr=2e-3
```



## Experiment Tracking with Weight and Bias

- The pipeline supports various logging and experiment tracking with Weights & Biases (WandB). Thus, the pipeline will automatically log training metrics, model checkpoints, and configuration details to your WandB project. Ensure your `WANDB_TOKEN` environment variable is set.
  To do this, you can add the following to a .env file:

``` env
WANDB_TOKEN="your_wandb_api_key_here"
```
This is then loaded in the main script using [python-dotenv](https://pypi.org/project/python-dotenv/). 
``` python
from dotenv import load_dotenv
load_dotenv()  # Optionally pass the path to your .env file

#Then import to script as:
import os
WANB_TOKEN = os.environ.get("WANDB_TOKEN") #make sure the name matches what it was saved with in the .env file
```
   This is already handled in `main.py` in the current pipeline , but make sure your .env file exists and contains the correct token.



- **Loguru** is also integrated for general-purpose application logging.

  

## Preprocessing BRAT Data
If your data is in BRAT format (i.e., `.txt` and `.ann` files), you can use the preprocessing pipeline script to convert them to machine learning–ready CoNLL format.

Script: `pipelines/data-pipelines/data-preprocessing/convert_brat_to_conll.py`

This script does the following:

- Loads `.ann` and `.txt` files from a given input directory

- Tokenizes the text and aligns tokens with entity spans

- Splits by file using predefined ratio

- Outputs `.txt` CoNLL-formatted files (train, val, test)


Example Usage:
``` bash
python pipelines/data-pipelines/data-preprocessing/convert_brat_to_conll.py data=cell-finder
```
Where:

- `convert_brat_to_conll.py`: Is the script that runs the conversion

- `data=cell-finder`: Is the Hydra configuration that tells the script which dataset config to use (from configs/data/cell-finder.yaml)




## Getting Started
To set up and run the pipeline, follow these steps:

### Prerequisites

- Python 3.8+: The pipeline is developed and tested with Python 3.8 and newer versions.

- Virtual Environment Tool (Recommended for managing dependencies):

   - [pyenv](https://github.com/pyenv/pyenv)

   - [uv](https://docs.astral.sh/uv/) (fast dependency manager)

   - Alternatively: `venv`, `virtualenv`, or `conda`

### Steps
1. **Clone the Repository:**
   
First, clone the repository to your local machine:

``` bash
git clone https://github.com/ML4LitS/OTAR3088.git 
cd Entity-Extraction-Modular-pipeline

```
2. **Setup virtual env:**
 - Using [uv](https://docs.astral.sh/uv/) (Recommended for fast setup):
For Linux/Mac
```bash
uv venv env_name #e.g uv venv ner-model
source env_name/bin/activate  # or `source .venv/bin/activate` if using uv default venv without name.
```
For Windows
```bash
source .venv/bin/activate.fish
```
Please see [uv](https://docs.astral.sh/uv/) for full instructions on setting up a virtual environment. 

- Using python's native virtual environment.
```bash
python -m venv .venv
source .venv/bin/activate
```

Once the virtual environment is setup, install the `requirements.txt` file. 
```bash
pip install -r requirements.txt
```

3. Setup your Weight and Bias API key as shown in [Experiment Tracking with Weight and Bias](#experiment-tracking-with-weight-and-bias) section



##  Future Enhancements
- Future iterations of this pipeline are planned to include:

- Direct data loading from the Hugging Face Hub.

- Support for optional test sets for Hugging Face data.

- Integration of additional evaluation metrics and visualisations.

- Support for more model architectures and pre-trained models.



## Contributing

We welcome issues, feature requests, and pull requests. To contribute:

- Fork the repo

- Create a feature branch

- Submit a pull request with detailed description


## Licencing 
