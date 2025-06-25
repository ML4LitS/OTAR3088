# End-to-End Entity Extraction Modular Pipeline

This repository hosts an end-to-end modular and scalable entity extraction pipeline. The primary goal is to provide a flexible and modular framework that supports various data formats, and modeling frameworks.  It is built with a focus on plug-in modularity, version control, reproducibility, and easy evaluation.
This pipeline has been developed for the OTAR3088 project initiative focused on knowledge extraction from scientific and biomedical literature. 

## Project Information


## Current Functionalities

The current iteration of the pipeline offers the following key features:

1. **Model Training Support:**

- Flair Models: Integration for training and fine-tuning sequence labelling models using the [Flair](https://flairnlp.github.io/) framework.

- Hugging Face Transformers: Support for training and fine-tuning transformer-based models via the Hugging Face [transformers](https://huggingface.co/docs/transformers/en/index) library.

2. **Data input for flair**

Flair models expect all input data (training, development/validation, and test sets) to be saved as individual .txt files within a designated folder. Each file should contain data in a format compatible with Flair's ColumnCorpus (typically CoNLL-like). Example folder structure is shown below: 
```
flair_dataset/
    ├── train.txt
    ├── val.txt
    └── test.txt
```
Example data structure: 
```
Word1    LABEL1
Word2    LABEL2
...
WordN    LABELN

WordA    LABELA
WordB    LABELB

```
3. **Data Input for Hugging Face:**

The Hugging Face integration currently supports loading data in txt/conll or csv/tsv formats.

- csv/tsv files are assumed to be in one of two structures:

   - CoNLL-style: Two columns (token and label) separated by a delimiter, with blank lines indicating sentence boundaries(same as used for Flair model)

   - Standard DataFrame format: Each row represents a sample, with one column containing a list of tokens and another containing a corresponding list of labels for the entire sentence/example.

It is currently assumed that train, test, and validation (or dev) splits are present in the input data directory(same as expected by flair). Future improvements will include support for optional test sets and direct loading of datasets from the Hugging Face Hub.

4.**Configuration Management with Hydra:**

The pipeline extensively uses Hydra for flexible and organised configuration management.

The primary configuration directory (config/) is structured into the following sub-directories:

- `data/`: Contains configuration parameters specific to each dataset (e.g., paths, format details, splits).

- `logging/`: Manages settings for various logging backends.

- `models/`: Defines configuration parameters for each supported model type (Flair, Hugging Face), including hyperparameters and architecture settings.

This structure allows for easy modification of experimental variables and promotes reproducibility. Defaults are set via the central config file `common_config.yaml`

5. **Experiment Tracking Integration:**

- The pipeline supports various logging and experiment tracking tools, with a current emphasis on Weights & Biases (WandB).

- Loguru is also integrated for general-purpose application logging.

WandB API Token: It is a prerequisite that your Weights & Biases API token is available as an environment variable (e.g., WANDB_API_KEY) for experiment tracking to function correctly. The pipeline expects this token to be set and loaded in the main script `main.py`.


