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

3. **Data Input for Hugging Face:**

The Hugging Face integration currently supports loading data in txt/conll or csv/tsv formats.

- csv/tsv files are assumed to be in one of two structures:

- CoNLL-style: Two columns (token and label) separated by a delimiter, with blank lines indicating sentence boundaries(same as used for Flair model)

- Standard DataFrame format: Each row represents a sample, with one column containing a list of tokens and another containing a corresponding list of labels for the entire sentence/example.

It is currently assumed that train, test, and validation (or dev) splits are present in the input data directory. Future improvements will include support for optional test sets and direct loading of datasets from the Hugging Face Hub.
