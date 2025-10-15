# OTAR3088: Automated Knowledge Extraction for Biomedical Literature

This repository hosts the codebase and resources for the OTAR3088 project — a collaborative initiative between [Europe PMC (EPMC)](https://europepmc.org), [ChEMBL](https://www.ebi.ac.uk/chembl/), and [Open Targets](https://www.opentargets.org/).

The project aims to modernise and extend the existing Named Entity Recognition (NER) workflows used by EPMC and Open Targets to cover a broader range of biomedical entities relevant to drug discovery — including variants, biomarkers, tissues/cell types, adverse events, and assay conditions.

By incorporating these new entity types, the project seeks to provide higher confidence in the relevance of target–disease associations and enhance downstream knowledge extraction and integration

---

## Key Objectives
- Extend existing NER pipelines to support new biomedical entity types.
- Develop a modular, flexible framework that enables easy replacement or integration of new NLP models and datasets as they become available.
- Explore and benchmark modern NLP architectures (e.g., Transformer-based models) and advanced fine-tuning techniques for biomedical text mining.

---

## 🧩 Repository Structure
```markdown
| Folder | Description |
OTAR3088/
│
├── Entity-Extraction-Modular-pipeline/      # Main modular pipeline for biomedical NER
│   ├── steps/                               
│   ├── configs/                             # YAML configuration files (Hydra-based)
│   ├── pipelines/                           # Data preprocessing and model training pipelines
│   ├── utils/                               # Helper functions and utilities
│   └── README.md                            # Documentation for this module (multi-page)
│
├── Data_mining/                             # Scripts & notebooks for dataset exploration or sourcing
├── Data_extraction-Query/                   # Query-based data extraction workflows
├── Scripts/                                 # General-purpose or legacy scripts
└── README.md                                # Central project documentation (this file)

```

