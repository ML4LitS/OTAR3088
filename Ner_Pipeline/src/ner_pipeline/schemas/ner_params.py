from dataclasses import dataclass, field
from typing import Literal, Union, Dict

import spacy

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast




nlp = spacy.load("en_core_sci_md", disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"]) 
nlp.add_pipe("sentencizer")
nlp.max_length = 10_000_000

@dataclass
class NerDataFields:
    text_col: str
    label_col: str
    ent_label_key: str




@dataclass
class BratConfig(NerDataFields):
  "Default configuration for processing BRAT files to IOB"
  do_filter: bool = True
  do_rename: bool = True
  save_dest: Literal["hf", "local"] = "hf"
  rename_map: Union[Dict[str, str], None] = field(default_factory=lambda: {"Anatomy": "Tissue"})


# @dataclass
# class CuratedArticleParams:
#     text_col: str
#     label_col: str
#     label_field: str



@dataclass
class IOBConfig(NerDataFields):
  tokenizer_backend: spacy.language.Language | PreTrainedTokenizerBase | PreTrainedTokenizerFast
  as_hf_dataset: bool = False

