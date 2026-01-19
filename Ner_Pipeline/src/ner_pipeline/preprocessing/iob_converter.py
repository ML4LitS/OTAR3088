from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union, Any
import pandas as pd

import spacy
import scispacy
from tqdm import tqdm

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast
from schemas.ner_params import IOBConfig, nlp




EntityDict = List[Dict[str, Any]]




class IOBConverter(ABC):
  def __init__(
    self, 
    data:Union[pd.Dataframe, EntityDict], 
    config: IOBConfig):
    self.data = data
    self.config = config

  
  def __repr__(self):
    return f"{self.__class__name__}(num_samples={(len(self.data))}, config_params={self.params})"


  def _validate_entity_schema(self, entities: SpacyJsonOBJ) -> None:
    if not isinstance(entities, list) or not all(isinstance(item, dict) for item in entities):
      raise ValueError(f"Expected list of dicts, got {type(entities)}")

    required_keys = {"start", "end", self.config.ent_label_key}
    for ent in entities:
      missing_key = required_keys - ent.keys()
      if missing_key:
        raise KeyError(f"Missing required entity dict key(s): {missing_key}")
    
  @abstractmethod
  def _tokenize_with_offsets(self, sentence: str):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_tokens_offsets()"
        )


  def _process_single_sentence(self, sentences:str, entities:EntityDict):
    self._validate_entity_schema(entities)
    tokens, offsets = self._tokenize_with_offsets(sentences)
    iob_tags = ["O"] * len(tokens)

    for ent in entities:
      ent_start, ent_end, label = ent["start"], ent["end"], ent[self.config.ent_label_key]
      is_first_token = True
      for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start >= ent_start and tok_end <= ent_end:
          iob_tag[idx] = f"B-{label}"
          is_first_token = False
        else:
          iob_tag[idx] = f"I-{label}"
    return tokens, iob_tags

  def batch_process(self, example: Dataset):
    all_tokens = []
    all_iob_tags = []

    for sentences, entities in tqdm(
      zip(example[self.config.text_col], example[self.config.label_col]),
      total=len(example),
      desc="Converting to IOB format"
    ):
      tokens, iob_tags = self._process_single_sentence(sentences, entities)
      all_tokens.append(tokens)
      all_iob_tags.append(iob_tags)

    return {"tokens": all_tokens, "tags": all_iob_tags}


  def convert(self) -> Union[Dataset, DatasetDict, List[Dict[List[str], List[str]]]]:
    if isinstance(self.data, pd.DataFrame):
      ds = Dataset.from_pandas(self.data)
    elif instance(self.data, list) and all(isinstance(item, dict) for item in self.data):
      ds = Dataset.from_list(self.data)

    processed_ds = ds.map(
      self.batch_process,
      batched=True,
      remove_columns=ds.column_names,
    )
    return processed_ds if self.config.as_hf_dataset else processed_ds.to_list()





class SpacyIOBConverter(IOBConverter):
  def __init__(
    self,
    data:Union[pd.Dataframe, EntityDict], 
    config: IOBConfig
    )-> Union[Dataset, DatasetDict, List[Dict[List[str], List[str]]]]:
    super().__init__(data, config)


  def _tokenize_with_offsets(self, sentences: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    doc = self.config.tokenizer_backend(sentences) 
    tokens = [tok.text for tok in doc]
    offsets = [(tok.idx, tok.idx + len(tok.text)) for tok in doc]
    return tokens, offsets
    
    
class HFIOBConverter(IOBConverter):
  def __init__(
    self,
    data:Union[pd.Dataframe, EntityDict], 
    config: IOBConfig
    )-> Union[Dataset, DatasetDict, List[Dict[List[str], List[str]]]]:
    super().__init__(data, config)

  def _tokenize_with_offsets(self, sentences: str):
    encoded = self.config.tokenizer_backend(
      sentences,
      return_offsets_mapping=True,
      return_special_tokens_mask=False,
      return_tensors=None,
    )

    tokens = encoded.tokens()
    offsets = encoded.offset_mapping
    cleaned_tokens, cleaned_offsets = [], []
    for token, (token_start, token_end) in zip(tokens, offsets):
      if token_start == token_end == 0:
        continue
      cleaned_tokens.append(token)
      cleaned_offsets.append((token_start, token_end))

    return cleaned_tokens, cleaned_offsets




