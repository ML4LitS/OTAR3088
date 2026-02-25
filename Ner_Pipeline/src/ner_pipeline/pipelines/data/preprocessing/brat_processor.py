#standard library

import glob
from pathlib import Path
from typing import List, Union, Optional, Dict, Literal
from dataclasses import dataclass, field

from tqdm import tqdm
from datasets import Dataset


#custom modules
from .iob_converter import SpacyIOBConverter, HFIOBConverter
from .entity_processor import sentencize_and_align_entity_spans, rename_ent, filter_ent

from ner_pipeline.utils.io.writers import write_to_conll
from ner_pipeline.utils.io.readers import load_brat
from ner_pipeline.utils.hf_hub_utils import HFParams

from ner_pipeline.schemas.ner_dataset_schema import BratConfig, IOBConfig, nlp, 




class BratProcessor:
  """
  Processes and convert BRAT files to BIO/IOB.
  Args:
    input_dir: Path to file/directory containing BRAT file(s)
    output_dir: Path to save processed files if save_dest is local
    brat_config: Required params defining BRAT processing schema.Check defined class(BRATConfig)
    iob_config: Required params defining IOB conversion type. Check specified class(IOBConfig)
    hf_params: Required params to push a processed brat dataset to Hf HUB. Only required if brat_config.save_dest="hf"
  Returns:
    None
  """
  def __init__(self,
               input_dir:Union[Path, str],
               output_dir:Union[Path, str]=None,
               brat_config:Optional[BratConfig]= None,
               iob_config:Optional[IOBConfig]=None,
               hf_params:Optional[PushToHubParams]= None
               ):
    self.input_dir = Path(input_dir)
    self.output_dir = Path(output_dir) if output_dir else None
    self.brat_config = brat_config or BratConfig()
    self.iob_config = iob_config or IOBConfig()
    self.hf_params = hf_params
    self.file_ids = self.get_file_ids()
    self.__validate_brat_config()

  def __repr__(self):
    return f"BratProcessor(data={self.file_ids}, brat_config={self.brat_config}, iob_config={self.iob_config})"

  def __validate_brat_config(self):
    if self.brat_config.save_dest not in ["hf", "local"]:
      raise ValueError("Unsupported save format")

    if self.brat_config.save_dest == "hf":
        if not self.hf_params:
          raise ValueError("HF params is required if save_dest=`hf`")
        if self.hf_params.is_private and not self.hf_params.token:
          raise ValueError("HF Auth Token is required if is_private=True")

    if self.brat_config.save_dest == "local" and not self.output_dir:
      raise ValueError("Output directory is required if save_dest=`local`")


  def get_file_ids(self):
    return [file.stem for file in self.input_dir.glob("*.txt")]


  def process_single(self, file_id): #
    do_filter = self.brat_config.do_filter
    do_rename = self.brat_config.do_rename
    ent_label_key = self.brat_config.ent_label_key
    rename_map = self.brat_config.rename_map

    txt_path = self.input_dir / f"{file_id}.txt"
    tokenized_text = parse_brat_to_sentences(txt_path, ent_label_key)
    if do_rename:
      entities = rename_ent(entities, dataset_format="spacy",
                            rename_map=rename_map, ent_label_key=ent_label_key)

    entities = filter_ent(entities) if do_filter else entities

    if brat_config.iob_converter_type == "hf":
      converter = HFIOBConverter(data=tokenized_text, config=iob_config)
    #else use default spacy
    else:
      converter = SpacyIOBConverter(data=tokenized_text, config=iob_config)
    iob_data = converter.convert()

    return iob_data


  def batch_process(self):
    all_datasets = []
    for file_id in tqdm(self.file_ids, desc="Processing brat to conll", total=len(self.file_ids)):
      iob_data = self.single_brat_processor(file_id)
      if not iob_data:
        continue
      if self.brat_config.save_dest == "hf":
        all_datasets.extend(iob_data)
        self._push_ds_to_hub(all_datasets)
      else:
        write_to_conll(iob_sentences, text_col="tokens", label_col="tags", 
                        output_dir=self.output_dir, file_name=file_id)

  def _push_ds_to_hub(self, dataset_lst:List[Dict]):
    dataset = Dataset.from_list(dataset_lst)

    dataset_kwargs = {
      "repo_id": self.hf_params.repo_id,
      "commit_message": self.hf_params.commit_message,
    }
    if self.hf_params.is_private:
      dataset_kwargs["token"] = self.hf_params.token
    dataset.push_to_hub(**dataset_kwargs)


def parse_brat_to_sentences(file_path:str, ent_label_key:str="label"):
  """
    Parses a BRAT document by inferring paired .txt/.ann files from a single path.
    
    This function normalizes entity offsets from document-relative to 
    sentence-relative positioning.
    
    Args:
        file_path: Path to either the .txt or .ann file of the BRAT pair.
        
    Returns:
        A list of sentencized dictionaries with aligned offsets, or None if 
        the document cannot be loaded.
  """
  path = Path(file_path)
  txt_path = path.with_suffix(".txt")
  print(txt_path)
  ann_path = path.with_suffix(".ann")

  if not txt_path.exists() or not ann_path.exists():
    raise FileNotFoundError(f"Missing .txt or .ann not found in directory for file:{path.stem}")
  
  dataset = load_brat(txt_path)
  print(dataset)
  if not dataset:
    return None
  text, entities = dataset[0]["doc"], dataset[0]["entities"]
  return sentencize_and_align_entity_spans(document=text, doc_annotations=entities,ent_label_key=ent_label_key)
  

