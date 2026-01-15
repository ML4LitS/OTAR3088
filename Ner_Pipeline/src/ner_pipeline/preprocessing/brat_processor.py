#standard library

import glob
from pathlib import Path
from typing import List, Union, Optional, Dict, Literal
from dataclasses import dataclass, field

from tqdm import tqdm
from datasets import Dataset




#custom modules
from utils.file_writers import write_to_conll
from utils.file_readers import load_brat
from iob_converter import SpacyIOBConverter, IOBConfig
from entity_processor import sentencize_and_align_entity_spans, rename_ent, filter_ent, nlp



@dataclass
class BratConfig:
  "Default configuration for processing BRAT files to IOB"
  do_filter: bool = True
  do_rename: bool = True
  ent_label_key: str = "label"
  text_col:str = "text"
  label_col:str = "entities"
  save_dest:Literal["hf", "local"] = "hf"
  rename_map: Union[Dict[str, str], None] = field(default_factory=lambda: {"Anatomy": "Tissue"})


@dataclass
class HFParams:
  "Default HF upload params"
  hf_dataset_name:str
  is_private:bool=False
  token:str=None
  commit_message:str="Add dataset to hub"


class BratProcessor:
  def __init__(self,
               input_dir:Union[Path, str],
               output_dir:Union[Path, str],
               config:Optional[BratConfig]= None,
               hf_params:Optional[HFParams]= None
               ):
    self.input_dir = Path(input_dir)
    self.output_dir = Path(output_dir)
    self.config = config or BratConfig()
    self.hf_params = hf_params
    self.file_ids = self.get_file_ids()
    self.__validate_configs()

  def __validate_configs(self):
    if self.config.save_dest not in ["hf", "local"]:
      raise ValueError("Unsupported save format")

    if self.config.save_dest == "hf":
        if not self.hf_params:
          raise ValueError("HF params is required if save_dest=`hf`")
        if self.hf_params.is_private and not self.hf_params.token:
          raise ValueError("HF Auth Token is required if is_private=True")

    if self.config.save_dest == "local" and not self.output_dir:
      raise ValueError("Output directory is required if save_dest=`local`")


  def get_file_ids(self):
    return [file.stem for file in self.input_dir.glob("*.txt")]


  def process_single(self, file_id): #file_ids, ent_label_key, input_dir, do_filter, do_rename
    do_filter = self.config.do_filter
    do_rename = self.config.do_rename
    ent_label_key = self.config.ent_label_key
    rename_map = self.config.rename_map

    input_dir = Path(self.input_dir)
    txt_path = input_dir / f"{file_id}.txt"
    dataset = load_brat(txt_path)
    if not dataset:
        return None
    text, entities = dataset[0]["text"], dataset[0]["entities"]
    if do_rename:
      entities = rename_ent(entities, dataset_format="spacy",
                            rename_map=rename_map, ent_label_key=ent_label_key)

    entities = filter_ent(entities) if do_filter else entities

    tokenized_text = sentencize_and_align_entity_spans(text, entities)

    config = IOBConfig(tokenizer_backend=nlp,
                        text_col="sentence",
                        label_col="entities",
                        ent_label_key=ent_label_key,
                        as_hf_dataset=False
    )
    converter = SpacyIOBConverter(data=tokenized_text, config=config)
    iob_data = converter.convert()

    return iob_data


  def batch_process(self):
    #create a list for saving appending all files if pushing to hub
    if self.config.save_dest == "hf":
        all_dataset_list = []
        for file_id in tqdm(self.file_ids, desc="Processing brat to conll"):
          iob_sentences = self.single_brat_processor(file_id)
          if iob_sentences:
            all_dataset_list.extend(iob_sentences)
        dataset = Dataset.from_list(all_dataset_list)

        dataset_kwargs = {
          "repo_id": self.hf_params.hf_dataset_name,
          "commit_message": self.hf_params.commit_message,
        }
        if self.hf_params.is_private:
          dataset_kwargs["token"] = self.hf_params.token
        dataset.push_to_hub(**dataset_kwargs)


    else:
    #save conll files individually to a local folder with file_id as name of each file
      for file_id in tqdm(self.file_ids, desc="Processing brat to conll"):
        iob_sentences = self.single_brat_processor(file_id)
        if iob_sentences:
          write_to_conll(iob_sentences, self.config.text_col, self.config.label_col, self.output_dir, file_name=file_id)
