import os
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
import logging

#load_brat------> if loading a single file, pass both file path and annotation file path. 
# If loading multiple files, then pass only directory path. 
# The function will take care of reading the files and their respective annotations using the file id name 


##script TO-DO : Include function doc strings 


def load_brat(input_path: Union[str, Path]):
    input_path = Path(input_path)
    dataset = []

    if input_path.is_file() and input_path.suffix == ".txt":
        ann_path = input_path.with_suffix(".ann")
        if ann_path.exists():
            file_id = input_path.stem
            full_text, entities = _read_brat(input_path, ann_path)
            dataset.append({"id":file_id, "text": full_text, "entities": entities})
        else:
            raise FileNotFoundError(f"Annotation file not found for {input_path}")
    
    elif input_path.is_dir():
        for file in os.listdir(input_path):
            if file.endswith(".txt"):
                file_id = file[:-4]
                text_file = input_path / f"{file_id}.txt"
                ann_file = input_path / f"{file_id}.ann"
                if text_file.exists() and ann_file.exists():
                    full_text, entities = _read_brat(text_file, ann_file)
                    dataset.append({"text": full_text, "entities": entities})
                else:
                    logging.warning(f"Missing .txt or .ann for file ID: {file_id}")
    else:
        raise FileNotFoundError(f"Invalid input path: {input_path}")
    
    return dataset


def _read_brat(txt_path: Path, ann_path: Optional[str] = None):
    try:
        file = Path(txt_path)
        full_text = file.read_text()

        entities = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("T"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                entity_info, entity_name = parts[0], parts[1], parts[2]

                # Some annotations have spans like: "Entity 0 10;12 15"
                try:
                    span = entity_info.split(" ")
                    label = span[0]
                    start = span[1]
                    end = span[-1]
                except Exception as e:
                    logging.warning(f"Skipping malformed span: {entity_info}")
                    continue

                entities.append({
                    "word": label,
                    "label": entity_name,
                    "start": int(start),
                    "end": int(end)
                    })

        return full_text, entities

    except Exception as e:
        logging.error(f"Error reading {txt_path.name}: {e}")
        return None, []
    




##load conll file types---> files are tab separated in the format : token   token_label

def load_conll(file_path:Path) -> List[List[str], List[str]]:

        all_tokens, all_labels = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            sentence_tokens, sentence_labels = [], []
            for line in f:
                line = line.strip() #removes whitespace
                if not line: #empty/blank lines marks the end of a sentence
                    if sentence_labels:
                        all_tokens.append(sentence_tokens)
                        all_labels.append(sentence_labels)
                        sentence_tokens, sentence_labels = [], []
                else:
                    parts = re.split('\s+', line)
                    if len(parts)>=2:
                        all_tokens.append[parts[0]] #first part represents tokens
                        all_labels.append[parts[-1]] #last part represents tags 

            if sentence_labels: #appending the last sentence in the file after loppping through
                all_tokens.append(sentence_tokens) #
                all_labels.append(sentence_labels)
            else:
                raise ValueError(f"Line format error: '{line}'")
            


        return all_tokens, all_labels