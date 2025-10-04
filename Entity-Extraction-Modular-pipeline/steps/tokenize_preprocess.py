
from typing import List, Tuple, Dict, Union, Optional
from pathlib import Path

import spacy
import scispacy
from tqdm import tqdm

from datasets import Dataset, DatasetDict, Sequence, Value, ClassLabel

from utils.file_writers import write_to_conll
from utils.file_parsers import load_brat
from utils.helper_functions import clean_text


nlp = spacy.load("en_core_sci_md", disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"]) 
nlp.add_pipe("sentencizer")
nlp.max_length = 10_000_000

#To-DO
"""
1. Abstract `rename_ent_cf` and `filter_ent` instead of hardcoding the replace dict. 
2. merge logic of `rename_ent_cf` with already existing `rename_ent` in `utils.helper_functions`
"""



def flatten_singleton_labels(entities_lst, label_key="labels"):


    if not isinstance(entities_lst, list):
        raise ValueError(f"Expected List of Dictionaries but got : {type(entities_lst)}")
    cleaned_entities_lst = []
    for ent in entities_lst:
        if isinstance(ent, dict) and label_key in ent and isinstance(ent[label_key] , list) and len(ent[label_key]) == 1:
            cleaned_ent = ent.copy()
            cleaned_ent[label_key] = cleaned_ent[label_key][0]
            cleaned_entities_lst.append(cleaned_ent)
        else:
            cleaned_entities_lst.append(ent)

    return cleaned_entities_lst




def rename_ent_cf(ent_list: List[Dict]) -> List[Dict]:
    """
    Rename 'Anatomy' labels to 'Tissues' in the entity list.
    """
    return [
        {**ent, "label": "Tissue"} if ent["label"].strip() == "Anatomy" else ent
        for ent in ent_list]


def filter_ent(ent_list: List[Dict]) -> List[Dict]:
    """
    Keep only entities with relevant labels: CellLine, CellType, Tissue.
    Handles casing and whitespace.
    """
    relevant_ent = {"cellline", "celltype", "tissue"}
    return [
        ent for ent in ent_list
        if ent.get("label", "").strip().lower() in relevant_ent
    ]


def cast_to_class_labels(dataset:Dataset, label_col:str, text_col:str, unique_tags:List):
    features = dataset.features.copy()
    features[text_col] = Sequence(Value("string"))
    features[label_col] = Sequence(ClassLabel(names=unique_tags,
                                              num_classes=len(unique_tags)
                                              ))
    return dataset.cast(features)


def tokenize_with_offsets(text: str, entities, label_key="label", nlp:scispacy=nlp):
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        sent_start = sent.start_char
        sent_end = sent.end_char
        sent_entities = []

        for ent in entities:
            if ent["start"] >= sent_start and ent["end"] <= sent_end:
              sent_entities.append({
                 "start": ent["start"] - sent_start,
                  "end": ent["end"] - sent_start,
                  "label": ent[label_key]
            })
        sentences.append({
            "sentence": sent.text.strip(),
            "entities": sent_entities
        })
    return sentences



def convert2iob(
    data: Union[str, Dict, List[Dict]],
    entities: List[Dict] = None,
    nlp:scispacy=nlp
) -> List[Dict]:
    """
    Convert sentence(s) and entities to IOB format.
    
    Accepts:
    - Single string + entities list
    - Single dict: {"sentence": str, "entities": [...]}
    - List of such dicts
    
    Returns:
    List of dicts: [{'tokens': [...], 'tags': [...]}]
    """
    
    
    # Case 1: list of dicts
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return [_process_sentence(item['sentence'], item['entities']) for item in data]
    
    # Case 2: single dict
    elif isinstance(data, dict) and 'sentence' in data and 'entities' in data:
        return [_process_sentence(data['sentence'], data['entities'])]
    
    # Case 3: single string + entities
    elif isinstance(data, str) and entities is not None:
        return [_process_sentence(data, entities)]
    
    else:
        raise ValueError("Unsupported input format for convert2iob()")



def _process_sentence(sent: str, entities: List[Dict]) -> Dict:

    # Tokenize sentence
    doc = nlp(sent)
    tokens = [token.text for token in doc]
    offsets = [(token.idx, token.idx + len(token.text)) for token in doc]

    # Initialize all tags as 'O'
    tags = ['O'] * len(tokens)

    # Assign IOB tags
    for ent in entities:
        start, end, entity = ent['start'], ent['end'], ent['label']
        for i, (token_start, token_end) in enumerate(offsets):
            if token_start >= start and token_end <= end:
                if token_start == start:
                    tags[i] = f'B-{entity}'
                else:
                    tags[i] = f'I-{entity}'

    return {'tokens': tokens, 'tags': tags}



def _shift_label(label):
  if label % 2 == 1:
    label += 1
  return label



def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
      if word_id is None:
        new_labels.append(-100)
      elif word_id != current_word:
        current_word = word_id # Start of a new word!
        new_labels.append(labels[word_id])
      else:
        new_labels.append(_shift_label(labels[word_id]))
    return new_labels


def tokenize_and_align(examples,
                       tokenizer,
                       *,
                       #device=None,
                       tag_col='labels',
                       text_col='words'):
    tokenized_inputs = tokenizer(
        examples[text_col],
        max_length=512,
        truncation=True,
        padding=True,
        is_split_into_words=True
    )
    new_labels = []

    for i, labels in enumerate(examples[tag_col]):
      word_ids = tokenized_inputs.word_ids(i)
      new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs



def process_single_file_brat(file_id: str, input_dir: Path,
                             do_filter=True,
                             do_rename=True):
    """
    Process one BRAT-annotated document and return labeled tokenized sentences in IOB/BIO format.
    """
    input_dir = Path(input_dir)
    txt_path = input_dir / f"{file_id}.txt"
    dataset = load_brat(txt_path)
    if not dataset:
        return None
    text, entities = dataset[0]["text"], dataset[0]["entities"]

    # text = clean_text(text)
    entities = rename_ent_cf(entities) if do_rename else entities
    entities = filter_ent(entities) if do_filter else entities
  
    tokenized_text = tokenize_with_offsets(text, entities)

    iob_data = convert2iob(tokenized_text, entities, nlp)

    return iob_data

  
def process_mult_file_brat(file_ids: List[str], 
                        text_col: str, label_col: str,
                        input_dir: Path, output_dir: Path,
                        filename:Optional[str]):
    """
    Process multiple files from brat to conll
    Args:
        file_ids: List of file_ids/name without extension e.g [file1,file2,......, filez]
        text_col: name of key in dict or column containing list of sentences/tokens. 
        label_ocl: name of key in dict or column containing list of labels
        input_dir: Path to directory containing brat files
        output_dir: Path to directory to save generated conll file

    return:
        None
    """
    for file_id in tqdm(file_ids, desc="Processing brat to conll"):
        iob_sentences = process_single_file_brat(file_id, input_dir)
        #print(iob_sentences)
        if iob_sentences:
            write_to_conll(iob_sentences, text_col, label_col, output_dir, file_name=filename)







