
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



def flatten_singleton_labels(entities_lst, ent_label_key="labels"):


    if not isinstance(entities_lst, list):
        raise ValueError(f"Expected List of Dictionaries but got : {type(entities_lst)}")
    cleaned_entities_lst = []
    for ent in entities_lst:
        if isinstance(ent, dict) and ent_label_key in ent and isinstance(ent[ent_label_key] , list) and len(ent[ent_label_key]) == 1:
            cleaned_ent = ent.copy()
            cleaned_ent[ent_label_key] = cleaned_ent[ent_label_key][0]
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


def tokenize_with_offsets(text: str, entities, ent_label_key="label", nlp:scispacy=nlp):
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
                  "label": ent[ent_label_key]
            })
        sentences.append({
            "sentence": sent.text.strip(),
            "entities": sent_entities
        })
    return sentences



def convert2iob(
    data: Union[str, Dict, List[Dict], pd.Series, pd.DataFrame],
    entities: List[Dict] = None,
    nlp:scispacy=nlp,
    ent_label_key: str = "label",
    return_hf: bool = False
) -> Union[Dict, List[Dict], Dataset]:
    """
    Convert sentences + entities into IOB format.

    Accepts:
    - text as single str and entities as list of dict e.g "Stem cells are amazing", [{"start": 0, "end": 10, "label": "CellType"}]
    - dict with keys sentence(value:str), entities(value:list) e.g {"sentence": str, "entities": [...]}
    - list of dicts(standard spacy annotation format): e.g `[{'sentence': 'Human embryonic stem cells.....','entities': [{'start':, 'end': , 'labels': '},{'start': , 'end': , 'labels': ''},....] 
    - pandas row (Series) with 'sentence' and 'entities'. Can also be used as an apply function to a dataframe e.g `df.apply(lambda row: convert2iob(row), axis=1)`
    - pandas DataFrame with 'sentence' and 'entities'. Pass dataframe directly as `convert2iob(df)

    Args:
        data: input data
        entities: required only if `data` is a string
        nlp: spaCy model (must be provided)
        ent_label_key: which key holds entity labels ("label" or "labels")
        return_hf: if True, returns HuggingFace Dataset object

    Returns:
        - HuggingFace Dataset if return_hf=True otherwise list of dicts: {"tokens":[], "tags":[]}
    """

    if nlp is None:
        raise ValueError("Please provide a loaded spaCy model to `nlp`")


    results = None

    # Case 1: list of dicts
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
      results = [_process_sentence(item["sentence"], item["entities"], ent_label_key, nlp) for item in data]

    # Case 2: single dict
    elif isinstance(data, dict) and "sentence" in data and "entities" in data:
      results = [_process_sentence(data["sentence"], data["entities"], ent_label_key, nlp)]

    # Case 3: pandas series
    elif isinstance(data, pd.Series) and "sentence" in data and "entities" in data:
      results = [_process_sentence(data["sentence"], data["entities"], ent_label_key, nlp)]

    # Case 4: pandas DataFrame 
    elif isinstance(data, pd.DataFrame):
      if not {"sentence", "entities"}.issubset(data.columns):
        raise ValueError("Expected DataFrame to contain 'sentence' and 'entities' columns")
      results = [_process_sentence(row["sentence"], row["entities"], ent_label_key, nlp) for _, row in data.iterrows()]

    # Case 5: raw string + entities
    elif isinstance(data, str) and entities is not None:
      results = [_process_sentence(data, entities, ent_label_key, nlp)]

    else:
      raise ValueError("Unsupported input format for convert2iob()")

    # Convert to HF dataset if set to True
    if return_hf:
      return Dataset.from_list(results)

    # Return single dict if only one item
    return results if len(results) > 1 else results[0]



def _process_sentence(sent: str, ents: List[Dict], ent_label_key:str,nlp:scispacy) -> Dict:
    # Tokenize sentence
    doc = nlp(sent)
    tokens = [tok.text for tok in doc]
    offsets = [(tok.idx, tok.idx + len(tok.text)) for tok in doc]

    # Initialise all tags as 'O'
    tags = ["O"] * len(tokens)

    # Assign IOB tags
    for ent in ents:
        start, end = ent["start"], ent["end"]
        entity = ent.get(ent_label_key)
        if entity is None:
            raise KeyError(f"Entity dict missing expected entity key '{ent_label_key}'")

        for i, (token_start, token_end) in enumerate(offsets):
            if token_start >= start and token_end <= end:
                tags[i] = f"B-{entity}" if token_start == start else f"I-{entity}"

    return {"tokens": tokens, "tags": tags}



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
                             ent_label_key:str,
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

    entities = rename_ent_cf(entities) if do_rename else entities
    entities = filter_ent(entities) if do_filter else entities
  
    tokenized_text = tokenize_with_offsets(text, entities)

    iob_data = convert2iob(tokenized_text, ent_label_key)

    return iob_data


  
def process_mult_file_brat(file_ids: List[str], 
                        text_col: str, 
                        label_col: str,
                        ent_label_key:str,
                        input_dir: Path, output_dir: Path,
                        filename:Optional[str]):
    """
    Process multiple files from brat to conll
    Args:
        file_ids: List of file_ids/name without extension e.g [file1,file2,......, filez]
        text_col: name of key in dict or column containing list of sentences/tokens. e.g "words", "tokens"
        label_col: name of key in dict or column containing list of entity labels e.g "tags", "ner_tags", "labels
        ent_label_key: key in entity dict that holds the particular label info. e.g "label" or "labels"
        input_dir: Path to directory containing brat files
        output_dir: Path to directory to save generated conll file

    return:
        None
    """
    for file_id in tqdm(file_ids, desc="Processing brat to conll"):
        iob_sentences = process_single_file_brat(file_id, input_dir, ent_label_key)
        if iob_sentences:
            write_to_conll(iob_sentences, text_col, label_col, output_dir, file_name=filename)







