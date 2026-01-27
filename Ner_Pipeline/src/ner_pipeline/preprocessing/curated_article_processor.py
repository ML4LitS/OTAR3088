from typing import List, Callable, Dict, List, Any
from dataclasses import dataclass
from collections import Counter
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from loguru import logger

from schemas.ner_params import NerDataFields
from entity_processor import sentencize_and_align_entity_spans


def detect_section_headers(text:str, entities: List[Dict[str, Any]]):
  """
  Detects if a given text segment is likely a section header based on its content
  or if it's a short text containing 'PMC' without any associated entities.

  Args:
      text (str): The text segment to check.
      entities (List[Dict[str, Any]]): A list of entities associated with the text segment.

  Returns:
      bool: True if the text segment is likely a section header, False otherwise.
  """
  text = text.strip()
  common_headers = {"abstract", "background", "results", 
                    "discussion", "source paper", "declaration of interests"
                    "data collection and avaliability", "acknowledgement"
                    }
  if "PMC" in text and len(text.split()) < 10:
    return True

  if any(common_header in text.lower() for common_header in common_headers) and not entities:
    return True
  

  return False



class ArticleNormaliser:
  """
    Normalises article data by concatenating text parts, removing section headers,
    and re-aligning entity annotations to sentence-level offsets.

  """
  def __init__(
    self, 
    params: NerDataFields,
    section_header_func: Callable
  ):
    """
    Args:
        params (NerDataFields): A config object containing column names for text,
            labels, and the key for entity labels within the label dictionaries.

        detect_section_headers_func (Callable): A function used to identify
            and filter out section headers from the article text.
    """
    self.params = params
    self.section_header_func = section_header_func
    self.paragraph_separator = "\n\n"

  def __repr__(self):
    return f"ArticleNormaliser(article_params={self.params})"

  def normalise(self, df:pd.DataFrame):
    fulltext_parts = []
    global_annotations = []
    cursor = 0

    for _, row in tqdm(df.iterrows(), total=len(df),desc="Normalising Articles", colour="blue"):
      text = row[self.params.text_col].strip()
      entities = row[self.params.label_col]

      if not text:
        continue
      if self.section_header_func(text, entities):
        continue
      fulltext_parts.append(text)
      for ent in entities:
        start_key = "start" if isinstance(ent["start"], int) else "startOffset"
        end_key = "end" if isinstance(ent["end"], int) else "endOffset"
        global_annotations.append(
          {
            "start": start + cursor,
            "end": end + cursor,
            "label": self.params.ent_label_key[0] 
                    if isinstance(ent[self.params.ent_label_key], list)
                    else ent[self.params.ent_label_key],
            "text": text
          }
        )
      cursor += len(text) + len(self.paragraph_separator)
      
    fulltext = self.paragraph_separator.join(fulltext_parts)
    sentence_data = sentencize_and_align_entity_spans(fulltext, 
                                                        global_annotations, 
                                                        label_field="label")

    return pd.DataFrame(sentence_data)





class NERDatasetAnalyser:
  """
  Analyses a sentence-level NER dataset. 
  Computes statistics and displays plots on sentence, entity distribution in dataset. 

  """
  def __init__(self, df, sent_col="sentence", ent_col="entities"):
    self.df = df
    self.sent_col = sent_col
    self.ent_col = ent_col
    self.df["sent_len"] = self.df[self.sent_col].apply(lambda x: len(x.split()))


  def _compute_stats(self):
    """
    Computes statistics about sentence and annotated entities in the dataset.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - total_num_sentence_with_entities (int): Count of sentences containing at least one entity.
            - entity_distribution_in_article (Dict[str, int]): A dictionary showing the count of each entity label.
    """
    label_counter = Counter()
    sentence_with_entities = 0

    for entities in self.df[self.ent_col]:
      if entities:
        sentence_with_entities += 1
        for entity in entities:
          label_counter.update([entity["label"]])
  
    total_sentences = df.shape[0]
    return {
        "total_num_sentence_with_entities": sentence_with_entities,
        "entity_distribution_in_article": dict(label_counter),
    }

 
  def get_sentence_stats_summary(self):
    """
    Calculates summary statistics for sentence lengths.

    Returns:
        Dict[str, float]: A dictionary containing:
            - avg_sent_len (float): Average sentence length.
            - min_sent_len (int): Minimum sentence length.
            - max_sent_len (int): Maximum sentence length.
    """
    return {
        "avg_sent_len": self.df["sent_len"].mean(),
        "min_sent_len": self.df["sent_len"].min(),
        "max_sent_len": self.df["sent_len"].max(),
    }

  def plot_data_stats(self):
    stats  = self._compute_stats()
    logger.info(f"Article Statitics: {stats}")
    logger.info(f"Sentence level Statistics: \n{self.get_sentence_stats_summary()}")

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    sns.histplot(self.df["sent_len"], ax=axes[0], bins=20)
    axes[0].set_xlabel("Sentence Length", fontsize=16)
    axes[0].set_title("Sentence Length Distribution", fontsize=16)

    ent_dist = stats["entity_distribution_in_article"]
    sns.barplot(x=list(ent_dist.keys()), y=list(ent_dist.values()), ax=axes[1], palette="rocket")

    axes[1].set_title("Entity Label Distribution", fontsize=16)
    axes[1].set_ylabel("Count", fontsize=16)
    axes[1].tick_params(axis="x", rotation=45, labelsize=16)

    plt.tight_layout()
    plt.show()