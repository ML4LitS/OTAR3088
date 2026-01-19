from typing import List, Callable, Dict, List, Any
from dataclasses import dataclass
from collections import Counter
from tqdm import tqdm

import scispacy
import matplotlib.pyplot as plt 
import seaborn as sns

from schemas.ner_params import NerDataFields, nlp



def detect_section_headers(text:str, entities: List[Dict[str, Any]]):
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



def sentencize_and_align_entity_spans(document: str, doc_annotations, label_field="label", nlp:scispacy=nlp):
    "This sentencises and aligns sentence offsets with entities dict"
    doc = nlp(document)
    sentence_annotations = []
    for sent in doc.sents:
        sent_start = sent.start_char
        sent_end = sent.end_char
        sent_entities = []

        for entity in doc_annotations:
            if entity["start"] >= sent_start and entity["end"] <= sent_end:
              sent_entities.append({
                 "start": entity["start"] - sent_start,
                  "end": entity["end"] - sent_start,
                  "label": entity[label_field]
            })
        sentence_annotations.append({
            "sentence": sent.text.strip(),
            "entities": sent_entities
        })
    return sentence_annotations





class NERDatasetAnalyser:
  def __init__(self, df, sent_col="sentence", ent_col="entities"):
    self.df = df
    self.sent_col = sent_col
    self.ent_col = ent_col
    self.df["sent_len"] = self.df[self.sent_col].apply(lambda x: len(x.split()))


  def _compute_stats(self):
    label_counter = Counter()
    sentence_with_entities = 0
    total_entities = 0


    for entities in self.df[self.ent_col]:
      if entities:
        sentence_with_entities += 1
        for entity in entities:
          label_counter.update([entity["label"]])
          total_entities += 1
    total_sentences = df.shape[0]
    return {
        "total_num_sentence_with_entities": sentence_with_entities,
        "total_annotated_entities": total_entities,
        "entity_distribution_in_article": dict(label_counter),
        "avg_entities_per_sentence": total_entities / total_sentences,
    }

 
  def get_sentence_stats_summary(self):
    return {
        "avg_sent_len": self.df["sent_len"].mean(),
        "min_sent_len": self.df["sent_len"].min(),
        "max_sent_len": self.df["sent_len"].max(),
    }

  def plot_data_stats(self):
    stats  = self._compute_stats()
    print(f"Article Statitics: {stats}")
    print(f"Sentence level Statistics: {self.get_sentence_stats_summary()}")

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    sns.histplot(self.df["sent_len"], ax=axes[0], bins=50, )
    axes[0].set_title("Sentence Length Distribution")

    ent_dist = stats["entity_distribution_in_article"]
    sns.barplot(x=list(ent_dist.keys()), y=list(ent_dist.values()), ax=axes[1])

    axes[1].set_title("Entity Label Distribution")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()