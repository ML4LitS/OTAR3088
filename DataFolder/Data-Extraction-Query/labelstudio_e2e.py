import csv
import json
import re
import pandas as pd
import spacy
from typing import List

nlp = spacy.load("en_core_web_sm")

def ls_dictionary_format(path_to_dict: str) -> pd.DataFrame:
    """
    TODO - Aware this function does not consider alternative dictionary formats for now
    Input dictionary is cleaned to comply with Label studio requirements
    """
    df = pd.read_csv(path_to_dict, sep="\t")
    df.columns = ["label", "id", "term"]
    df_clean = pd.concat([df["term"], df["label"]], axis=1, ignore_index=True)
    df_clean.columns = ["term", "label"]

    # Drop short terms in dictionary due to ambiguity
    # TODO - Q? Should terms like 'cell' be dropped also, or are they helpful markers for review?
    df_clean['length'] = df_clean.apply(lambda x: len(x['term']) > 2, axis=1)
    df_clean = df_clean[df_clean['length'] == True]
    df_clean = df_clean.drop(columns=['length'])

    return df_clean

def smart_boundary_regex(term: str) -> re.Pattern:
    """
    Allow letters, digits, spaces, hyphens, and slashes in terms
    For word boundaries - allow non-letter/digit character before and after
    """
    escaped = re.escape(term)
    pattern = rf"(?<!\w){escaped}(?:s|’s)?(?!\w)"
    return re.compile(pattern, re.IGNORECASE)

def lemmatize_term(term: str) -> str:
    """
    Return lemmatized form of term
    """
    lower = nlp(term.lower())
    lemmat = ' '.join([token.lemma_ for token in lower])
    return lemmat

def ls_formatter(dict_file: str, texts_file: str, output_json: str):
    """
    Formatting
    """

    # Map lemmatized terms to input dictionary terms
    term_lemma_map = {}
    with open(dict_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            term = row['term'].lower()
            label = row['label']
            lemma = lemmatize_term(term)
            term_lemma_map[lemma] = (term, label)

    # Read and process input texts
    with open(texts_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    annotations = []

    for text in texts:
        doc = nlp(text)
        tokens = [token for token in doc]
        ## Print next line to display lemmatizations
        # lemmatized_text = ' '.join([token.lemma_.lower() for token in tokens])
        results = []

        # Try to match given term's lemma in lemmatized version of the text
        for lemma, (orig_term, label) in term_lemma_map.items():
            lemma_tokens = lemma.split()
            n = len(lemma_tokens)
            for i in range(len(tokens) - n + 1):
                window = tokens[i:i + n]
                window_lemmas = [t.lemma_.lower() for t in window]

                if window_lemmas == lemma_tokens and window:
                    start_char = window[0].idx
                    end_char = window[-1].idx + len(window[-1])
                    results.append({
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": start_char,
                            "end": end_char,
                            "text": text[start_char:end_char],
                            "labels": [label]
                        }
                    })

        res = {
            "data": {"text": text},
            "annotations": [{"result": results}] if results else []
        }
        annotations.append(res)

    # Save annotations output
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    hits = len(annotations[0]['annotations'][0]['result'])
    print(f"Saved {hits} dictionary annotation(s) to {output_json}")

def collate_dictionaries(dictionaries: List, path_to_output: str):
    """
    Collating provided dictionaries to one master df for annotation
    :param dictionaries: paths to input tsv
    :param path_to_output
    """
    master_df = pd.DataFrame(columns=['term', 'label'])
    for dictionary in dictionaries:
        df_clean = ls_dictionary_format(path_to_dict=dictionary)
        master_df = pd.concat([master_df, df_clean])
    master_df.to_csv(path_to_output, sep="\t", index=False)

if __name__ == "__main__":
    # Input dictionaries sourced from ChEMBL and BRENDA
    cell = "./cell_df.tsv"
    bcell = "./brendacell_df.tsv"
    tissue = "./tissue_df.tsv"
    btissue = "./brendatissue_df.tsv"

    master_path = './output/labelstudio/master_dictionary.tsv'

    # Aggregate input dictionaries, so multiple are annotated in one run
    collate_dictionaries(dictionaries=[cell, tissue, bcell, btissue], path_to_output=master_path)
    # Annotate texts using collated dictionaries, write to file in Label studio format
    ls_formatter(dict_file=master_path, texts_file="./output/labelstudio/sample.txt", output_json="./output/labelstudio/test.json")