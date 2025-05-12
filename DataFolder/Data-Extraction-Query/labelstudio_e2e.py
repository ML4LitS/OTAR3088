import csv
import json
import re
import pandas as pd
import spacy
from typing import Dict

nlp = spacy.load("en_core_web_sm")

def ls_dictionary_format(path_to_dict: str, path_to_output: str) -> pd.DataFrame:
    """
    :return: Refactored dictionary pd.DataFrame in LabelStudio format
    Writing tsv to file of cleaned dictionary
    """
    # TODO - Ensure there are no other input dictionary formats to account for re. column names
    df = pd.read_csv(path_to_dict, sep="\t")
    df.columns = ["label", "id", "term"]
    df_clean = pd.concat([df["term"], df["label"]], axis=1, ignore_index=True)
    df_clean.columns = ["term", "label"]

    # Drop short terms in dictionary due to ambiguity
    df_clean['length'] = df_clean.apply(lambda x: len(x['term']) > 2, axis=1)
    df_clean = df_clean[df_clean['length'] == True]
    df_clean.drop(columns=['length'])

    df_clean.to_csv(path_to_output, sep="\t", index=False)
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
        # Print to display lemmatizations
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

    # --- Save output ---
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(annotations)} dictionary annotation(s) to {output_json}")

if __name__ == "__main__":
    # Aggregate input dictionaries, so multiple are annotated in one run
    # Input dictionaries sourced from ChEMBL
    cell = "./cell_df.tsv"
    tissue = "./tissue_df.tsv"
    master_df = pd.DataFrame
    for dictionary in [cell, tissue]:
        df_clean = ls_dictionary_format(path_to_dict="./cell_df.tsv", path_to_output="./output/labelstudio/cell_ls_dict.tsv")
    ls_formatter(dict_file="./output/labelstudio/cell_ls_dict.tsv", texts_file="./output/labelstudio/sample.txt", output_json="./output/labelstudio/test.json")