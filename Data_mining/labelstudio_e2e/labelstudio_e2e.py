import csv
import json
import html, re
import pandas as pd
import requests
import spacy
import textwrap
from tqdm import tqdm
# from label_studio_sdk import Client, LabelStudio
from random import randrange
from typing import Dict, List, Optional
from epmc_to_json import *

nlp = spacy.load("en_core_web_sm")


def search_epmc(query: str, page_size: int = 10) -> pd.DataFrame:
    # Pulled from Amina's script
    full_search_query = f"{query} HAS_FULLTEXT:Y AND OPEN_ACCESS:Y AND LICENSE:CC"
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": full_search_query,
        "format": "JSON",
        "pageSize": page_size
    }

    response = requests.get(base_url, params=params)
    data = response.json()
    try:
        if "resultList" in data:
            article_list = [
                (
                    article.get("pmcid", None),
                    article.get("title", None),
                    article.get("pubType", None),
                    article.get("journalTitle", None)
                ) for article in data["resultList"]["result"]
            ]
        else:
            article_list = []
        return pd.DataFrame(article_list, columns=['PMCID', 'Title', 'PubType', 'Journal'])

    except requests.exceptions.RequestException as e:
        print(f"Error loading results from EPMC. See error: {e}")
        return pd.DataFrame(columns=['PMCID', 'Title', 'PubType', 'Journal'])


def search_pmcids(search_queries: List) -> List:
    '''
    Run to grab all full texts for search terms passed in search_queries List
    - PMCIDs are collected on the basis that a full text article from a journal unique to
      the full list of articles returned is found
    - This process is repeated until the conditions are met
    '''
    pmcids = []
    journal_check = []
    # for search_query in ['HeLa', 'MIO-M1', 'synthetic tissue']:
    for search_query in search_queries:
        compare = len(pmcids)
        print(f"Selecting one '{search_query}'-related paper for full-text extraction from ePMC")
        res_df = search_epmc(search_query)
        # Check articles returned are journal articles
        res_df = res_df[res_df['PubType'].str.contains('journal article')]

        while len(pmcids) <= compare:
            # Grab random PMCID from result df for full text extraction
            selected_article = res_df.iloc[randrange(len(res_df) - 1)]
            if selected_article['Journal'] not in journal_check:
                print(selected_article)
                journal_check.append(selected_article['Journal'])
                pmcids.append(selected_article['PMCID'])
                print('\n')
            else:
                # We want to ensure a variety of journals are used, try again
                continue
    print('Completed search queries')
    return pmcids


def collect_full_text(pmcids: List[str]) -> Dict:
    """
    Passed a list of PMCIDs, the full text XML from ePMC is fetched, cleaned,
    and filtered to keep only relevant sections for NER. The final output for
    each PMCID is a single block of text.

    Args:
        pmcids: A list of PMCIDs as strings.

    Returns:
        A dictionary of the structure {pmcid: single_block_of_text, ...}
    """
    cleaned_full_texts = {}

    okHeaders = [
        'title',
        'abstract',
        'intro',
        'experiment',
        'methods',
        'analysis',
        'result',
        'discussion',
        'conclusion',
        'opinion'
    ]

    for pmcid in tqdm(pmcids):
        # Assuming get_epmc_full_text_xml(pmcid) returns the XML string
        xml_string = get_epmc_full_text_xml(pmcid)
        if not xml_string:
            continue

        header_tags = ['title', 'abstract']
        pattern = re.compile(r'(<({tags})[^>]*>)(.*?)(</\2>)'.format(tags='|'.join(header_tags)))
        text = pattern.sub(r'\1***\3***\4', xml_string)

        block_level_tags = ['p', 'title', 'sec', 'abstract', 'contrib']
        text = re.sub(r'</({tags})>'.format(tags='|'.join(block_level_tags)), '\n\n', text)

        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)

        # STAGE 5: Filtering Logic
        chunks = text.split('\n\n')
        final_content = []
        keep_section = False

        for chunk in chunks:
            clean_chunk = chunk.strip()
            if not clean_chunk:
                continue

            is_header = clean_chunk.startswith('***') and clean_chunk.endswith('***')
            header_text = clean_chunk.replace('***', '').strip().lower()

            if is_header:
                if any(ok_header in header_text for ok_header in okHeaders):
                    keep_section = True
                    # Add the cleaned header text to results
                    final_content.append(header_text.capitalize())
                else:
                    keep_section = False
            elif keep_section:
                final_content.append(clean_chunk)

        final_output = " ".join(final_content)

        cleaned_full_texts[pmcid] = final_output

    return cleaned_full_texts


def ls_dictionary_format(path_to_dict: str) -> pd.DataFrame:
    """
    TODO - Aware this function does not consider other dictionary formats for now
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


def ls_formatter(dict_file: str, texts_file: str, output_json: str, pmcid: Optional[str] = None):
    """
    dict_file: path to dictionary file to annotate with
    texts_file: text to annotate
    output_json: annotations in text, in labelstudio format for upload
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
        if pmcid:
            text = text + '\n' + f'Source paper: {pmcid}'
        res = {
            "data": {"text": text},
            "annotations": [{"result": results}] if results else []
        }
        annotations.append(res)

    # Save annotations output
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    # hits = len(annotations[0]['annotations'][0]['result'])
    # print(f"Saved {hits} dictionary annotation(s) to {output_json}")


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


# def import_task():
#     MY_KEY = 'Insert LabelStudio Key'
#
#     # Initialize the Client with the URL and API Key
#     client = Client(url='http://localhost:8080', api_key=MY_KEY)
#
#     # Ensure that the connection is valid
#     try:
#         client.check_connection()
#         print("Successfully connected to Label Studio!")
#     except Exception as e:
#         print(f"Connection failed: {e}")
#
#     # Correctly fetch projects using client.get_projects()
#     try:
#         projects = client.get_projects()  # Fetch all projects
#         if not projects:
#             print("No projects found!")
#         else:
#             print("Fetched projects:", projects)
#     except Exception as e:
#         print(f"Failed to fetch projects: {e}")
#
#     # Example: Using the client to import tasks into a specific project
#     project_id = 1  # Replace with your actual project ID
#     tasks_data = [{"text": "Some text"}, {"text": "More text"}]
#
#     try:
#         # Fetch the specific project by ID using client.get_project()
#         project = client.get_project(project_id)
#         # Import tasks into the project
#         project.import_tasks(tasks_data)  # Import tasks into the project
#         print(f"Successfully imported tasks into project {project_id}")
#     except Exception as e:
#         print(f"Failed to import tasks: {e}")


def write_ls_textfile(input_text: str, path_to_outfile: str):
    with open(path_to_outfile, "w+") as f:
        f.write(input_text)
    f.close()


if __name__ == "__main__":
    # TODO - Add argparse for future usability
    # Input dictionaries sourced from ChEMBL and BRENDA
    cell = "./cell_df.tsv"
    bcell = "./brendacell_df.tsv"
    tissue = "./tissue_df.tsv"
    btissue = "./brendatissue_df.tsv"

    master_path = './output/labelstudio/master_dictionary.tsv'

    # Aggregate input dictionaries, so multiple are annotated in one run
    collate_dictionaries(dictionaries=[cell, tissue, bcell, btissue], path_to_output=master_path)

    # Annotate texts using collated dictionaries, write to file in Label studio format
    ls_formatter(dict_file=master_path, texts_file="./output/labelstudio/sample.txt",
                 output_json="./output/labelstudio/test.json", pmcid=None)

    # Cell line names derived from ChEMBL assay descriptions
    # Grabbing the top and bottom 5 for variation in papers seen
    cellline_freqs = '/Users/withers/GitProjects/OTAR3088/Data_mining/chembl_sql/cell_line/assay_cell_type_freq.csv'
    celllines = pd.read_csv(cellline_freqs)
    top = celllines['assay_cell_type'].to_list()[:5]
    bottom = celllines['assay_cell_type'].to_list()[-5:]
    search_queries = top + bottom + ['synthetic tissue']

    pmcids = search_pmcids(search_queries=search_queries)
    cleaned_full_texts = collect_full_text(pmcids=pmcids)

    print('\nWriting annotations to file...\n')
    for k, v in tqdm(cleaned_full_texts.items()):
        annotated_path = f'./output/labelstudio/{k}_annotation.txt'
        ls_json_path = f'./output/labelstudio/{k}_annotated.json'
        write_ls_textfile(input_text=v, path_to_outfile=annotated_path)
        ls_formatter(dict_file=master_path,
                     texts_file=annotated_path,
                     output_json=ls_json_path,
                     pmcid=k)