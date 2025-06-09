import pandas as pd
import psycopg
import requests
from typing import List, Tuple
from variables import headers, password, user

pd.options.display.max_columns = None
pd.options.display.max_rows = 10


def aact_query(query:str) -> pd.DataFrame:
    """
    Postgres query to AACT clinical trials DB, returning details of AEs reported for a given clinical trial
    """
    conn = psycopg.connect(dbname="aact",
                           host="aact-db.ctti-clinicaltrials.org",
                           user=user,
                           password=password,
                           port=5432)
    cursor = conn.cursor()

    cursor.execute(query)
    response_df = pd.DataFrame(cursor.fetchall())
    return response_df


def aact_data_gather(nct_id: str, query: str) -> Tuple[str, List, List, List, List]:
    response_df = aact_query(query)
    if not response_df.empty:
        severe_aes_filter = response_df[response_df["event_type"] == "serious"]
        severe_aes = get_set_from_col(full_df=severe_aes_filter, col_name="adverse_event")

        other_aes_filter = response_df[response_df["event_type"] == "other"]
        other_aes = get_set_from_col(full_df=other_aes_filter, col_name="adverse_event")

        # nct_id = response_df["nct_id"][0]
        study_title = response_df["study_title"][0]
        print(study_title)
        aes = get_set_from_col(full_df=response_df, col_name="adverse_event")
        patient_groups = get_set_from_col(full_df=response_df, col_name="ctgov_group_code")
        # group_types = get_set_from_col(full_df=response_df, col_name="event_type") ## Types = ['serious', 'other']
        # TODO Get % affected / see relevance of 'other' vs 'serious'
        return study_title, aes, severe_aes, other_aes, patient_groups
    else:
        return "", [], [], [], []


def display_widget(text: List, placeholder:str, start_statement:str):
    display_text = "<br>".join([f"- {a}" for a in text])

    b = widgets.HTML(
        value = display_text,
        placeholder = placeholder,
        description = 'Scroll',
        disabled=True
    )

    a = widgets.HBox([b], layout=widgets.Layout(height='150px', width='1000px', overflow_y='auto'))

    print(start_statement)
    display(a)


def get_set_from_col(full_df: pd.DataFrame, col_name: str) -> List:
    return list(set(full_df[col_name]))


def query_epmc(query: str, page_size: int = 10) -> List:
    query = f"{query} HAS_FT:Y AND OPEN_ACCESS:Y"
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size
    }

    resp = requests.get(url, params=params)
    resp_json = resp.json()

    if 'resultList' in resp_json:
        pmids = [i['pmid'] for i in resp_json['resultList']['result']]
        return pmids
    else:
        return []


def query_bioc(pmid: str) -> List:
    resp = requests.get(f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/unicode")
    resp_json = resp.json()
    # pprint(resp_json)
    text = []
    for search_hit in resp_json:
        docs = search_hit['documents']
        for doc in docs:
            passages = doc['passages']
            sections = []
            for block in passages:
                section = block['infons']['section_type']
                sections.append(section)
                text.append(block['text'])
                # if section in ['INTRO', 'DISCUSS', 'RESULTS']:
                #     text.append(block['text'])
            # print(list(set(sections)))
    return text
