import os
import re
import requests
import pandas as pd
import ast
from tqdm import tqdm
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set directories and constants
# TODO- SAVEDIR could be user input like argparse? depending on the perceived use of the script
SAVE_DIR = "/nfs/production/literature/amina-mardiyyah/data"
OLS_URL = "https://www.ebi.ac.uk/ols4/api/ontologies/cl/terms"
MAX_TERMS_PER_QUERY = 5  # Max number of terms allowed in a single query--> Cell type synonymns can be too many for some terms
SESSION = requests.Session()

#1st part of Script: Extracting Human Cell types from OLS
# Parameters for OLS API query #TODO - not sure if this would ever be needed but the dict could be constructed in a function so it is instantiated by the user of the script?
ols_params = {
    "pageSize": 100,      # Max records per query
    "q": "human",         # Only human cell types
    "taxonomy": 9606,     # NCBI taxonomy ID for humans
    "page": 1,            # Start page
}

# TODO- from typing import Dict ^ (just giving a type hint example)
# TODO- def fetch_terms_per_page(session: requests.session, url: str, params: Dict[str,str]) -> Dict[str,str]:
def fetch_terms_per_page(session, url, params): # TODO- Maybe rename the function
    """Fetch a single page of terms from the OLS API."""
    response = session.get(url, params=params)
    response.raise_for_status()
    return response.json()

# Initialize data and calculate the total number of pages to query
init_data = fetch_terms_per_page(SESSION, OLS_URL, ols_params)
total_pages = init_data["page"]["totalPages"]
print(f"Total pages of human cell type terms: {total_pages}")

# TODO- Could generalise the function as well here so it is just a call to OLS, specific ontology or a tag 'cell type' is passed to the function to entend the URL - /cl/terms
# Collect all human cell type terms from OLS
def fetch_all_cell_terms(total_pages):
    """Fetch all human cell terms across all pages."""
    cell_terms = []

    for page in tqdm(range(1, total_pages + 1), desc="Fetching Cell Type terms", unit="page"):
        ols_params["page"] = page
        page_data = fetch_terms_per_page(SESSION, OLS_URL, ols_params)

        if "_embedded" in page_data and "terms" in page_data["_embedded"]:
            cell_terms.extend([{
                "term": term["label"],
                "synonym": term.get("synonyms"),
                "description": term.get("description")
            } for term in page_data["_embedded"]["terms"]])

    print(f"Total human cell types collected: {len(cell_terms)}")
    return pd.DataFrame(cell_terms)

# Fetch terms and save to CSV #TODO- Sufficiently shown in the code no need for comment
cell_terms_df = fetch_all_cell_terms(total_pages)
print(f"Saving cell types to CSV at {SAVE_DIR}")
cell_terms_df.to_csv(os.path.join(SAVE_DIR, "C-types_df.csv"), index=False)


#2nd Part of script: Using the extracted terms to form a search query for EPMC
# Function to split large lists into smaller chunks #TODO- Same as before, think the code explains enough
def split_into_chunks(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from a list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# Construct search queries with a maximum of MAX_TERMS_PER_QUERY terms per query #TODO- remove comment
def construct_search_queries(df, max_terms_per_query=MAX_TERMS_PER_QUERY): #TODO- Would avoid 'df' as a name
    """Construct search queries from terms and synonyms, chunking as necessary."""
    queries = []
    
    for _, row in df.iterrows():
        term = row["term"]
        synonym_list = ast.literal_eval(row["synonym"]) if pd.notnull(row["synonym"]) else []
        
        # Combine terms and synonyms into a single list, then chunk
        all_terms = [term] + synonym_list
        for chunk in split_into_chunks(all_terms, max_terms_per_query):
            queries.append(" or ".join(f"'{item}'" for item in chunk))

    return queries

# Construct queries
cell_queries = construct_search_queries(cell_terms_df)

# Function to fetch articles from EuropePMC based on a query
def extract_articles(query, page_size=10):
    """Fetch articles from EuropePMC based on a search query."""
    epmc_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "pageSize": page_size,
        "query": query,
        "format": "json"
    }
    response = SESSION.get(epmc_url, params=params)
    
    # Handle errors
    if response.status_code != 200:
        print(f"Error: {response.status_code} for query: {query}") #TODO- You could print the response error message to?
        return pd.DataFrame()

    data = response.json()
    articles = []

    if "resultList" in data:
        for article in data["resultList"]["result"]:
            articles.append({
                "title": article.get("title"),
                "PMCID": article.get("pmcid"),
                "isOpenAccess": article.get("isOpenAccess"),
                "pubtype": article.get("pubType"),
                "search_query": query
            })
    
    return pd.DataFrame(articles)

# Concurrently fetch articles for all queries
def extract_articles_concurrently(queries, max_workers=10):
    """Fetch articles from EuropePMC concurrently for multiple queries."""
    all_articles = pd.DataFrame()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_articles, query) for query in queries]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting articles"):
            result_df = future.result()
            if not result_df.empty:
                all_articles = pd.concat([all_articles, result_df], ignore_index=True)

    return all_articles

#TODO- I guess as in the cell line script, I would pull the function methods out into a script
#TODO- At the mo this reads like a jupyter notebook? I guess worth thinking about how these functions are to be used in future
# Extract articles and save results to CSV
final_articles_df = extract_articles_concurrently(cell_queries)
print(f"Total articles found: {final_articles_df.shape}")
final_articles_df.to_csv(os.path.join(SAVE_DIR, "CT-articles_df.csv"), index=False)
