import os
import re
import requests
import pandas as pd
from tqdm import tqdm

# Define directory paths
BASE_DIR = "/nfs/production/literature/amina-mardiyyah"
DATA_DIR = os.path.join(BASE_DIR, "new_data")

# Load data from CSV
def load_data(filename="intact_data_cl.csv"):
    """
    Load and preprocess the cell line data from CSV.

    Args:
        filename (str): The filename of the CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with relevant columns.
    """
    intact_df = pd.read_csv(os.path.join(DATA_DIR, filename))
    intact_df = intact_df[["LABEL", "Definition", "Altertive term"]].drop(0)
    return intact_df

# Remove duplicates and missing values
def clean_data(df):
    """
    Clean the DataFrame by removing duplicates and missing values.

    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    print(f"Initial data shape: {df.shape}")
    print(f"Number of duplicate LABEL entries: {df['LABEL'].duplicated().sum()}")

    # Drop duplicate rows based on LABEL
    df = df.drop_duplicates(subset=['LABEL']).dropna(subset=["LABEL"])

    print(f"Data shape after cleaning: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    return df

# Construct search query based on LABEL, Alternative Term, and Definition columns
def construct_search_query(row):
    """
    Build a search query for each cell line based on label, alternative term,
    and relevant phrases found in the definition.

    Args:
        row (pd.Series): A row from the DataFrame containing cell line data.

    Returns:
        str: Constructed search query.
    """
    label = row["LABEL"]
    alternative_term = row.get("Altertive term", "")
    definition = row.get("Definition", "")

    # Extract cell-related phrases from definition using regex
    cell_related_phrases = re.findall(r'([A-Za-z0-9\s\-]*cell\sline|[A-Za-z0-9\s\-]*cells?)', definition)
    cell_related_lst = [phrase.strip() for phrase in cell_related_phrases]

    # Construct the full query with "or" syntax
    query_terms = [label] + ([alternative_term] if alternative_term else []) + cell_related_lst
    return " or ".join(query_terms)

# Search EuropePMC for articles matching a query
def search_epmc(query, page_size=10):
    """
    Send a query to EuropePMC and retrieve articles.

    Args:
        query (str): The search query for EuropePMC.
        page_size (int): Number of articles to retrieve per query.

    Returns:
        pd.DataFrame: DataFrame containing retrieved articles with details.
    """
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": query, "format": "json", "pageSize": page_size}
    response = requests.get(base_url, params=params)
    data = response.json()

    # Parse response into a DataFrame
    articles = []
    if 'resultList' in data:
        for article in data['resultList']['result']:
            articles.append({
                "PMCID": article.get('pmcid'),
                "Title": article.get('title'),
                "PubType": article.get("pubType"),
                "IsOpenAccess": article.get('isOpenAccess')
            })

    return pd.DataFrame(articles)

# Aggregate search results from all queries into one DataFrame
def aggregate_search_results(queries):
    """
    Aggregate search results from EuropePMC for all queries.

    Args:
        queries (list): List of search queries.

    Returns:
        pd.DataFrame: Combined DataFrame of all search results.
    """
    all_results = pd.DataFrame()
    for query in tqdm(queries, desc="Searching EPMC for Cell Line articles"):
        result_df = search_epmc(query)
        result_df['search_query'] = query
        all_results = pd.concat([all_results, result_df], ignore_index=True)
    
    # Filter for open-access articles
    all_results = all_results[all_results["IsOpenAccess"] == 'Y']
    return all_results

# Main execution
def main():
    # Load and preprocess data
    intact_df = load_data()
    intact_df = clean_data(intact_df)

    # Construct search queries
    intact_df['search_query'] = intact_df.apply(construct_search_query, axis=1)
    search_queries = intact_df['search_query'].tolist()
    print(f"Total search queries generated: {len(search_queries)}")

    # Perform search and save results
    results_df = aggregate_search_results(search_queries)
    print(f"Total open-access articles found: {results_df.shape[0]}")
    results_df.to_csv(os.path.join(DATA_DIR, "CL-articles_df.csv"), index=False)

# Run the main function
if __name__ == "__main__":
    main()
