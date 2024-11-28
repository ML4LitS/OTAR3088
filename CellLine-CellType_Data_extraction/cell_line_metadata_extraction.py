import os
import re
import requests
import pandas as pd
from tqdm import tqdm

# Define directory paths
# TODO- Potentially make this a directory in git so it's usable for others as default
BASE_DIR = "/nfs/production/literature/amina-mardiyyah"
DATA_DIR = os.path.join(BASE_DIR, "new_data")

# Load data from CSV
# TODO- Not needed to name the file as input - ties the function to that source only
# TODO- Alternative for the """..."""s and declaring a default input could be using the typing package
# TODO- Typing instead would look like this on line 15 - def load_data(filename: pd.DataFrame) -> pd.DataFrame:
# TODO- I really like the type hints ^ it saves you some lines of code if nothing else :)
def load_data(filename="intact_data_cl.csv"):
    """
    Load and preprocess the cell line data from CSV.

    Args:
        filename (str): The filename of the CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with relevant columns.
    """
    # TODO- I think more pythonic style would be to name the df variables descriptively so more 'input_df', 'clean_df' blah blah
    # TODO- Logic for this is to preserve the data in case you need / traceable
    intact_df = pd.read_csv(os.path.join(DATA_DIR, filename))
    # TODO - How is the line below different to the clean_data function? Maybe it could belong there if it is too 'cleaning'
    intact_df = intact_df[["LABEL", "Definition", "Altertive term"]].drop(0)
    return intact_df

# TODO- I think if your code is descriptive enough you don't need these comments outside of methods
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
def construct_search_query(row): #TODO- Again typing could save 10 lines
    """
    Build a search query for each cell line based on label, alternative term,
    and relevant phrases found in the definition.

    Args:
        row (pd.Series): A row from the DataFrame containing cell line data.

    Returns:
        str: Constructed search query.
    """
    label = row["LABEL"]
    alternative_term = row.get("Altertive term", "") # TODO- Would maybe add the delcaration names for clarity
    definition = row.get("Definition", "")           # TODO- So (key="Definition", default="")

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
    # TODO- Could check if resp.status_code == 200: continue one, else: see what the failure is?
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
    # TODO- else: blah... To cover in code tests, should an exception be caught
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

    # TODO- Would filter for this much earlier, maybe on returning the query from epmc if it isnt open then don't generate the result
    # Filter for open-access articles
    all_results = all_results[all_results["IsOpenAccess"] == 'Y']
    return all_results

# Main execution #TODO- Not needed
def main():
    # Load and preprocess data
    intact_df = load_data() # TODO - Maybe name these based on the file format rather than the source
    intact_df = clean_data(intact_df)

    # Construct search queries
    intact_df['search_query'] = intact_df.apply(construct_search_query, axis=1)
    search_queries = intact_df['search_query'].tolist()
    print(f"Total search queries generated: {len(search_queries)}")

    # Perform search and save results
    results_df = aggregate_search_results(search_queries)
    print(f"Total open-access articles found: {results_df.shape[0]}")
    results_df.to_csv(os.path.join(DATA_DIR, "CL-articles_df.csv"), index=False)

# TODO- All in I would consider splitting this to be a specific run script for cell line, with a functions file externally imported
# TODO- Then all the formatting and calling functions can be generic, built on as needed?

# Run the main function # TODO- Don't think this comment is needed
if __name__ == "__main__":
    main()
