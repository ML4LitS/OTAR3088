import re, json, os, requests, time
from tqdm import tqdm
import split2sent_par

from argparse import ArgumentParser
import logging
from pathlib import Path
import pandas as pd


parser = ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="path to the file to be used for data extraction. Only `csv` format currently supported")
parser.add_argument("--log_dir", type=str, required=False, help="path to directory for logging")
parser.add_argument("--output_dir", type=str, default="./", help="path to directory to save extracted data")
parser.add_argument("--format2extract", type=str, default="sent",
                    help="type of data extraction. Data can be extracted split in sentences or split in paragraphs. Use `sent` for sentences and `par` for paragraph. For paragraph, figure caption and section headers are included")
parser.add_argument("--save_name", type=str, required=True, help="name for saving dataset")

# initialise args
args = parser.parse_args()

if not args.log_dir:
    logs_path = Path(args.output_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
else:
    logs_path = args.log_dir

# Configure logging
logging.basicConfig(
    filename=f"{logs_path}/{args.save_name}.log",
    filemode="w",
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.INFO
)


print("reading data.......")
metadata = pd.read_csv(args.data_path)
logging.info(f"Current data parsed has shape: {metadata.shape}")


def construct_search_query(row: pd.Series) -> str:
    """
    construct search query for entity + all synonyms in row
    """
    term = row["term"]
    term = f'"{term}"'

    # TODO - Check if typo is how column name is spelt in csv
    synonym = [f'"{i.strip()}"' for i in row['synonymn'].split(",")]

    query = " OR ".join([term] + synonym)
    query = f"({query})"

    return query


metadata['search_query'] = metadata.apply(construct_search_query, axis=1)
logging.info(metadata.head())


query_list = metadata.search_query.tolist()
len(query_list)


def search_epmc(query: str, page_size: int = 25) -> pd.DataFrame:
    """
    # Search EuropePMC using constructed entity query
    # not implementing cursor-mark, fetching all possible results isn't important
    """
    full_search_query = f"{query} HAS_FT:Y AND OPEN_ACCESS:Y AND LICENSE:CC"  # search full text + open access articles
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": full_search_query,
        "format": "JSON",
        "pageSize": page_size
    }
    
    resp = requests.get(base_url, params=params)
    data = resp.json()
    try:
        if "resultList" in data:
            article_list = [
                (
                    article.get("pmcid", None),
                    article.get("title", None),
                    article.get("pubType", None)
                ) for article in data["resultList"]["result"]
            ]
        else:
            article_list = []
        return pd.DataFrame(article_list, columns=['PMCID', 'Title', 'PubType'])

    except requests.exceptions.RequestException as e:
        print(f"Error loading results from ePMC. See error: {e}")
        return pd.DataFrame(columns=['PMCID', 'Title', 'PubType'])


# test with sample query, printing to terminal
print("Example search results for a sample query:")
print("*************************************")
print(search_epmc(query_list[0]))


# Search for all queries and aggregate results into one DataFrame
articles_df = pd.DataFrame()
 

for query in tqdm(query_list, desc="Searching ePMC for Cell Line articles----->"):
    result_df = search_epmc(query)
    result_df['search_query'] = query
    articles_df = pd.concat([articles_df, result_df], ignore_index=True)

print(f"resulting dataframe when all query is applied has shape: {articles_df.shape}")
logging.info(articles_df.head(10))


def fetch_epmc_batches(query: str, page_size: int = 100, delay: int = 1):
    """
    Fetch batches of articles from EuropePMC. *not used by default in script*
    """
    query = f"{query} HAS_FT:Y AND OPEN_ACCESS:Y"
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    cursor_mark = "*"  # Initial cursor for the first batch

    while True:
        params = {
            "query": query,
            "format": "json",
            "pageSize": page_size,
            "cursorMark": cursor_mark,
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        if 'resultList' in data and 'result' in data['resultList']:
            yield data['resultList']['result']
        else:
            break

        # Check for the next cursorMark
        next_cursor_mark = data.get("nextCursorMark")
        if not next_cursor_mark or next_cursor_mark == cursor_mark:
            break  # No more results or cursor hasn't advanced

        cursor_mark = next_cursor_mark
        time.sleep(delay)  # Avoid overloading API


def search_epmc(query: str, page_size: int = 25, delay: int = 1) -> pd.DataFrame:
    """
    Process all request batches
    """
    articles_list = []

    for batch in fetch_epmc_batches(query, page_size=page_size, delay=delay):
        for article in batch:
            title = article.get('title')
            pmcid = article.get('pmcid')
            article_type = article.get("pubType")
            articles_list.append((pmcid, title, article_type))

    all_articles_df = pd.DataFrame(articles_list, columns=['PMCID', 'Title', 'PubType'])
    return all_articles_df


# check for missing values
print(f'Count check for missing values: {str(articles_df.isnull().sum())}')

# Check for missing values in the 'PMCID' column
if articles_df['PMCID'].isnull().any():
    print("Missing values detected in 'PMCID' column. Dropping rows with missing values...")
    articles_df = articles_df[articles_df['PMCID'].notnull()]
    logging.info(f"Data shape after dropping missing values: {articles_df.shape}")
else:
    print("No missing values detected in 'PMCID' column.")

# Check for duplicate values in the 'PMCID' column
if articles_df['PMCID'].duplicated().any():
    print("Duplicate values detected in 'PMCID' column. Dropping duplicate rows...")
    articles_df = articles_df.drop_duplicates(subset='PMCID', keep='first')
    logging.info(f"Data shape after dropping duplicates: {articles_df.shape}")
else:
    print("No duplicate values detected in 'PMCID' column.")


# check for retracted publications
if (articles_df['PubType'] == 'retraction of publication').any():
    # dropping retracted articles
    articles_df = articles_df.query("PubType!='retraction of publication'")
    logging.info(f"Data shape after dropping retracted articles: {articles_df.shape}")


if args.format2extract == 'sent': 
    print("Sentenciser in progress")
    func_call = split2sent_par.sentencise_articles
    df = split2sent_par.parallel_process_articles(articles_df, func_call)
    print(df.head())
    df['Sentences'] = df['Sentences'].apply(split2sent_par.postprocess_sentences)
    print("Sentenciser completed")
    logging.info(f"Sentencised data has shape: {df.shape}")


elif args.format2extract == "par":
    print("Extracting articles paragraphs and section_titles")
    func_call = split2sent_par.process_in_paragraph
    df = split2sent_par.parallel_process_articles(articles_df, func_call, process_par=True)
    print("Paragraph extraction completed")
    logging.info(f"Data split in paragraphs has shape: {df.shape}")


print(f"Saving final dataframe to directory: {args.output_dir} ---> with name: {args.save_name}")
df.to_csv(f"{os.path.join(args.output_dir, args.save_name)}.csv", index=False)
