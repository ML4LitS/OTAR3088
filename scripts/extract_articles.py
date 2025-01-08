import re,json,os, requests, time
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

#initialise args
args = parser.parse_args()

if not args.log_dir:
    logs_path = Path(args.output_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

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



#function to construct query
def construct_search_query(row):
    term = row["term"]
    term = f'"{term}"'

    synonymn = [f'"{i.strip()}"' for i in row['synonymn'].split(",")]

    query = " OR ".join([term] + synonymn)
    query = f"({query})"

    return query


metadata['search_query'] = metadata.apply(construct_search_query, axis=1)
logging.info(metadata.head())


query_list = metadata.search_query.tolist()
len(query_list)


# Function to search EuropePMC using entity of interest query and store results in a DataFrame
#not implementing cursormark as all possible results isn't important
def search_epmc(query, page_size=25):
    query = f"{query} HAS_FT:Y AND OPEN_ACCESS:Y" #strictly extracting only open access and fulltext articles
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size, #pagesize params can be changed if more results are desired
    }

    response = requests.get(base_url, params=params)
    data = response.json()
    

    # List to store article data
    articles_list = []

    if 'resultList' in data:
        for article in data['resultList']['result']:
            title = article.get('title')
            pmcid = article.get('pmcid')
            article_type = article.get("pubType") 
            articles_list.append((pmcid, title,
                                  article_type, 
                                  ))

    # Create a DataFrame with the articles
    df = pd.DataFrame(articles_list, columns=['PMCID', 'Title', 
                                              'PubType'
                                             ])
    
    return df

#testing function with sample query and printing to terminal
print("Displaying example search results for a sample query")
print("*************************************")
print(search_epmc(query_list[0]))


# Search for all query and aggregate results into one DataFrame
query_df = pd.DataFrame()
 

for query in tqdm(query_list, desc="Searching EPMC for Cell Line articles----->"):
    result_df = search_epmc(query)
    result_df['search_query'] = query
    query_df = pd.concat([query_df, result_df], ignore_index=True)

print(f"resulting dataframe when all query is applied has shape: {query_df.shape}")
logging.info(query_df.head(10))


# Separate function to fetch batches of articles from EuropePMC----> not used by default in script
def fetch_epmc_batches(query, page_size=100, delay=1):
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
            break  # Exit loop if no more results or cursor hasn't advanced

        cursor_mark = next_cursor_mark
        time.sleep(delay)  # Delay between requests to avoid overloading the API

# Function to process all batches and return a DataFrame
def search_epmc(query, page_size=25, delay=1):
    articles_list = []

    # Fetch batches using the generator
    for batch in fetch_epmc_batches(query, page_size=page_size, delay=delay):
        for article in batch:
            title = article.get('title')
            pmcid = article.get('pmcid')
            article_type = article.get("pubType")
            articles_list.append((pmcid, title, article_type))

    # Create a DataFrame with the articles
    df = pd.DataFrame(articles_list, columns=['PMCID', 'Title', 'PubType'])
    return df


#check for missing values
print(query_df.isnull().sum())


# Check for missing values in the 'PMCID' column
if query_df['PMCID'].isnull().any():
    print("Missing values detected in 'PMCID' column. Dropping rows with missing values...")
    query_df = query_df[query_df['PMCID'].notnull()]
    logging.info(f"Data shape after dropping missing values: {query_df.shape}")
else:
    print("No missing values detected in 'PMCID' column.")

# Check for duplicate values in the 'PMCID' column
if query_df['PMCID'].duplicated().any():
    print("Duplicate values detected in 'PMCID' column. Dropping duplicate rows...")
    query_df = query_df.drop_duplicates(subset='PMCID', keep='first')
    logging.info(f"Data shape after dropping duplicates: {query_df.shape}")
else:
    print("No duplicate values detected in 'PMCID' column.")


#check for retracted publications
if (query_df['PubType'] == 'retraction of publication').any():
    #dropping retracted articles
    query_df = query_df.query("PubType!='retraction of publication'")
    logging.info(f"Data shape after dropping retracted articles: {query_df.shape}")






if args.format2extract == 'sent': 
    print("Sentenciser in progress")
    df = split2sent_par.sentencise_in_parallel(query_df)
    print("Sentenciser completed")
    logging.info(f"Sentencised data has shape: {df.shape}")


elif args.format2extract == "par":
    print("Extracting articles paragraphs and section_titles")
    df = split2sent_par.get_df(query_df)
    print("Extraction completed")
    logging.info(f"Data split in paragraphs has shape: {df.shape}")


#save df to path

print(f"Saving final dataframe to directory: {args.output_dir} ---> with name: {args.save_name}")
df.to_csv(f"{os.path.join(args.output_dir, args.save_name)}.csv", index=False)


