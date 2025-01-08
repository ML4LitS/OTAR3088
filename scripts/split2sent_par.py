
from tqdm import tqdm
import pandas as pd
import time

import spacy, requests

from bs4 import BeautifulSoup as bs
from concurrent.futures import ThreadPoolExecutor, as_completed




#load spacy model for sentenciser
nlp = spacy.load("en_core_web_lg") #model can be changed
nlp.max_length = 10_000_000 
nlp.disable_pipe("parser") #not necessary but makes pipeline run faster
nlp.enable_pipe("senter")

#function to fetch articles xml 
def get_xml(pmcid):
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an error if the response code is not 200
        return bs(response.content, "lxml-xml")
        
    except requests.exceptions.HTTPError as http_err: 
        print(f"HTTP error occurred: {http_err}")

    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    print(f"Failed to retrieve XML file for PMCID: {pmcid}") #articles can be available in other formats other than xml, we can skip these
    return None
            

#function to fetch paragraph text from xml--> used for sentenciser
def get_paragraphs_text(xml):
    """
    Extracts the title and full text from the given XML.
    """
    if xml:
        try:
            title = xml.find("article-title").text if xml.find("article-title") else "This article is missing a title"
            full_text = " ".join([p.text for p in xml.find_all("p") if p.text])  # Handle non-empty paragraphs
            return title, full_text
        except Exception as e:
            print(f"Error processing XML: {e}")
            return "Error extracting title", ""
    else:
        return "No XML provided", ""
    
#function to sentencise articles
def sentencise_articles(row,nlp=nlp):
    articles = []
    pmcid = row["PMCID"]
    xml = get_xml(pmcid)
    title, full_text = get_paragraphs_text(xml)
    if full_text:
        doc = nlp(full_text)
        sentences = [sent.text for sent in doc.sents]
        for sent in sentences:
            articles.append({"PMCID": pmcid, "Title": title, "Sentences": sent})
    return articles



def sentencise_in_parallel(df): 
    articles = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = list(tqdm(
            executor.map(sentencise_articles, [row for _, row in df.iterrows()]),
            desc="Processing articles",
            total=len(df)
        ))
        for result in futures:
            articles.extend(result)  # Collect all results into a single list
    
    return pd.DataFrame(articles)










# Function to process the sections, paragraphs, and figure captions of an article
def process_article(pmcid):
    """
    Processes an article's sections, paragraphs, and figure captions, extracting them into structured rows.

    Args:
        pmcid (str): The PMCID of the article.

    Returns:
        list: A list of tuples containing PMCID, section title, and paragraph text.
    """
    soup = get_xml(pmcid)
    
    # If XML fetch fails, return empty result
    if soup is None:
        return []

    rows = []
    
    # Get abstract first (if available)
    abstract = soup.find("abstract")
    if abstract:
        # Extract abstract text from <p> tags within the abstract
        abstract_paragraph = abstract.find("p")
        if abstract_paragraph:
            abstract_text = abstract_paragraph.get_text(separator=" ").strip()
            rows.append((pmcid, "Abstract", abstract_text))
    
    # Process sections and paragraphs
    for section in soup.find_all("sec", recursive=True):  # Process top-level sections only
        section_title = section.title.text if section.title else "Unnamed section"
        
        # Extract all paragraphs within the section (including subsections recursively)
        section_paragraphs = section.find_all("p", recursive=True)
        for paragraph in section_paragraphs:
            paragraph_text = paragraph.get_text(separator=" ").strip()
            if paragraph_text:
                rows.append((pmcid, section_title, paragraph_text))
        
        # Handle figure captions within the section
        for figure in section.find_all("fig", recursive=True):  # Recursive to capture nested figures
            caption = figure.find("caption")
            if caption:
                caption_text = caption.get_text(separator=" ").strip()
                figure_title = f"{section_title} - Figure Caption"
                rows.append((pmcid, figure_title, caption_text))
    
    return rows
        

# Function to fetch and process articles concurrently
def fetch_and_process_articles_concurrently(pmcids, max_workers=4):
    """
    Fetch and process articles concurrently using threading.
    
    Args:
        pmcids (list): List of PMCIDs to process.
        max_workers (int): Maximum number of threads to use.
    
    Returns:
        pd.DataFrame: DataFrame with PMCID, Section, and Paragraph_text columns.
    """
    result = []
    
    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pmcid = {executor.submit(process_article, pmcid): pmcid for pmcid in pmcids}
        
        # Track progress with tqdm
        for future in tqdm(as_completed(future_to_pmcid), total=len(pmcids), unit="articles"):
            try:
                data = future.result()
                result.extend(data)
            except Exception as e:
                pmcid = future_to_pmcid[future]
                print(f"Error processing PMCID {pmcid}: {e}")
    
    # Create DataFrame from the accumulated results
    return pd.DataFrame(result, columns=["PMCID", "Section", "Paragraph_text"])

# Function to process DataFrame with PMCIDs
def get_df(df, max_workers=4):
    """
    Processes a DataFrame of PMCIDs and extracts article sections, paragraphs, and captions concurrently.

    Args:
        df (pd.DataFrame): DataFrame containing a "PMCID" column.
        max_workers (int): Maximum number of threads to use for fetching articles.
    
    Returns:
        pd.DataFrame: A new DataFrame with PMCID, Section, and Paragraph_text columns.
    """
    pmcids = df["PMCID"].tolist()
    
    # Fetch and process articles concurrently
    return fetch_and_process_articles_concurrently(pmcids, max_workers)