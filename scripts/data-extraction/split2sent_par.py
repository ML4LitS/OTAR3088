
"""
Script Name: split2sent_par.py
Description: This script contains functions used for splitting articles into sentences and paragraphs. It assumes articles are extracted in XML format
Usage: Works as a utility script imported into another.Functions can be imported independently, although, important to note that some are inter-dependent 
"""

__author__ = "EPMC team"
__email__ = "*****@ebi.ac.uk"
__version__ = "1.0.0"




from tqdm import tqdm
import pandas as pd
from typing import Tuple, Dict, Any, List
import requests, re, time

import spacy, scispacy


from bs4 import BeautifulSoup as bs
from multiprocessing import cpu_count, Pool


#using scibert spacy model for biomedical texts
 #model can be changed to different sizes.  Disabling other unnessary components to make pipeline run faster
nlp = spacy.load("en_core_sci_lg", disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"]) 
nlp.add_pipe("sentencizer")
nlp.max_length = 10_000_000 



#function to fetch articles xml 
def get_xml(pmcid: str) -> bs:
    """
    Extracts XML only content from EPMC using article PMCID
    
    Args:
    pmcid: Unique Article Identifier

    Returns:
    On a successful call to the API, returns a soup element, otherwise None

    """
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
            




def filter_tags(soup:bs) -> bs:
    """
    Cleans the XML content by removing specific tags and unwanted sections.

    This function removes tags such as mathematical formulae, supplementary materials, 
    acknowledgments, and disclaimers from the BeautifulSoup-parsed XML content. 
    It also removes sections with titles matching specified keywords.

    Args:
        soup (BeautifulSoup-bs object): Parsed XML content.

    Returns:
        soup: Cleaned XML with the specified tags and sections removed.
    """

    # Tags to ignore by name
    tags2ignore = ['inline-formula', 'supplementary-material', 'ack', 'contrib-group', 
                   "disclaimer","Disclosure",
                   'sup', 'Acknowledgments','COI-statement']
    
    # Keywords indicating sections to be removed
    section_keywords = ['Disclaimer', 'author contributions', 
                       'Conflict of interest', "Publisher’s note", 
                       'Supplementary information', "Supplementary material", "Disclosure", 
                       ]

    # Remove tags by name
    for tag_name in tags2ignore:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove sections based on specific keywords in titles
    for section in soup.find_all('sec'):
        title_tag = section.find('title')
        if title_tag and any(keyword.lower() in title_tag.text.lower() for keyword in section_keywords):
            section.decompose()

    return soup



#function to fetch paragraph text from xml--> used for sentenciser
def get_full_text(soup:bs) -> Tuple[str,str]:
    """
    Extracts article body and abstract text 
    This function extracts an article title, abstract, body text. 
    The abstract and body text are later concatenated together

    Args: 
    soup(bs): a BeautifulSoup object containing the XML content of the article

    Returns: 
    title(str): Article title 
    full_text(str): Article body(abstract + body)

    """
    
    if soup:
        try:
            #filter unwanted tags
            soup = filter_tags(soup)
            #extract article title
            title = soup.find("article-title").text if soup.find("article-title") else "Article is missing a title" #encountered some of these earlier, specifically with books. so catching them here
            #clean title for extra whitespace using clean text fun
            title = clean_text(title)
            
            #extract abstract 
            abstract = soup.find("abstract")
            abstract_text = (" ".join([clean_text(p.text) for p in abstract.find_all("p") if p.text]) if abstract else "")
            
            #extract body of article
            
            body_tag = soup.find("body")
            body_text = (" ".join([clean_text(p.text) for p in body_tag.find_all("p")
                                            if p.text]) if body_tag else "")
            
            full_text = f"{abstract_text}{body_text}".strip()
            
            return title, full_text
        
        
        except Exception as e:
            print(f"Error processing XML: {e}")
            return "Error extracting title"
        
        
        
    else:
        return "No XML provided", ""
        
            
                       

def clean_text(text:str) -> str:
    """
    This function cleans a text by filtering reference patterns in text, 
    extra whitespaces, escaped latex-style formatting appearing in text body instead of predefined latex tags

    Args: 
    text(str): The text to be cleaned
    
    Returns: 
    tex(str): The cleaned text 
    
    """
   
    # Remove LaTeX-style math and formatting tags #already filtered from soup content but some still appear
    text = re.sub(r"\{.*?\}", "", text)  # Matches and removes anything inside curly braces {}
    text = re.sub(r"\\[a-zA-Z]+", "", text)  # Matches and removes characters that appears with numbers
    
    # Remove reference tags like [34] or [1,2,3]
    text = re.sub(r"\[\s*(\d+\s*(,\s*\d+\s*)*)\]", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def sentencise_articles(row: Dict[str, Any], nlp: spacy.language.Language = nlp) -> List[Dict[str, str]]:
    """
    This function processes a row from a dataframe containing a PMCID and extracts sentence-level information from the article's text.
    
    Args:
        row (Dict[str, Any]): A dataframe series with "PMCID" as key for the article.
        nlp (spacy.language.Language): A Sci-SpaCy NLP model used for sentence segmentation.
        
    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing:
            - "PMCID": The PMCID identifier for an article.
            - "Title": The title of the article.
            - "Sentences": Segmented sentences from the article text.
    """

    articles = []
    pmcid = row["PMCID"]
    xml = get_xml(pmcid)
    title, full_text = get_full_text(xml)
    if full_text:
        doc = nlp(full_text)
        sentences = [sent.text for sent in doc.sents]
        for sent in sentences:
            articles.append({"PMCID": pmcid, "Title": title, "Sentences": sent})
    return articles


def postprocess_sentences(sentence:List) -> List:
    """
    Cleans a list of sentences after the sentence segmentation by removing unwanted characters like "a)", ")", or "(" 
    from the beginning or end of each sentence. Only used for sentence split articles

    Args: 
    Sentences: A list of sentences

    Returns: 
    Cleaned_Sentences: List of cleaned sentences
    """
    cleaned = re.sub(r"^[()\da-zA-Z]+\)|^\)|^\(|\)[()\da-zA-Z]+$", "", sentence.strip())
    return cleaned





def process_in_paragraph(row):
    """
    Processes an article's sections, paragraphs, and figure captions.

    Args:
        row: A pandas series.

    Returns:
        list: A list of tuples containing PMCID, Title, Section-title, and paragraph text.
    """
    pmcid = row["PMCID"]
    soup = get_xml(pmcid)

    if not soup:
        return []  # Return empty list if no XML content

    try:
        soup = filter_tags(soup)
        title = soup.find("article-title").text if soup.find("article-title") else "Article is missing a title" #encountered some of these earlier, specifically with books. so catching them here
        title = clean_text(title)
        rows = []

        # Process abstract if available
        abstract = soup.find("abstract")
        if abstract:
            abstract_paragraph = abstract.find("p")
            if abstract_paragraph:
                abstract_text = clean_text(abstract_paragraph.get_text(separator=" ").strip())
                rows.append((pmcid, title, "Abstract", abstract_text))

        # Process sections and paragraphs
        for section in soup.find_all("sec", recursive=True):
            section_title = section.find("title").get_text(strip=True) if section.find("title") else "Unnamed section"
            
            # Extract paragraphs
            for paragraph in section.find_all("p", recursive=True):
                paragraph_text = clean_text(paragraph.get_text(separator=" ").strip())
                if paragraph_text:
                    rows.append((pmcid, title, section_title, paragraph_text))
            
            # Handle figure captions
            for figure in section.find_all("fig", recursive=True):
                caption = figure.find("caption")
                if caption:
                    caption_text = clean_text(caption.get_text(separator=" ").strip())
                    figure_title = f"{section_title} - Figure Caption"
                    rows.append((pmcid, title, figure_title, caption_text))
                

        return rows

    except Exception as e:
        print(f"Error processing XML for PMCID {pmcid}: {e}")
        return []
        

def parallel_process_articles(df: pd.DataFrame, func: callable, process_par:bool=False) -> pd.DataFrame:
    """
    Generic parallel processing function to speed up all other functions. Works for both splitting in sentences and paragraph.
    

    Args:
        df (pd.DataFrame): The input DataFrame.
        func (callable): The function to apply multiprocessing to: sentencise_articles or process_in_paragraph
        processing_type(str): choices ['sent', 'par']; where sent=sentenciser, and 'par'=split paragraph

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    
        
    data = [row for _, row in df.iterrows()]
    

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(func, data), total=len(data), desc="Extracting and processing Articles"))
        #print(f"While results looks like this: {results}")
    # Flatten lists and filter out malformed results if applicable
    flattened_results = [item for sublist in results if isinstance(sublist, list) for item in sublist]
    if process_par:
        columns = ["PMCID", "Title", "Section", "Paragraph_text"]
        articles = pd.DataFrame(flattened_results, columns=columns)
        return articles
    
    return pd.DataFrame(flattened_results)