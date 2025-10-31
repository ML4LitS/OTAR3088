import requests
from typing import Dict, List
import re
from bs4 import BeautifulSoup as bs
from labelstudio_e2e import write_ls_textfile, ls_formatter

# Re-running with original cellate papers for annotation V3
paper_uids = {
        "PMC12081310": "PMCID",
        "PMC11578878": "PMCID",
        "PMC10287567": "PMCID",
        "PMC8809252": "PMCID",
        "PMC9271637": "PMCID",
        "PMC12479362": "PMCID",
        "PMC12435838": "PMCID",
        "PMC12408821": "PMCID",
        "PMC12396968": "PMCID",
        "PMC12256823": "PMCID",
        "PMC12116388": "PMCID",
        "PMC12133578": "PMCID",
        "33981032": "PMID",
        "40986340": "PMID",
        "40712580": "PMID"
}
<<<<<<< HEAD
=======
paper_uids = {
    "PMC8494645": "PMCID",
    "PMC8195859": "PMCID" #not available
} # My papers
>>>>>>> ac2051c5 (variant data processing)

master_path = './output/labelstudio/master_dictionary.tsv' # Concatenated dictionary for pre-annotation


def peek_at_xml(soup: bs):
    # Used for quick check on xml obj that have been written to file
    content = soup.prettify()
    print(content)


def convert_pmid_pmcid(pmid: str):
    """
    Query attempts so convert PMID to available PMCID, None if no conversion is possible
    """
    base_url = 'https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/'
    
    params = {
        'ids': pmid,
        'format': 'json',
        'idtype': 'pmid',
        'tool': 'python_sript',
        'email': 'withers@ebi.ac.uk'
    }
    
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    if data.get('status') == 'ok' and data.get('records'):
        record = data['records'][0]
        pmcid = record.get('pmcid')
        if pmcid != None:
            print(f"PMID {pmid} mapped to PMCID {pmcid}")
            return pmcid
        else:
            print(f"Unable to find PMCID for PMID {pmid}")
            return pmcid
    return None


def get_epmc_full_text(uid: str, map: Dict) -> str:
    """
    E.g. xml_out = get_epmc_fulltext_xml('PMC2231364')
    """
    
    if map[uid] == "PMID":
        pmcid = convert_pmid_pmcid(uid)
        if pmcid == None:
            pass
            # Passing NoneType uid
    else:
        # Check PMCID is complete
        if not uid.startswith('PMC'):
            pmcid = f'PMC{uid}'
        else:
            pmcid = uid
    
    if pmcid != None and pmcid.startswith("PMC"): #Good to go
        # Build query
        base_url = 'https://www.ebi.ac.uk/europepmc/webservices/rest'
        url = f'{base_url}/{pmcid}/fullTextXML'    

        response = requests.get(url, params={})
        
        # Check if request was successful
        if response.status_code == 200:
            return response.text
        elif response.status_code == 404:
            print(f'Article {pmcid} not found or full-text XML not available')
        else:
            response.raise_for_status()


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


def process_sections(soup: bs, uid: str) -> List:
    """
    .
    """
    processed_list = []

    try:
        soup = filter_tags(soup)
        title = soup.find("article-title").text if soup.find("article-title") else "Article is missing a title" #encountered some of these earlier, specifically with books. so catching them here
        title = clean_text(title)

        # Process abstract if available
        abstract = soup.find("abstract")
        if abstract:
            abstract_paragraph = abstract.find("p")
            if abstract_paragraph:
                abstract_text = clean_text(abstract_paragraph.get_text(separator=" ").strip())
                processed_list.append((uid, title, "Abstract", abstract_text))

        # Process sections and paragraphs
        for section in soup.find_all("sec", recursive=True):
            section_title = section.find("title").get_text(strip=True) if section.find("title") else "Unnamed section"
            
            # Extract paragraphs
            for paragraph in section.find_all("p", recursive=True):
                paragraph_text = clean_text(paragraph.get_text(separator=" ").strip())
                if paragraph_text:
                    processed_list.append((uid, title, section_title, paragraph_text))
            
            # Handle figure captions
            for figure in section.find_all("fig", recursive=True):
                caption = figure.find("caption")
                if caption:
                    caption_text = clean_text(caption.get_text(separator=" ").strip())
                    figure_title = f"{section_title} - Figure Caption"
                    processed_list.append((uid, title, figure_title, caption_text))
                

        return processed_list

    except Exception as e:
        print(f"Error processing XML for UID {uid}: {e}")
        return []

def paper_list_to_str(text_list: List) -> str:

    # Start out by adding the PMCID, title and first section header
    text = text_list[0][0]+"\n"
    text += text_list[0][1]+"\n"
    current_header = text_list[0][2]
    text += current_header+"\n"
    for tup in text_list:
        # tuple format - (PMCID, article title, section header, text)
        if tup[2] != current_header:
            text += tup[2]+"\n"
            current_header = tup[2]
        text += tup[3]+"\n"
    return text


def main():
    for uid in paper_uids.keys():
        fulltext_xml = get_epmc_full_text(uid, paper_uids)
        if fulltext_xml != None:
            soup = bs(fulltext_xml, 'lxml')
            fulltext_clean = filter_tags(soup)
            fulltext_clean_list = process_sections(fulltext_clean, uid)
            # peek_at_xml(fulltext_clean)
            fulltext = paper_list_to_str(fulltext_clean_list)
            with open(f"./extracted_texts/{uid}.txt", "a") as out_file:
                out_file.writelines(fulltext)
                out_file.close()
            
            # Format for LabelStudio
<<<<<<< HEAD
            annotated_path = f'./output/labelstudio/V4/{uid}_annotation.txt'
            ls_json_path = f'./output/labelstudio/V4/{uid}_annotation.json'
=======
            annotated_path = f'./output/labelstudio/sc/{uid}_annotation.txt'
            ls_json_path = f'./output/labelstudio/sc/{uid}_annotation.json'
>>>>>>> ac2051c5 (variant data processing)
            write_ls_textfile(input_text=fulltext, path_to_outfile=annotated_path)
            ls_formatter(
                dict_file=master_path,
                texts_file=annotated_path,
                output_json=ls_json_path,
                pmcid=uid)


if __name__ == "__main__":
    main()