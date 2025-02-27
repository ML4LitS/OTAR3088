import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from trial_to_paper_utils import *
from variables import nct_id
from pprint import pprint

if __name__ == '__main__':
    # TODO - User input NCTID / move to streamlit service

    # 1. AACT queried with starting NCT ID of interest to collect referenced adverse events
    study_title, aes, severe_aes, other_aes, patient_groups = aact_data_gather(nct_id)
    # TODO Get % affected / see relevance of 'other' vs 'serious'
    if study_title:
        print(
            f"For CT.gov trial '{study_title}' ({nct_id}), {len(aes)} unique AEs were recorded in {len(patient_groups)} patient group(s)")
        pprint(aes)
    else:
        print(f"No trials were found searching for ID: {nct_id}")

    # 2. Search ePMC for papers mentioning trial ID, failing this search for papers relating to compound name
    trial_in_pmids = query_epmc(query=nct_id, page_size=25)
    print(f"NCT ID referenced in {len(trial_in_pmids)} PubMed paper(s).\n{trial_in_pmids}")
    if not trial_in_pmids:
        print('NCT ID not referenced in papers, searching for compound name instead')
        trial_in_pmids = query_epmc(query="VX-548", page_size=25) #TODO - Drug name as alternative to NCT ID, automate this

    test_pmid = trial_in_pmids[0] #TODO - remove test, do for all results & consider subject test groups
    text = query_bioc(pmid=test_pmid)
    print_text = False
    if print_text:
        print("\n".join(text))
        print(len(text))
        print(text[3])

    # TODO Test open source models for AE detection in text, compare to those recorded in trial
    # pipe = pipeline(task="token-classification", model="MutazYoune/BiomedBERT-Adverse-Events-NER_pun", tokenizer="MutazYoune/BiomedBERT-Adverse-Events-NER_pun")
    pipe = pipeline(task="token-classification", model="MutazYoune/Medical-NER-Adverse-Events-NER", tokenizer="MutazYoune/Medical-NER-Adverse-Events-NER")
    print('- - - - - - - - - -')
    all_sections = []
    for x in tqdm(text):
        try:
            # print(x)
            res = pipe(x)
            if res:
                section_df = pd.DataFrame(res)
                # pprint(section_df)
                all_sections.append(section_df)
            # print('----------------------------------')
        except:
            continue
    paper_aes = pd.concat(all_sections, ignore_index=True)
    model_name = pipe.model.name_or_path
    model_name = model_name.rpartition('/')[-1]
    pprint(paper_aes)
    paper_aes.to_csv(f'./output/{model_name}_AEs_{test_pmid}.csv')