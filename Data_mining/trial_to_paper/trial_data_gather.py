import pandas as pd
from trial_to_paper_utils import *
from typing import Optional, TypedDict
from variables import user, password

# SQL query pulls info per trial, search for given string(s) in indication names field 
trial_indication_query = """
SELECT DISTINCT ON (c.nct_id)
    s.nct_id,
    s.brief_title AS study_title,
	c.names AS indication,
    s.phase AS study_phase,
	s.overall_status as status,
	i.name as intervention,
	i.intervention_type,
	i.description as intervention_description,
	b.description
FROM 
    "all_conditions" as c
JOIN
	studies as s on s.nct_id = c.nct_id
JOIN
	brief_summaries as b on b.nct_id = c.nct_id
JOIN
	interventions as i on i.nct_id = c.nct_id
"""
ind_headers = [
    "nct_id", "study_title", "indication",
    "study_phase", "status", "intervention",
    "intervention_type", "intervention_description",
    "study_description"
    ]


# SQL query for given trial ID, return all adverse events per patient group
trial_ae_query = f"""
SELECT 
	c.names AS indication,
    s.nct_id,
    s.brief_title AS study_title,
    s.phase AS study_phase,
	s.overall_status as status,
	ae.adverse_event_term as adverse_event,
	ae.event_type,
	ae.organ_system,
	ae.subjects_at_risk,
	ae.subjects_affected as subjects_affected,
	ae.ctgov_group_code,
	ae.description,
	r.title as result_group_treatment,
	r.description as result_group_desc
FROM 
    "reported_events" as ae
JOIN
	all_conditions as c on c.nct_id = ae.nct_id
JOIN
    studies as s ON s.nct_id = ae.nct_id
JOIN 
	result_groups as r ON r.id = ae.result_group_id
"""
ae_headers = [
    "indication", "nct_id", "study_title",
    "study_phase", "status", "adverse_event",
    "event_type", "organ_system", "subjects_at_risk",
    "subjects_affected", "ctgov_group_code", "ae_description",
    "result_group_treatment", "result_group_description"]

class SearchDict(TypedDict):
    """ 
        Search strings followed by whether or 
        not the term is to be case sensitive
    """
    term: str
    case_sensitive: bool

search_dict = {
    "CKD": True,
    "kidney disease": False
}

def search_indication_builder(query:str, input_dict: SearchDict, limit: Optional[int] = 1000) -> str:
    """
    input_dict: terms to be searched for in postgresql query,
    followed by desire to be / not be case sensitive
    limit: search limit for query, if passed 0 then no limit is applied
    returns: generated search to feed into postgresql query
    """
    query_addition = ""
    for term in input_dict:
        if input_dict[term] == True:
            if query_addition == "":
                query_addition = query_addition + f"c.names LIKE '%{term}%'"
            else:
                query_addition = query_addition + f" AND c.names LIKE '%{term}%'"
        elif input_dict[term] == False:
            if query_addition == "":
                query_addition = query_addition + f"c.names ILIKE '%{term}%'"
            else:
                query_addition = query_addition + f" AND c.names ILIKE '%{term}%'"
    if limit != 0:
        query = f"{query}\nWHERE\n\t({query_addition})\nORDER BY\n\tc.nct_id\n\nLIMIT {str(limit)};"
    else:
        query = f"{query}\nWHERE\n\t({query_addition})\nORDER BY\n\tc.nct_id;"
    return query

def search_ae_query(query: str, nct_id: str) -> str:
    query = f"{query}\nWHERE\n\tae.nct_id = '{nct_id}';"
    return query

def main():
    ind_query = search_indication_builder(query=trial_indication_query, input_dict=search_dict)
    indication_df = aact_query(query=ind_query)
    print(indication_df.head(), "\n- - - - - - - ")
    ind_terms = "_".join(x for x in search_dict.keys()).replace(" ", "_")
    indication_df.to_csv(f"./output/{ind_terms}_sample.csv", header=ind_headers)

    # TODO - Replace this with for loop over indication_df, passing write to file where there are no AEs recorded in trial
    nctids_to_search = [
        "NCT00246129", "NCT00345839", "NCT00364845", "NCT00427037",
        "NCT00598442", "NCT00924781", "NCT01113801", "NCT01464190",
        ]

    for nct_id in nctids_to_search:
        ae_query = search_ae_query(query=trial_ae_query, nct_id=nct_id)
        print(ae_query)
        ae_df = aact_query(query=ae_query)
        print(ae_df.head(0))
        ae_df.to_csv(f"./output/AEs_{nct_id}.csv", header=ae_headers)
        # ae_df.to_csv(f"./output/AEs_{nct_id}.csv",)



if __name__ == "__main__":
    main()