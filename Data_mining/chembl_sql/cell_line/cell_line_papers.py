from chembl_sql_utils import get_range_col_and_papers, sqlite_query

"""
Investigating the use of cell lines referenced most frequently in the ChEMBL assay descriptions
for the selection of suitable literature for model training
"""

# Local version of SQLite ChEMBL database queried
db_path = '/Users/withers/Documents/DataDownloads/ChEMBL/chembl_34_sqlite/chembl_34.db'

"""
    Unique sets of cell line <--> paper, linked to cell line via docs table
    - Select from assays table + docs table
    - Not null for pmid, title, assay_cell_type, human only, paper published 2020-23 (in-keeping w. ChEMBL release)
"""

query = """
            SELECT 
                A.assay_organism, 
                A.assay_cell_type, 
                A.doc_id AS assay_doc_id, 
                D.doc_id AS doc_id, 
                D.pubmed_id AS pmid, 
                D.title,
                D.year,
                D.doc_type
            FROM 
                assays AS A
            RIGHT JOIN 
                docs AS D 
                ON A.doc_id = D.doc_id
            WHERE 
                A.assay_cell_type IS NOT NULL
                AND A.assay_organism LIKE '%Homo sapiens%'
                AND pmid IS NOT NULL
                AND D.title IS NOT NULL
                AND D.year > 2019
            GROUP BY 
                D.pubmed_id, D.title, A.assay_organism, A.assay_cell_type, A.doc_id;
        """
outfile_cleaned = './cell_line_papers.csv'
headers = ['assay_organism', 'assay_cell_type', 'assay_doc_id', 'doc_id', 'pmid', 'title', 'year', 'doc_type']

if __name__ == "__main__":
    """
    (1.) query - SQL query to run
         outfile - path to output file, if required
         headers - Headers for save file
    (2.) res_df - results of SQL query to ChEMBL
         col_name - column to collect frequencies and subsequent chosen range from
         Aim here to identify x suitable source papers to use as input for model training
    """
    res_df = sqlite_query(db_version=db_path, query=query, outfile=outfile_cleaned,
                          save_cleaned=False, path_to_dictionary='', headers=headers)

    frequency_count = res_df['assay_cell_type'].value_counts().reset_index()
    frequency_count.columns = ['assay_cell_type', 'frequency']
    frequency_count.to_csv('./assay_cell_type_freq.csv')

    # for given range of cell lines in results sorted by frequency, grab papers referencing them
    # Bottom x rows
    bottom_df, bottom_pmids = get_range_col_and_papers(res_df=res_df, col_name='assay_cell_type', sort_by='year', range=(-100, 'end'))
    bottom_df.to_csv("./bottom.csv")
    # Top x rows
    top_df, top_pmids = get_range_col_and_papers(res_df=res_df, col_name='assay_cell_type', sort_by='year', range=(0, 100))
    top_df.to_csv("./top.csv")
    top_pmids.extend(bottom_pmids)

    # Clean list of PMIDs to convert to PMCIDs for full text annotations
    searchable_pmids = list(set(top_pmids))
    with open("searchable_pmids", "w+") as f:
        [f.write(str(x)+"\n") for x in searchable_pmids]
    f.close()
