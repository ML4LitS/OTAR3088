from chembl_sql_utils import get_top_ten_and_papers as gtt, sqlite_query

"""
Investigating the use of cell lines referenced most frequently in the ChEMBL assay descriptions
for the selection of suitable literature for model training
"""

# Version of SQLite ChEMBL database queried
db_path = 'chembl_34.db'

"""
    Unique sets of cell line <--> paper, linked to cell line via docs table
    - Select from assays table + docs table
    - Not null for pmid, title, assay_cell_type, human only, paper published post 2020
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
                AND D.year > 2020
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
         col_name - column to collect frequencies and subsequent top-10 of
         Aim here to identify 10 suitable source papers to use as input for model training
    """
    res_df = sqlite_query(db_version = db_path, query = query, outfile = outfile_cleaned,
                          save_cleaned = False, path_to_dictionary = '', headers = headers)

    # for top-10 most frequently occurring cell lines in results, grab papers referencing them
    gtt(res_df = res_df, col_name ='assay_cell_type', sort_by='year')
