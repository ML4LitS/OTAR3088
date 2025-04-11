from chembl_sql_utils import sqlite_query

"""
SQL queries collected around extracting info from ChEMBL assay descriptions
"""

# Version of SQLite ChEMBL database queried
db_version = 'chembl_34.db'

path_to_cell_line_dict = './chembl_cell_dictionary.csv' # Sourced from ChEMBL

""" INPUT DATA - EDIT / COMMENT AS REQUIRED """
"""
    (1.) Assay descriptions, relevant to cell lines model works
    - Select from assays table
    - Not null, description contains 'cell', human only, mapped to a cell type
"""
# query = """
#             SELECT assay_id, assay_organism, assay_tissue, assay_strain, assay_cell_type, cell_id, doc_id, description AS d
#             FROM assays
#             WHERE  d IS NOT NULL AND  d LIKE '%cell%' AND assay_organism LIKE '%Homo sapiens%' AND  assay_cell_type NOT NULL;
#         """

# outfile_all = 'chembl_assay_desc_cell.csv'

# outfile_cleaned = './cell_line_sentences.csv'
# headers = ['assay_id', 'assay_organism', 'assay_tissue', 'assay_strain',
#            'assay_cell_type', 'cell_id', 'doc_id', 'description']

"""
    (2.) Info from ChEMBL to pull cell dictionary data
"""
query = """
          SELECT * FROM cell_dictionary;
        """
outfile = 'cell_dictionary.csv'
headers = ['cell_id', 'cell_name', 'cell_description', 'cell_source_tissue',
           'cell_source_organism', 'cell_source_tax_id', 'clo_id', 'efo_id',
           'cellosaurus_id', 'cl_lincs_id', 'chembl_id', 'cell_ontology_id']

if __name__ == "__main__":
    """
    db_version - specify which ChEMBL version to query
    query - SQL query to run
    outfile - path to output file, if required
    save_cleaned - save output of SQL query as 1 row per entity, no duplicates
    headers - Headers for save file
    """

    res_df = sqlite_query(db_version=db_version, query=query, path_to_dictionary='',
                          outfile=outfile, save_cleaned=False, headers=headers)
