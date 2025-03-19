import pandas as pd
import sqlite3
from typing import List, Optional
import csv

from IPython.core.display_functions import display

"""
Collection of util functions relating to querying ChEMBL SQLite database
"""

def write_csv(outfile: str, headers: List, rows: List):
    """ Write output CSV, including provided headers """
    with open(outfile, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(headers)
        # Write data rows
        csvwriter.writerows(rows)
        print(f'Query completed, output exported to: {outfile}')


def map_to_dictionary(input_df: pd.DataFrame, path_to_dictionary: str) -> pd.DataFrame:
    """
    Map cell name from input_df to row in dictionary
    Retrieve cell description to enrich data / aid interpretation
    """
    dictionary = pd.read_csv(path_to_dictionary)
    descriptions = []
    for i, x in input_df.iterrows():
        match = dictionary[dictionary['cell_name'] == i]
        if len(match) > 0:
            descriptions.append(match['cell_description'].iloc[0])
        else:
            descriptions.append('')
    input_df['entity_definition'] = descriptions
    output_df = input_df.iloc[:, [1, 0]]
    return output_df


def clean_output_for_model(res_df: pd.DataFrame, count_length: int, path_to_dictionary: str, output: str):
    # TODO- This function is still specific to the cell line example and should be generalised
    """
    Filter for 1 given index / entity for results table, length limited by count_length
    Aim to provide one example sentence / description for each mentioned entity type
    """
    captured_res = {}
    counter = 1

    for i, x in res_df.iterrows():
        cell_line = x['assay_cell_type']
        if cell_line not in captured_res.keys() and 'panel' not in cell_line.lower() and counter <= count_length:
            # Cell line not yet encountered, not a panel entry (screens over multiple cell lines)
            captured_res[x['assay_cell_type']] = x['description']
            counter += 1
    processed_df = pd.DataFrame.from_dict(captured_res,
                                          orient='index',
                                          columns=['description'])
    processed_df_mapped = map_to_dictionary(processed_df, path_to_dictionary=path_to_dictionary)
    processed_df_mapped.to_csv(output)
    print(f'Query completed, cleaned output exported to: {output}')


def get_top_ten_and_papers(res_df: pd.DataFrame, col_name: str, sort_by: str):
    """
    Grab subsets of results df for each of the top 10 most-common mentions in a specified column
    E.g. for the top 10 cell lines occuring in the results df, list each subset df
    """
    column_frequencies = res_df[col_name].value_counts()
    top_ten = column_frequencies[:10]
    for index_label in top_ten.index:
        df_subset = res_df[res_df[col_name] == index_label]
        df_subset = df_subset.sort_values(by=[sort_by], axis=0, ascending=False)
        with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
            display(df_subset)


def sqlite_query(db_version: str, query: str, outfile: Optional[str], save_cleaned: Optional[bool],
                 path_to_dictionary: Optional[str], headers: List) -> pd.DataFrame:
    """ Query to database, writes output to CSV """
    connection = sqlite3.connect(db_version)
    try:
        cursor = connection.cursor()
        # Execute the query
        cursor.execute(query)
        returned_rows = cursor.fetchall()  # Result format = [Tuple, ... ,Tuple]
        res_df = pd.DataFrame(returned_rows)
        res_df.columns = headers

        if save_cleaned:
            # One row per entity
            clean_output_for_model(res_df=res_df, count_length=50,
                                   path_to_dictionary=path_to_dictionary, output=outfile)

        else:
            # pprint(res_df)
            if outfile:
                # Write to CSV, not processed output
                write_csv(outfile=outfile, headers=headers, rows=returned_rows)

    finally:
        connection.close()

    return res_df
