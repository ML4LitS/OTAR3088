"""
Processing data from protein feature model, data sourced from figshare, under:
'Human-in-the-loop approach to identify functionally important residues of proteins from literature'.
Starting with v2.1 data due to authors concluding this data produced the best model
"""
import argparse
import csv
import pandas as pd
import polars as pl
from os import listdir
from typing import List, Optional

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, help="Path to dir. containing input files")
parser.add_argument("-o", "--output_dir", type=str, help="Path to output dir. to write to")
parser.add_argument("-c" "--column_name", type=str, help="Optional, specify which column to filter. 'column_2' set as default") # TODO - Set as optional arg
parser.add_argument("-k", "--keep_terms", type=str, help="Comma separated str of entity types to keep. 'O' is accounted for already") # TODO - sort for multiple entity types
parser.add_argument("-f", "--file_type", type=str, help="File type to read-in, assuming type 'tsv'")
args = vars(parser.parse_args())

file_type_map = {"tsv": "\t",
                 "csv": ","}


def group_segments(df: pl.dataframe) -> pl.dataframe:
    """
    Group segments of text, split by 'null' / new line character.
    Complies with downstream task requirements
    """

    # Create a group identifier: increment when word is null
    df = df.with_columns([
        pl.col('words').is_null().cum_sum().alias('group_id')
    ])
    
    # Filter out the null rows (they were just separators)
    df = df.filter(pl.col('words').is_not_null())
    
    # Group by group_id and aggregate words and labels into lists
    grouped_df = df.group_by('group_id').agg([
        pl.col('words').alias('words'),
        pl.col('labels').alias('labels')
    ]).sort('group_id')
    
    # Drop the group_id column (just used for grouping)
    grouped_df = grouped_df.select(['words', 'labels'])

    # To pandas for final formatting - drop rows with no mentions of keep_terms
    grouped_df = grouped_df.to_pandas()
    keep_terms = ['B-mutant', 'I-mutant']
    filtered_df = grouped_df[grouped_df['labels'].apply(lambda x: any(item in keep_terms for item in x))]

    return filtered_df

def data_filtering(input_dir: str,
                   output_dir: str,
                   keep_terms: List,
                   file_type: Optional[str] = "tsv"):
    """
    Function to filter IOB file, annotation entries which are not of interest.
    - Read-in all files in input_dir ending with file_type
    - Parse specified column to remove any terms not in keep_terms & "O"
    - Write to output_dir
    """
    in_files = [f for f in listdir(input_dir) if f.endswith(file_type)]
    print("Located the following files for processing:"+"\n"+str([x for x in in_files]),"\n")
    
    keep_terms = keep_terms.split(sep=",")
    allowed_types = ["O"]
    allowed_types += [f"B-{term}" for term in keep_terms]
    allowed_types += [f"I-{term}" for term in keep_terms]
    # Check allowed types are as expected
    print(f"Allowed types for filtering: {allowed_types}\n")

    try:
        separator = file_type_map[file_type]
    except KeyError as e:
        print(f"Unsupported separator {file_type}, revise")

    for i, full_path in enumerate([input_dir+f for f in in_files]):
        iob = pl.read_csv(
                            full_path,
                            separator=separator,
                            has_header=False
                        )
        iob.columns = ["words", "labels"]
        
        # Unique values before filtering
        # print(f"Unique tags before filtering: {iob['labels'].unique().to_list()}")

        changes_count = iob.filter(~pl.col('labels').is_in(allowed_types)).height # Items that will be changed
        iob = iob.with_columns(
            pl.when(pl.col("labels").is_in(allowed_types))
              .then(pl.col("labels"))
              .otherwise(pl.lit("O"))
              .alias("labels")
        )

        # Unique values after filtering
        # print(f"Unique tags after filtering: {iob['labels'].unique().to_list()}")

        # Format to list per segment of text, per row of df
        iob = group_segments(iob)

        # Write to parquet
        iob = iob[['words', 'labels']]
        iob.to_parquet(f"./output/{in_files[i][:-4]}.parquet")

        # Write to output directory
        iob.to_csv(
            f"./output/{in_files[i]}",
            sep=file_type_map[file_type],
            index=False,
            quoting=csv.QUOTE_NONE,
            escapechar='\\'
        )

        print(f"Processed file '{full_path}' with {changes_count} changes written to {output_dir}")
        print(iob)
        print("- - - - - ")


def main():
    data_filtering(args["input_dir"], args["output_dir"], args["keep_terms"], args["file_type"])

if __name__ == "__main__":
    main()