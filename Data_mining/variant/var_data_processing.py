"""
Processing data from protein feature model, data sourced from figshare, under:
'Human-in-the-loop approach to identify functionally important residues of proteins from literature'.
Starting with v2.1 data due to authors concluding this data produced the best model
"""
import argparse
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
        
        # Unique values before filtering
        # print(f"Unique tags before filtering: {iob['column_2'].unique().to_list()}")

        changes_count = iob.filter(~pl.col('column_2').is_in(allowed_types)).height # Items that will be changed
        iob = iob.with_columns(
            pl.when(pl.col("column_2").is_in(allowed_types))
              .then(pl.col("column_2"))
              .otherwise(pl.lit("O"))
              .alias("column_2")
        )

        # Unique values after filtering
        # print(f"Unique tags after filtering: {iob['column_2'].unique().to_list()}")

        # Write to output directory
        iob.write_csv(
            f"./output/{in_files[i]}",
            separator=file_type_map[file_type],
            include_header=False
        )

        print(f"Processed file '{full_path}' with {changes_count} changes written to {output_dir}")
        print(iob)
        print("- - - - - ")


def main():
    data_filtering(args["input_dir"], args["output_dir"], args["keep_terms"], args["file_type"])

if __name__ == "__main__":
    main()