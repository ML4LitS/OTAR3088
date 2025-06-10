import pandas as pd
import subprocess
from typing import Optional

in_cell_dict = pd.read_csv(
    filepath_or_buffer="/Users/withers/Documents/DataDownloads/ChEMBL/chembl_34_sqlite/chembl_cell_dictionary.csv")
in_tissue_dict = pd.read_csv(
    filepath_or_buffer="/Users/withers/Documents/DataDownloads/ChEMBL/chembl_34_sqlite/chembl_tissue_dict.csv", sep="\t")

def process_dict(
        input_df: pd.DataFrame, outfile_path: str,
        terms: str, uids: Optional[str] = None):
    term_col = input_df[terms]
    if uids:
        uid_col = input_df[uids]
        with open(outfile_path, 'w+') as outfile:
            for term, uid in zip(term_col, uid_col):
                outfile.write(term+"\t"+uid+"\n")
    else:
        with open(outfile_path, 'w+') as outfile:
            [outfile.write(term+"\n") for term in term_col]
    outfile.close()

process_dict(
    input_df=in_cell_dict, outfile_path="./output/leadmine/cell_dictionary.dict",
    terms = "cell_name", uids = "chembl_id"
             )

process_dict(
    input_df=in_tissue_dict, outfile_path="./output/leadmine/tissue_dictionary.dict",
    terms = "pref_name", uids = "chembl_id"
             )

# java -jar /Users/withers/Documents/leadmine-3.19/bin/compilecfx.jar ./output/leadmine/cell_dictionary.dict -o ./output/leadmine/cell_dictionary.cfx
# java -jar /Users/withers/Documents/leadmine-3.19/bin/compilecfx.jar -i ./output/leadmine/tissue_dictionary.dict -o ./output/leadmine/tissue_dictionary.cfx
cell_exec = ["java", "-jar", "/Users/withers/Documents/leadmine-3.19/bin/compilecfx.jar", "./output/leadmine/cell_dictionary.dict", "-o cell_dictionary.cfx"]
tissue_exec = ["java", "-jar", "/Users/withers/Documents/leadmine-3.19/bin/compilecfx.jar", "-i", "./output/leadmine/tissue_dictionary.dict", "-o tissue_dictionary.cfx"]
result = subprocess.run(cell_exec, shell=True, capture_output=True, text=True)
print(result.stdout)

result = subprocess.run(tissue_exec, shell=True, capture_output=True, text=True)
print(result.stdout)