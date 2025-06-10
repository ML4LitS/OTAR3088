import pandas as pd

cell = "/Users/withers/GitProjects/OTAR3088/DataFolder/Data-Extraction-Query/output/leadmine/cell_dictionary.dict"
tissue = "/Users/withers/GitProjects/OTAR3088/DataFolder/Data-Extraction-Query/output/leadmine/tissue_dictionary.dict"
brendacell = "/Users/withers/GitProjects/OTAR3088/DataFolder/Data-Extraction-Query/output/brenda/brenda_cells.txt"
brendatissue = "/Users/withers/GitProjects/OTAR3088/DataFolder/Data-Extraction-Query/output/brenda/brenda_tissues.txt"


def clean_up_dictionary(path_to_infile: str, outfile: str, flag: str, entity_name: str):
    """
    Clean file to comply with pubtator input format
    path_to_infile: Path to generated dictionary-like file
    outfile: path to write tsv to
    """
    res_list = []
    with open(path_to_infile) as in_file:
        count = 0
        for l in in_file.readlines():
            elements = l.split("\t")
            if flag == "CHEMBL":
                res_list.append([entity_name, elements[1].strip(), elements[0].strip()])
            elif flag == "BRENDA":
                res_list.append([entity_name, count, elements[0].strip()])
                count += 1
            else:
                print("Error with flag - check spelling")
        in_file.close()

    res_df = pd.DataFrame(res_list)
    res_df.to_csv(outfile, sep="\t", index=False, header=False)

# clean_up_dictionary(cell, "./cell_df.tsv", "CHEMBL", "CELL")
# clean_up_dictionary(tissue, "./tissue_df.tsv", "CHEMBL", "TISSUE")
# clean_up_dictionary(brendacell, "./brendacell_df.tsv", "BRENDA", "CELL")
# clean_up_dictionary(brendatissue, "./brendatissue_df.tsv", "BRENDA", "TISSUE")