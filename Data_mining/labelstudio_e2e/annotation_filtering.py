import glob
import numpy as np
import pandas as pd
import re
from pprint import pprint
from typing import Dict, List, Tuple
from labelstudio_e2e import collate_dictionaries, ls_formatter

# Papers from cellfinder data to investigate
# TODO - This should all be argparse
root_to_anns = "/Users/withers/Downloads/cellfinder1_brat/"
ann_path = root_to_anns + "*.ann"
pmcids = glob.glob(ann_path)
text_path = root_to_anns + "*.txt"
texts = glob.glob(text_path)
annotation_data = list(zip(pmcids, texts))

# Paths to dictionary/dictionaries of interest
tissue_df = "./tissue_df.tsv"
btissue_df = "./brendatissue_df.tsv"
# Entity type of interest
entity_review = "Anatomy"


def readin_ann(path_to_ann: str) -> pd.DataFrame:
    """
    :param path_to_ann: path of .ann file
    :return: result df
    """
    input_df = pd.read_csv(path_to_ann, sep="\t", header=None)
    ann_file_df = pd.DataFrame(np.column_stack([x.split() for x in input_df[1]])).transpose()
    input_df = input_df.drop(columns=[0, 1])

    ann_file_df.columns = ["label", "start", "end"]
    ann_file_df["term"] = input_df
    return ann_file_df


def readin_txt(path_to_txt: str) -> List:
    with open(path_to_txt, "r") as f:
        input_text = f.readlines()
    f.close()
    ## Optional - print lines in input_text
    # [print(x, "\n") for x in input_text]
    return input_text


def filter_df(annotation_df: pd.DataFrame, label: str) -> List:
    filtered_df = annotation_df[annotation_df["label"] == label]
    terms = filtered_df["term"].to_list()
    terms = [x.lower() for x in terms]
    filtered_list = list(set(terms))
    return filtered_list


def combine_dicts(dicts_paths: List) -> List:
    if len(dicts_paths) > 1:
        res_list = pd.read_csv(dicts_paths[0], sep="\t", header=None)[2].to_list()
        for input_path in dicts_paths[1:]:
            extend_list = pd.read_csv(input_path, sep="\t", header=None)[2].to_list()
            res_list.extend(extend_list)
    final_list = [x.lower() for x in res_list]
    final_list = list(set(final_list))
    return final_list


def annotation_review(task: Tuple) -> Tuple[List, List, List]:
    cellfinder_df = readin_ann(path_to_ann=task[0])
    # Cleaned, unique terms annotated in text
    filtered = filter_df(annotation_df=cellfinder_df, label=entity_review)
    # Get overlap of .ann files and dictionary annotation terms
    overlap = [term for term in filtered if term in master_dict]
    # Check for annotations out of dictionary
    cellfinder_only = [term for term in filtered if term not in master_dict]
    return filtered, overlap, cellfinder_only


def printout(pmcid: str, overlap: List, filtered_terms: List, cellfinder_only: List):
    print(
        f"\n- - - - - - - - - Reviewing entities of type '{entity_review}' in file {pmcid}.ann - - - - - - - - -\n"
    )

    if len(overlap) == 0:
        print("*** No overlap between annotations and dictionaries. Annotations are as shown: ***")
        print("-", "\n- ".join(cellfinder_only))
        drop = 0
    else:
        drop = 100 - (len(overlap) / len(filtered_terms) * 100)
        print("Terms tagged in annotations overlapping with dictionary:")
        print("-", "\n- ".join(overlap))
        print("\nTerms annotated, but not in dictionary:")
        print("-", "\n- ".join(cellfinder_only))
    return drop


def write_report(out_path_report: str, out_path_txts: str, report_collection: Dict, average_dropout: float):
    with open(out_path_report, "w+") as f:
        f.write("Report on comparison of cellfinder annotations vs. dictionary terms")
        for pmcid, results in report_collection.items():
            f.write(
                f"\n\n- - - - - - - Reviewing entities of type '{entity_review}' in file {pmcid}.ann - - - - - - -\n"
            )
            if len(results["overlap"]) == 0:
                f.write("*** No overlap between annotations and dictionaries. Annotations are as shown: ***\n- ")
                f.write("\n- ".join(results["cellfinder_only"]))
            else:
                f.write("Terms tagged in annotations overlapping with dictionary:\n- ")
                f.write("\n- ".join(results["overlap"]))
                f.write("\n\nTerms annotated, but not in dictionary:\n- ")
                f.write("\n- ".join(results["cellfinder_only"]))
        f.write(
            f"\nAverage drop in coverage when requiring dictionary overlap of annotations = {'%.2f' % average_dropout}"
        )
        f.close()

    with open(out_path_txts, "w+") as f:
        i = 0
        for pmcid, results in report_collection.items():
            text = ' '.join(results['cellfinder_txt'])
            text = text.split(sep=". ")
            if i == 0:
                i += 1
            else:
                f.write(
                    f"* * * * * * * * * * * * * * * * * * * * * NEXT FILE * * * * * * * * * * * * * * * * * * \n"
                )
            f.write(f"                                       PMCID {pmcid}                                      \n\n")
            f.write('.\n'.join(text) + '\n')
        f.close()


if __name__ == "__main__":
    # Compile dictionaries of interest for comparison to .ann
    master_dict = combine_dicts(dicts_paths=[tissue_df, btissue_df])
    # Iterate over papers for annotation generation & data fallout assessment when filtering terms
    dropout = []
    report_collection = {}
    for annotation_task in annotation_data:
        pmcid = re.findall(r'[^\/]+(?=\.)', annotation_task[0])[0]
        filtered_terms, overlap, cellfinder_only = annotation_review(annotation_task)
        # Read in corresponding text
        cellfinder_txt = readin_txt(path_to_txt=annotation_task[1])

        percent_drop = printout(pmcid=pmcid,
                                overlap=overlap,
                                filtered_terms=filtered_terms,
                                cellfinder_only=cellfinder_only,)
        report_collection[pmcid] = {
            "overlap": overlap,
            "filtered_terms": filtered_terms,
            "cellfinder_only": cellfinder_only,
            "cellfinder_txt": cellfinder_txt
        }
        dropout.append(percent_drop)

    average_dropout = sum(dropout) / len(dropout)
    write_report(out_path_report="./annotation_review.txt",
                 out_path_txts="./annotation_review_texts.txt",
                 report_collection=report_collection,
                 average_dropout=average_dropout)

    # Annotate corresponding .txts with dictionary
    #
    #
    # cellfinder_only = [term for term in tissue_check_set if term not in tissue]
    # print("\n".join(cellfinder_only))
    # # Write out adjusted .ann file
    #
    # # TODO - Coverage compared to cover of dictionary in sample
    # percent = len(cellfinder_only) / len(tissue) * 100
    # print('%.2f' % percent)

    # ls_formatter(dict_file="./master_dict", texts_file=f"/Users/withers/Downloads/cellfinder1_brat/{pmcid}.txt",
    #              output_json=f"./{pmcid}_annot.json")
