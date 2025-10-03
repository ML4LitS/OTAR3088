from typing import Union, List
import pandas as pd
from datasets import Dataset, DatasetDict
from nervaluate import Evaluator
from IPython import display, HTML


def inference_data_loader(dataset:Union[Dataset, DatasetDict], 
                          split:str="train") -> Dataset:
    """
    Checks dataset structure and composition in the case of presence of multiple splits when loading from huggingface. 
    Args:
        dataset: Dataset or DatasetDict object
        split: Split to use for inference. Options: [train, validation, test]
    Returns:
        datasets.Dataset(obj) 
    """
    if isinstance(dataset, DatasetDict) and split in dataset:
        return dataset[split]
    elif isinstance(dataset, Dataset):
        return dataset
    else:
        raise ValueError("Dataset format not recognized.")


def init_nervaluate(true_label:List[List[str, str]], 
                    pred_labels:List[List[str, str]], 
                    ner_labels:List=["CellLine", "CellType", "Tissues"]
                    ) --> Evaluator: 
    """
    Args: 
        true_labels: Ground Truth Labels
        pred_labels: Model's predictions
        ner_labels: List of unique ner tag names without "O" label. 
    Returns:
        nervaluate.Evaluator Object
    """

    return Evaluator(true_labels, pred_labels, tags=ner_labels, loader='list')


def extract_and_flip_evaluator_result(results):
    """
    Extracts results from a nervaluator object, 
    then passes to a dataframe and returns a transposed format for better visualisation

    """
    extracted_results = {k: v.__dict__ for k, v in results.items()}
    results_df = pd.DataFrame(extracted_results).T
    resuls_df = convert_2_percent(results_df)

    return extracted_results, results_df

def display_entity_tables(results):
    """
    Display nervaluate's results per entity type as a Pandas dataframe
    
    """
    for entity, evals in results.items():
      eval_dict = extract_and_flip_evaluator_result(evals)
      df = pd.DataFrame.from_dict(eval_dict, orient="index")
      df = convert_2_percent(df)
      display(HTML(f"<h3 style='margin-top:20px'>{entity}</h3>"))
      display(df)



def convert_2_percent(df):
  for col in df.columns:
    if col.startswith(("precision", "recall", "f1")):
        df[col] = df[col].apply(lambda x: f"{x*100:.2f}%")
  return df


def display_entity_tables(results):
    for entity, evals in results.items():
        # Convert EvaluationResult objects to dicts
        eval_dict = {eval_type: eval_result.__dict__ for eval_type, eval_result in evals.items()}
        
        # Make dataframe for this entity
        df = pd.DataFrame.from_dict(eval_dict, orient="index")
        df = convert_2_percent(df)
        # Show title + dataframe
        display(HTML(f"<h3 style='margin-top:20px'>{entity}</h3>"))
        display(df)