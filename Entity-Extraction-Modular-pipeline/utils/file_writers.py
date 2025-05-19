from typing import List, Dict, Tuple
from pathlib import Path



def write_to_conll(
    sentences: List[List[Dict[str, str]]], 
    save_path: str, 
    file_name: str
) -> None:
    """
    Writes token-label pairs in CoNLL format to a file.

    Args:
        sentences (List[List[Dict[str, str]]]): 
            A list of sentences, where each sentence is a list of dictionaries 
            with keys 'text' (token) and 'label' (IOB tag).
        save_path (str): Directory where the file will be saved.
        file_name (str): Name of the output file (without extension).

    Returns:
        None
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    output_file = save_path / f"{file_name}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            for token in sentence:
                f.write(f"{token['text']}\t{token['label']}\n")
            f.write("\n")  # Sentence boundary