from typing import List, Dict, Optional
from pathlib import Path



def write_to_conll(
    sentences: List[List[Dict[str, str]]],
    output_path: Optional[str] = None,
    file_name: Optional[str] = None
) -> None:
    """
    Writes token-label pairs in CoNLL format to a file.

    Args:
        sentences (List[List[Dict[str, str]]]):
            A list of sentences, where each sentence is a list of dictionaries
            with keys 'text' (token) and 'label' (IOB tag).
        output_path (Optional[str]): Directory where the file will be saved. 
            Defaults to 'outputs/' if not provided.
        file_name (str): Name of the output file (without extension). Defautls to "conll_file" if not provided.

    Returns:
        None
    """
    output_path = Path(output_path) if output_path else Path("outputs")
    output_path.mkdir(parents=True, exist_ok=True)

    if not file_name:
      file_name = "conll_file"
    output_file = output_path / f"{file_name}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            for token in sentence:
                f.write(f"{token['text']}\t{token['label']}\n")
            f.write("\n")  # Sentence boundary