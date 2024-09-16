import pandas as pd
import torch
from transformers import AutoTokenizer


def tokenize_and_align_labels(examples, device, tokenizer):
    """
    Tokenize and align labels for sequence labelling.

    Args:
        examples (dict): A dictionary containing 'tokens' and 'ner_tags'.
        device (torch.device): The device to move tensors to (CPU or GPU).
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding the text.

    Returns:
        dict: Tokenized inputs with aligned labels for sequence labelling.
    """
    task = "ner"
    label_all_tokens = True
    tokenized_inputs = tokenizer(
        examples["tokens"],
        max_length=512,
        truncation=True,
        padding=True,
        is_split_into_words=True,
        return_tensors='pt'
    )
    
    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # First token of each word
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    labels = torch.tensor(labels).to(dtype=torch.int64).to(device)  # Move labels to device
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def convert_IOB_transformer(test_list, pattern):
    """
    Converts a sequence of tokens/tags to IOB2 format.

    Args:
        test_list (list): A list of tokens or tags.
        pattern (str): A pattern used to split the sequence.

    Returns:
        list: A list of lists, with tokens/tags grouped.
    """
    new_list = []
    sub_list = []
    
    for i in test_list:
        if i != pattern:
            sub_list.append(i)
        else:
            new_list.append(sub_list)
            sub_list = []
    
    return new_list


def get_token_ner_tags(data, label2id):
    """
    Transforms NER tags and tokens from the dataframe into structured format.

    Args:
        data (pd.DataFrame): DataFrame containing 'tokens' and 'ner_tags'.
        label2id (dict): Dictionary mapping labels to IDs.

    Returns:
        pd.DataFrame: DataFrame with aligned tokens and NER tags.
    """
    # Mapping NER tags
    ner_tag_list_ = data['ner_tags'].map(label2id).fillna(
        '#*#*#*#*#*#*#*#*').tolist()
    
    token_list_ = data['tokens'].tolist()

    # Convert tokens and NER tags using IOB2 format conversion
    token_list = convert_IOB_transformer(test_list=token_list_, pattern='')
    ner_tag_list = convert_IOB_transformer(test_list=ner_tag_list_, pattern='#*#*#*#*#*#*#*#*')
    
    df = pd.DataFrame(list(zip(token_list, ner_tag_list)),
                      columns=['tokens', 'ner_tags'])

    return df


def convert2int(col):
    """
    Converts list elements from float to int.

    Args:
        col (list): List of numeric NER tags.

    Returns:
        list: List of integer NER tags.
    """
    if col is None:
        return None
    return [int(x) for x in col]


#callable func: Others can be used independently if desired
def process_dataset(data, label2id, device, model_checkpoint='bioformers/bioformer-8L'):
    """
    Process a dataset to tokenize and align labels for NER task.

    Args:
        data (pd.DataFrame): DataFrame containing 'tokens' and 'ner_tags'.
        label2id (dict): Mapping of labels to IDs.
        device (torch.device): The device to move tensors to (CPU or GPU).
        model_checkpoint (str): Model checkpoint name to use for tokenizer. Default is "Bioformer".

    Returns:
        torch.Tensor: Tokenized and aligned dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    processed_data = get_token_ner_tags(data, label2id)
    tokenized_data = tokenize_and_align_labels(processed_data, device, tokenizer)
    return tokenized_data
