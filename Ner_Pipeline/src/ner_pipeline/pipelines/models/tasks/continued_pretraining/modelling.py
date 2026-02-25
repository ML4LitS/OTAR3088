

import math
from itertools import chain
from typing import Union

from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

import evaluate
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import TrainingArguments, PreTrainedTokenizer, PreTrainedTokenizerFast

from ...shared.factory import format_model_checkpoint_name





def initialise_new_embeddings(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast],
    init_strategy: str = "match_old",  # ["match_old", "normal"]
):
    """
    Reinitialise embeddings for newly added tokens in-place.

    Assumes:
    - tokenizer has been extended BEFORE calling this
    - model.resize_token_embeddings(len(tokenizer)) has been called

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizer
    init_strategy : str
        - "match_old": match mean/std of original embeddings (recommended)
        - "normal": N(0, initializer_range)
    """

    embedding_layer = model.get_input_embeddings()
    embedding_weight = embedding_layer.weight

    old_vocab_size = embedding_weight.shape[0] - (
        len(tokenizer) - tokenizer.vocab_size
    )
    new_vocab_size = embedding_weight.shape[0]

    if new_vocab_size <= old_vocab_size:
        logger.info("No new tokens detected. Skipping embedding reinitialisation.")
        return model

    new_token_ids = list(range(old_vocab_size, new_vocab_size))
    logger.info(f"Reinitialising {len(new_token_ids)} new token embeddings")

    with torch.no_grad():
        if init_strategy == "match_old":
            old_embs = embedding_weight[:old_vocab_size]
            mean = old_embs.mean(dim=0)
            std = old_embs.std(dim=0)

            embedding_weight[new_token_ids] = (
                torch.randn_like(embedding_weight[new_token_ids]) * std + mean
            )

        elif init_strategy == "normal":
            initializer_range = getattr(
                model.config, "initializer_range", 0.02
            )
            embedding_weight[new_token_ids].normal_(
                mean=0.0, std=initializer_range
            )

        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

    logger.success(
        f"New embeddings initialised | std={embedding_weight[new_token_ids].std().item():.4f}"
    )

    return model




def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)



def compute_perplexity(metrics, loss):
    try:
        perplexity = math.exp(metrics[loss])
    except OverflowError:
        perplexity = float("inf")
    return perplexity


def group_texts(examples: Dataset, max_seq_len: int):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_seq_len) * max_seq_len
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)]
        for k, t in concatenated_examples.items()
    }
    return result






def get_experiment_subfolder(cfg: DictConfig) -> str:
    """
    Construct a unique experiment subfolder path based on model type, 
    training strategy, and experiment-specific parameters.

    """
    training_strategy = cfg.training_strategy.lower()
    model_name = cfg.model_checkpoint
    data_name = cfg.dataset_name
    data_kwargs = cfg.data_kwargs

    # Base folder: model/strategy
    base_folder = f"Continued_pretraining/TAPT/{model_name}/{data_kwargs}-{training_strategy}"

    if cfg.use_whole_word_mask:
        base_folder += "_ww_mask"

    #if training_strategy in ["tapt_reinit_only", "tapt_reinit_llrd"]:
    if "reinit" in training_strategy:
        experiment_subfolder = f"{base_folder}/{training_strategy}_{cfg.reinit_k_layers}K"

        #Append classifier reinitialisation flag if activated
        if getattr(cfg, "reinit_classifier", False):
            experiment_subfolder += "_with_reinit_classifier"
        else:
            experiment_subfolder += "_no_reinit_classifier"

        # Append LLRD value, if LLRD value not == 1.0, only true if training_strategy==reinit_llrd
        if training_strategy in ["tapt_reinit_llrd","llrd_only"]:
            llrd_value = cfg.llrd
            if abs(llrd_value - 1.0) > 1e-6:
                experiment_subfolder += f"_llrd{llrd_value}"

    else:
        #For base 
        experiment_subfolder = base_folder

    return experiment_subfolder




def tokenize_func(example, text_col, tokenizer, block_size,is_truncate):
    results = tokenizer(example[text_col],
                        truncation=is_truncate,
                         #padding="max_length",
                         max_length=block_size,
                         return_special_tokens_mask=True,
                         return_attention_mask=True,
                     
                       )
    # if tokenizer.is_fast:
    #     result["word_ids"] = [result.word_id(i) for i in range(len(result['input_ids']))]

    return results
def clean_text(example, text_col):
    sentences = [x.strip() for x in example['Sentences']]
    return {"Sentences": sentences}


def get_tapt_training_args(cfg, output_dir):
    return TrainingArguments(
        output_dir=f"{output_dir}-LR_{str(cfg.lr)}",
        overwrite_output_dir=True,
        eval_strategy="epoch",
        #warmup_steps=500,
        gradient_accumulation_steps=2,
        #eval_steps=10,
        #logging_steps=10,
        save_strategy="epoch",
        learning_rate=cfg.lr,
        seed=cfg.seed,
        data_seed=cfg.seed,
        adam_epsilon=cfg.adam_epsilon,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        dataloader_drop_last=False,
        remove_unused_columns=False if cfg.use_whole_word_mask else True,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        save_total_limit=None,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_grad_norm=1.0,
        push_to_hub=cfg.push_to_hub,
        hub_strategy="end",
        report_to="none",
        fp16=True,
        greater_is_better=False 
    )


def get_logging_params(BASE_PATH, cfg: DictConfig):
    """
    Builds dynamic logging and tracking configurations for different training strategies.
    Ensures consistency between Loguru and Weights & Biases logging directories.
    """
    strategy = cfg.training_strategy.lower()
    lr = cfg.lr
    model_checkpoint = get_checkpoint_name(cfg.model_checkpoint)
    data_name =   get_checkpoint_name(cfg.dataset_name)
    if cfg.data_version:
        data_version = cfg.data_version 
        data_log = f"{data_name}_{data_version}" 
    else:
        data_version = ""
        data_log = data_name
        
        
    base_log_dir = f"{BASE_PATH}/logs/Continued_Pretraining/TAPT/{data_log}/{model_checkpoint}/{strategy}"
    base_log_filename = f"{data_log}-LR_{lr}"
    #wandb_tags = [strategy, model_name, data_name, "ner", "hydra"]
    #wandb_run_name = f"{model_name}-{data_name}_{data_version}-{strategy}-LR_{lr}"
    log_dir = f"{base_log_dir}/whole_word_mask" if cfg.use_whole_word_mask else base_log_dir
    log_filename = f"{base_log_filename}_model.log"

    if strategy == "tapt_base":
        log_filename = f"{base_log_filename}_model.log"

    #elif strategy in ["tapt_reinit_only", "tapt_reinit_llrd"]:
    elif "reinit" in strategy:
        # wandb_run_name += f"-{cfg.reinit_k_layers}K-Layers"
        # wandb_tags.append(f"{cfg.reinit_k_layers}K-Layers")

        if getattr(cfg, "reinit_classifier", cfg.reinit_classifier):
            #wandb_run_name += "-with_reinit_classifier"
            log_dir = f"{log_dir}/with_reinit_classifier"
            #wandb_tags.append("with_reinit_classifier")
        else:
            #wandb_run_name += "-no_reinit_classifier"
            log_dir = f"{log_dir}/without_reinit_classifier"
            #wandb_tags.append("without_reinit_classifier")

        if strategy == "tapt_reinit_llrd" and hasattr(cfg, "llrd") and abs(cfg.llrd - 1.0) > 1e-6:
            log_filename = f"{base_log_filename}_{cfg.reinit_k_layers}K_LLRD-{cfg.llrd}_model.log"
            #wandb_run_name += f"-LLRD-{cfg.llrd}"
            #wandb_tags.append(f"LLRD-{cfg.llrd}")
        else: #reinit-only
            log_filename = f"{base_log_filename}_{cfg.reinit_k_layers}K_model.log"

    elif strategy == "tapt_llrd_only":
        if hasattr(cfg, "llrd") and abs(cfg.llrd - 1.0) > 1e-6:
            log_filename = f"{base_log_filename}_llrd-{cfg.llrd}_model.log"
            #wandb_run_name += f"-LLRD-{cfg.llrd}"
            #wandb_tags.append(f"llrd-{cfg.llrd}")
    # elif strategy == "freeze_layers":
    #     n_layers = getattr(cfg, "n_layers_to_freeze", cfg.n_layers_to_freeze)
    #     if n_layers is not None:
    #         log_filename = f"{base_log_filename}_freeze-{n_layers}L_model.log"
    #         #wandb_run_name += f"-freeze-{n_layers}L"
    #         #wandb_tags.append(f"freeze_bottom_{n_layers}-Layers")
    else:
        log_filename = f"{base_log_filename}_strategy_model.log"


    log_dir = log_dir
    log_filename = log_filename

    logger.info(f"Log files for this run are saved to: {log_dir}/{log_filename}")
    return log_dir, log_filename
   


