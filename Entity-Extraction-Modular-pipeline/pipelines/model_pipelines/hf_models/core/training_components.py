from omegaconf import DictConfig
from utils.hf_utils import (
                            prepare_metrics_hf,
                            init_tokenizer_data_collator, 
                            )
from .training_utils import (
                            prepare_datasets, 
                            prepare_training_args,
                            get_model,
                            get_model_backbone,
                            get_training_steps,
                            custom_warmup_steps,
                            CustomCallback
                            )

from ..training_strategies.reinit_llrd import (apply_reinit, 
                                   apply_llrd
                            )  
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from loguru import logger








def build_base_training_components(cfg:DictConfig, output_dir:str, device:str, use_wandb, wandb_run=None, run_artifact=None):
    """
    Factory function that builds all HuggingFace training components.
    
    Args:
        cfg (DictConfig): Hydra config object
        output_dir (str): Output directory for model artifacts and logs
        device (str): Device to run the model on ("cpu" or "cuda")
        use_wandb (bool): Whether to log to Weights & Biases
        wandb_run: Wandb run object (if use_wandb is True)
        run_artifact: Wandb artifact object (if use_wandb is True)
    Returns:
        dict: containing tokenized training and validation datasets, tokenizer, model, training args
    """
    
    #Load datasets + labels
    train_dataset, val_dataset, unique_tags, label2id, id2label = prepare_datasets(cfg, wandb_run)

    #Tokenize dataset
    tokenizer, data_collator, tokenize_fn = init_tokenizer_data_collator(cfg.model.model_name_or_path)
    tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=val_dataset.column_names)

    #prepare model
    model = get_model(cfg.model.model_name_or_path, len(unique_tags), label2id, id2label, device)

    #Training args
    training_args = prepare_training_args(cfg, output_dir)

    #prepare metrics
    compute_metrics = prepare_metrics_hf(unique_tags)

    # Optional log to wandb
    if use_wandb and wandb_run is not None:
        wandb_run.log({"Model checkpoint used for this run": cfg.model.model_name_or_path})
        wandb_run.log({
            "Unique labels": unique_tags, 
            "Num classes": len(unique_tags)
        })
    components = {"trainer_kwargs":
        {
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_val,
        "model": model,
        "processing_class": tokenizer,
        "args": training_args,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "id2label": id2label,
    },
    "callbacks": [CustomCallback],
    "strategy_kwargs": {}
    }
    logger.info(f"All training components for this run have been built successfully as below:/n{components}")
    return components


def build_reinit_llrd_components(cfg:DictConfig, output_dir:str, device:str, use_wandb, wandb_run=None, run_artifact=None):
    """
    Factory function that builds all HuggingFace training components for reinitializing and LLRD.
    
    Args:
        cfg (DictConfig): Hydra config object
        output_dir (str): Output directory for model artifacts and logs
        device (str): Device to run the model on ("cpu" or "cuda")
        use_wandb (bool): Whether to log to Weights & Biases
        wandb_run: Wandb run object (if use_wandb is True)
        run_artifact: Wandb artifact object (if use_wandb is True)
    Returns:
        dict: containing tokenized training and validation datasets, tokenizer, model, training args
        Applies reinit and/ LLRD if specified in cfg
    """

    #load base components
    components = build_base_training_components(cfg, output_dir, device, use_wandb, wandb_run, run_artifact)
    
    #apply reinit if specified
    components["trainer_kwargs"]["model"] = apply_reinit(model=components["trainer_kwargs"]["model"], 
                                                        config=cfg) #Flag: prev. 'cfg=cfg' caused error
    
    #apply llrd if specified
    optimizer, lr_scheduler = apply_llrd(cfg,
                                        components["trainer_kwargs"]["model"],
                                        training_args=components["trainer_kwargs"]["args"],
                                        train_dataset=components["trainer_kwargs"]["train_dataset"],
                                        )
    if optimizer is not None and lr_scheduler is not None:
        components["strategy_kwargs"]["optimizers"] = (optimizer, lr_scheduler)
    
    return components

    

        
