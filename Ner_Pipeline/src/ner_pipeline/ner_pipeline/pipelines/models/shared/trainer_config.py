from typing import (
                    List, Dict, 
                    Union, Optional, 
                    Callable, Any
                )
from dataclasses import field, dataclass
from omegaconf import DictConfig

from wandb.sdk.wandb_run import Run as WandbRun
from wandb import Artifact as WandbArtifact

import torch.nn as nn
from datasets import Dataset
from transformers import (
    PreTrainedTokenizerBase, 
    PreTrainedTokenizerFast, 
    DataCollatorForTokenClassification, 
    DataCollatorForWholeWordMask,
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments,
    TrainerCallback

)

from .model_factory import CustomCallback



@dataclass
class BuildContext:
    """
    Container for all external inputs required to build HuggingFace
    training components.

    This object represents the *execution context* for a training run,
    separating environment- and run-specific concerns from the
    component construction logic.

    Attributes
    ----------
    cfg : DictConfig
        Hydra configuration object containing all experiment, model,
        data, and training parameters.
    output_dir : str
        Directory where model checkpoints, logs, and artifacts
        will be written.
    device : str
        Device identifier (e.g. "cpu", "cuda", "cuda:0") on which
        the model will be instantiated.
    wandb_run : Optional[WandbRun], default=None
        Active Weights & Biases run for logging, if enabled via cfg.
    run_artifact : Optional[RunArtifact], default=None
        W&B artifact used to log model checkpoints or datasets metadata.
    """
    
    cfg: DictConfig
    output_dir: str
    device: str
    wandb_run: Optional[WandbRun] = None
    wandb_artifact: Optional[WandbArtifact] = None

@dataclass(frozen=True)
class BaseTrainerKwargs:
    """
    Container object for keyword arguments passed directly to the
    HuggingFace `Trainer` constructor.


    Attributes
    ----------
    train_dataset : Dataset
        Tokenized training dataset.
    eval_dataset : Dataset
        Tokenized validation dataset.
    model : torch.nn.Module
        Model instance used for training.
    processing_class : PreTrainedTokenizerBase | PreTrainedTokenizerFast
        Tokenizer used for preprocessing and postprocessing.
    args : TrainingArguments
        HuggingFace training arguments.
    data_collator : DataCollator
        Data collator responsible for batch construction.
    compute_metrics : Callable
        Metric computation function used during evaluation.
    id2label : Dict[int, str]
        Mapping from label IDs to label names.
    """
    train_dataset: Dataset
    eval_dataset: Dataset
    model: nn.Module
    processing_class: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast]
    data_collator: DataCollatorForTokenClassification
    compute_metrics: Callable
    id2label: Dict[int, str]

@dataclass(kw_only=True)
class BuildComponents:
    trainer_kwargs: BaseTrainerKwargs
    strategy_kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    callbacks: Union[List[CustomCallback], List[TrainerCallback]]


@dataclass(frozen=True)
class TaptTrainerKwargs(BaseTrainerKwargs):
    data_collator: Union[DataCollatorForLanguageModeling, DataCollatorForWholeWordMask]
    preprocess_logits_for_metrics: Callable