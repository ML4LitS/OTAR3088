from typing import (
                    List, Dict, 
                    Union, Optional, 
                    Callable, Any,
                    Type
                )
from dataclasses import dataclass

from transformers import DataCollatorForTokenClassification
    
from  ...shared.trainer_config_base import BaseTrainerKwargs
from ...shared.trainer_base import HFTrainingOrchestratorConfig



@dataclass(frozen=True)
class NerTrainerKwargs(BaseTrainerKwargs):
    """
    Ner container object for keyword arguments passed directly to the
    HuggingFace `Trainer` constructor specific for NER Training.
    Inherits from `BaseTrainerKwargs`
    """
    data_collator: DataCollatorForTokenClassification
    id2label: Dict[int, str]



@dataclass(frozen=True)
class NerTrainingOrchestratorConfig(HFTrainingOrchestratorConfig):
    pass






