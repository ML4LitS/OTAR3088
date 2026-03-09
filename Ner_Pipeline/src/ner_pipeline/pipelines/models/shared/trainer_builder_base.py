from abc import ABC, abstractmethod
from dataclasses import replace
from typing import List, Any, Dict, Optional
from omegaconf import DictConfig

from .modelling_base import TrainingStrategyFactory


class HFTrainingCompBuilder(ABC):
    def __init__(self, context):
        self.context = context

    @abstractmethod
    def _build_components(self):
        pass

    @abstractmethod
    def apply_strategy(self):
        pass


        

