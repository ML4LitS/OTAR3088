from dataclasses import replace
from typing import List, Any, Dict, Optional
from omegaconf import DictConfig

from .trainer_config import BuildComponents, BuildContext, BaseTrainerKwargs
from ner_pipeline.utils.common import set_seed

class BaseHFTrainingCompBuilder:
    """
    Builder class responsible for constructing and incrementally enriching
    necessary training components used in instantiating model training 
    using HuggingFace's `Trainer` class.

    Parameters
    ----------
    context : BuildContext
        Execution context providing configuration, device placement,
        output paths, and experiment logging handles.

    Returns
    -------
    HFTrainerComponents
        A Fully constructed and optionally strategy-augmented training
        components ready to be passed to a HuggingFace `Trainer`.

    """
    def __init__(self, context: BuildContext):
        self.context = context
        self._components = self._build_components()

    @property
    def cfg(self) -> DictConfig:
        return self.context.cfg

    def _build_components(self):
        #init cfg
        cfg = self.cfg
        #init seed for reproducibility
        set_seed(cfg.seed)

        wandb_run, wandb_artifact, output_dir, device = (self.context.wandb_run,
                                                self.context.wandb_aritifact,
                                                self.context.output_dir,
                                                self.context.device
                                                )
