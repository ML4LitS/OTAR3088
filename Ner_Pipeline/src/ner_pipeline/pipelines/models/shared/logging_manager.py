

from pathlib import Path
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from loguru import logger

from .trainer_config_base import TrainingStrategyName
from .factory import format_model_checkpoint_name



from loguru import logger
from omegaconf import DictConfig
from pathlib import Path


class BaseLoguruHelper:
    """
    """

    def __init__(self, cfg:DictConfig):
        self.cfg = cfg
        self.lr = cfg.lr
        self.model_name = format_model_checkpoint_name(cfg.model.model_name_or_path)
        self.data_name = cfg.data.name
        self.data_version = getattr(cfg.data, "version_name", "")
        self.training_kwargs = getattr(cfg, "training_kwargs", "")
        self.training_strategy = cfg.training_strategy
        self.task_type = cfg.task_type

        self.use_data_aug = getattr(cfg, "use_data_aug", False)
        self.data_aug_method = getattr(cfg, "data_aug_method", None)
        self._is_configured = False

    
    @property
    def log_dir(self):
        if not self._is_configured:
            raise ValueError("Logging params are not yet configured."
                            "Call `.configure()` before accessing class attributes")
        return self._log_dir

    @property
    def log_filename(self):
        if not self._is_configured:
            raise ValueError("Logging params are not yet configured."
                            "Call `.configure()` before accessing class attributes")
        return self._log_filename

    def configure(self) -> None:
        """
        Configures Loguru logging and updates the cfg in-place with:
        - logging directory and filename

        This method mutates self.cfg and does not return anything.
        """
        if self._is_configured:
            return
        
        self._log_dir = self._log_data_aug_method(self._build_log_dir())
        self._log_filename = self._build_log_filename()
        self._setup_sink(self._log_filename, self._log_dir)
        self._is_configured = True
        
        

    def _setup_sink(self, log_filename:str, log_dir:Path) -> None:
        print(f"Logging dir set to: {log_dir}")
        log_path = log_dir / f"{log_filename}.log"
        log_dir.mkdir(parents=True, exist_ok=True)
        

        logger.remove(0)
        logger.add(log_path,
            format="[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] | <level>{level}</level> | <cyan>{message}</cyan>",
            mode="w", level="DEBUG",
            retention="4 months"
            )
        logger.success(f"Loguru initialised at: {log_path}")

    def _build_log_filename(self):
        if self.data_version:
            return f"{self.data_name}_{self.data_version}_lr-{self.lr}"
        return f"{self.data_name}_lr-{self.lr}"

    def _log_data_aug_method(self, base_log_dir: Path) -> Path:
        if self.use_data_aug and self.data_aug_method:
            return base_log_dir / f"with_{self.data_aug_method}"
        return base_log_dir / "no_data_aug"

    def _build_log_dir(self) -> Path:
        parts = [
            "logs",
            self.task_type,
            (f"{self.data_name}_{self.data_version}"
            if self.data_version
            else self.data_name),
            self.model_name,
            self.training_strategy,
            self.training_kwargs
        ]
        return Path(*parts)
    
    @classmethod
    def __repr__(cls) -> str:
        return f"{cls.__name__}()"

class ReinitLoguruHelper(BaseLoguruHelper):
    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)
        self.reinit_k_layers = cfg.reinit_k_layers
        self.reinit_classifier = getattr(cfg, "reinit_classifier", False)

    def configure(self):
        """
        Builds on BaseHelper and adds reinitialisation of
        classifier flags to log_filename
        """
        if self._is_configured:
            return
        self._log_filename = self._log_k_layers(self._build_log_filename())
        self._log_dir = self._log_reinit_classifier(self._build_log_dir())
        self._setup_sink(self._log_filename, self._log_dir)
        self._is_configured = True
           

    def _log_k_layers(self, log_filename:str) -> str:
        return f"{log_filename}_{self.reinit_k_layers}K"

    def _log_reinit_classifier(self, log_dir:Path) -> Path:
        if self.reinit_classifier:
          return log_dir / "with_classifier"
        return log_dir / "no_classifier"



class LLRDLoguruHelper(BaseLoguruHelper):
    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)
        self.llrd_factor = cfg.llrd_factor

    def configure(self):
        """
        Builds on BaseHelper on adds LLRD flags to log_filename
        """
        if self._is_configured:
            return
        self._log_filename = self._log_llrd_factor(self._build_log_filename())
        self._log_dir = self._log_data_aug_method(self._build_log_dir()) 
        self._setup_sink(self._log_filename, self._log_dir)
        self._is_configured = True
        

    def _log_llrd_factor(self, log_filename):
        "Appends LLRD Factor from configure to log_filename"
        return f"{log_filename}_llrd-{self.llrd_factor}"


class ReinitLLRDLoguruHelper(ReinitLoguruHelper, LLRDLoguruHelper):
    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)

    def configure(self) -> None:
        """
        Inherits from both `ReinitLoguruHelper & LLRDLoguruHelper`,
        adds `reinit_classifier` and `llrd_factor flags to
        `log_dir` and/or `log_filename`

        """
        if self._is_configured:
            return
        self._log_filename = self._log_k_layers(self._build_log_filename())
        self._log_filename = self._log_llrd_factor(self._log_filename)
        self._log_dir = self._log_reinit_classifier(self._build_log_dir())
        self._setup_sink(self._log_filename, self._log_dir)
        self._is_configured = True



class GroupedLLRDLoguruHelper(BaseLoguruHelper):
    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)
        pass


class LoguruHelperFactory:
    "Factory class responsible for instantiating the correct logging helper"
    _registry = {
        TrainingStrategyName.BASE: BaseLoguruHelper,
        TrainingStrategyName.REINIT: ReinitLoguruHelper,
        TrainingStrategyName.LLRD: LLRDLoguruHelper,
        TrainingStrategyName.REINIT_LLRD: ReinitLLRDLoguruHelper,
        TrainingStrategyName.GROUPED_LLRD: GroupedLLRDLoguruHelper,
    }
    @classmethod
    def create(cls, cfg:DictConfig):
        strategy = TrainingStrategyName(cfg.training_strategy.lower())
        helper = cls._registry[strategy](cfg)
        print(f"Logging helper set to : {helper}")
        return helper
  
