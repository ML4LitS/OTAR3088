
import os
from dotenv import load_dotenv

from loguru import logger
import wandb 

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

import torch

from ner_pipeline.utils.common import create_output_dir, set_seed
from ner_pipeline.pipelines.models.tasks.ner.ner_trainer import NerTrainingOrchestrator
from ner_pipeline.pipelines.models.tasks.ner.trainer_builder import NerTrainingCompBuilder
from ner_pipeline.pipelines.models.shared.trainer_config_base import (BuildContext, 
                                                                      PushToHubParams)
from ner_pipeline.pipelines.models.shared.metrics_base import MetricsLogger
from ner_pipeline.pipelines.models.shared.trainer_base import HFTrainingOrchestratorConfig
from ner_pipeline.pipelines.models.shared.experiment_manager import ExperimentSubfolderFactory
from ner_pipeline.pipelines.models.shared.logging_manager import LoguruHelperFactory


@hydra.main(config_path="../config", config_name="common", version_base=None)
@logger.catch
def run(cfg:DictConfig):
    #seed reproducibility seed
    set_seed(cfg.seed) 
    #load environment variables
    load_dotenv()
    BASE_PATH = os.environ.get("BASE_PATH")

    #initialise logger
    loguru_helper = LoguruHelperFactory.create(cfg)
    loguru_helper.configure()

    #build experiment subfolder
    subfolder_builder = ExperimentSubfolderFactory.create(cfg)
    subfolder_builder.build()
    experiment_subfolder = subfolder_builder.subfolder

    #init device
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    logger.info(f"Current device set as: {device}")

    wandb_run, run_artifact = None, None
    #disable wandb if not set to true in config
    if not cfg.use_wandb:
      os.environ["WANDB_MODE"] = "disabled"
      logger.info(f"Logging to Wandb is disabled for this run.")
    

    # else:
    #  wandb logic not yet enabled in pipeline

    output_dir = create_output_dir(base_path=BASE_PATH, 
                                  experiment_subfolder=experiment_subfolder)

    
    #build training context
    context = BuildContext(
                cfg = cfg,
                output_dir = output_dir,
                device = device, 
                #wandb_run =  wandb_run
                #wandb_artifact = wandb_artifact
            )
    #build training components
    training_comp = NerTrainingCompBuilder(context)

    #init training metrics logger
    metrics_logger = MetricsLogger()

    #set push to hub params
    hub_params = PushToHubParams(
                            repo_id = cfg.repo_id,
                            push_to_org_repo = cfg.push_to_org_repo,
                            commit_message = cfg.commit_message
                            )

    #pass everything to training orchestrator config
    orchestrator_conf = HFTrainingOrchestratorConfig(
                                builder = training_comp,
                                metrics_logger = metrics_logger,
                                hub_params = hub_params,
                                publish_model = cfg.publish_model
                                )

    #pass orchestrator config to main training orchestrator
    training_orchestrator = NerTrainingOrchestrator(orchestrator_conf)

    #execute training
    training_orchestrator.execute()

if __name__ == "__main__":
    run()