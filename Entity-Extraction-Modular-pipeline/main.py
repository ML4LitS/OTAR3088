
from pipelines.model_pipelines.flair_models import flair_pipeline
from pipelines.model_pipelines.hf_models import base_trainer
from utils.helper_functions import create_output_dir, set_seed, setup_loguru
from utils.wandb_utils import init_wandb_run

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
import logging 
import wandb 

import torch 

from dotenv import load_dotenv
import os

#set hydra to print full error trace useful for debugging. Comment out if not needed
os.environ["HYDRA_FULL_ERROR"]="1"
#force wandb cache dir to specific location as it could default to home directory which has limited space
os.environ["WANDB_CACHE_DIR"]="./"
#confirm wandb cache dir is set correctly
print(os.environ["WANDB_CACHE_DIR"])




@hydra.main(config_path="config", config_name="common_config", version_base=None)
@logger.catch
def train_model(cfg: DictConfig):

    #load environment variables
    load_dotenv()
    BASE_PATH = os.environ.get("BASE_PATH")

    #get run model name
    model_name = cfg.model.name.lower()
    dataset_name = cfg.data.name.lower()
    version_name = cfg.data.version_name

    #init logger
    setup_loguru(cfg.logging)

    #seed reproducibility seed
    set_seed(cfg.seed) 
    logger.info(f"Reproducibility seed set to {cfg.seed}")

    #init device
    device = "cuda" if torch.cuda.is_available() else "cpu" #set device options["cpu", "gpu"]

    logger.info(f"Current device set as: {device}")

    wandb_run, run_artifact = None, None
    #disable wandb if not set to true in config
    if not cfg.use_wandb:
      os.environ["WANDB_MODE"] = "disabled"
      logger.info(f"Logging to Wandb is disabled for this run. Local logs can be found at: {cfg.logging.loguru.log_dir}.")
    

    else:
      #init wandb if set to true in config
      wandb_token = os.environ.get("WANDB_TOKEN")
      wandb.login(key=wandb_token)
      wandb_run, run_artifact = init_wandb_run(mode="train", cfg=cfg)
      logger.info(f"Logging to Wandb is enabled for this run. Run logs and metadata will be logged to: {cfg.logging.wandb.run.project}")
      wandb_run.log({"Current device for run" : device})
    

    #init output dir for model logs and results
    output_dir = create_output_dir(base_path=BASE_PATH, name=f"{model_name}_{dataset_name}-{version_name}")

    #log current hydra output dir
    logger.info(f"Current Hydra output dir set at: {HydraConfig.get().runtime.output_dir}")

    
    
    if model_name == "flair":
        flair_pipeline.flair_trainer(cfg, wandb_run, run_artifact, output_dir)

    elif model_name == "hf":
        base_trainer.hf_trainer(cfg, wandb_run, run_artifact, output_dir, device)
    else:
        raise ValueError(f"Unsupported model type: {model_name}. Choose 'flair' or 'hf'.")

    wandb_run.finish() if cfg.use_wandb and wandb_run is not None else None



if __name__ == "__main__": train_model()