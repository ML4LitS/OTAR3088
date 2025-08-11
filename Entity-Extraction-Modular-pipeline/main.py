
from pipelines.model_pipelines import flair_pipeline, hf_pipeline
#from pipelines.model_pipelines.hf_pipeline import data_loader
from utils.helper_functions import create_output_dir, set_seed, setup_loguru

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
import logging 
import wandb 

import torch 

from dotenv import load_dotenv
import os


# os.environ["HYDRA_FULL_ERROR"]="1"

def init_wandb_run(cfg:DictConfig):

    plain_cfg = OmegaConf.to_container(cfg, resolve=True)
    model_name = cfg.model.name.lower()
    dataset_name = cfg.data.name.lower()
    version_name = cfg.data.version_name
    run = wandb.init(
        name=f"{model_name}_{dataset_name}-{version_name}-{wandb.util.generate_id()}",
        config=plain_cfg,
        job_type="Model-Training",
        **cfg.logging.wandb,
        sync_tensorboard=True if model_name=="flair" else False
    )
    artifact = wandb.Artifact(
        name=f"{model_name}_{dataset_name}-Training",
        description=f"NER model training for {model_name} using {dataset_name}, version {version_name}",
        type="model",
        metadata={"Model architecture": model_name, "Dataset": dataset_name}
    )
    return run, artifact



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


    if cfg.use_wandb:
    #init wandb
      wandb_token = os.environ.get("WANDB_TOKEN")
      wandb.login(key=wandb_token) 
      wandb_run, run_artifact = init_wandb_run(cfg)
      logger.info(f"Logging to Wandb is enabled for this run. Run logs and metadata will be logged to: {cfg.logging.wandb.project}")
      wandb_run.log({"Current device for run" : device})
    else:
      wandb_run, run_artifact = None, None
      logger.info(f"Logging to Wandb is disabled for this run. Local logs can be found at: {cfg.logging.loguru.log_dir}.")

    

    #init output dir for model logs and results
    output_dir = create_output_dir(base_path=BASE_PATH, name=f"{model_name}_{dataset_name}-{version_name}")
    

    logger.info(f"Current Hydra output dir set at: {HydraConfig.get().runtime.output_dir}")

    
    
    if model_name == "flair":
        flair_pipeline.flair_trainer(cfg, wandb_run, run_artifact, output_dir)

    elif model_name == "hf":
        hf_pipeline.hf_trainer(cfg, wandb_run, run_artifact, output_dir, device)
    else:
        raise ValueError(f"Unsupported model type: {model_name}. Choose 'flair' or 'hf'.")

    wandb_run.finish() if cfg.use_wandb else None



if __name__ == "__main__": train_model()