
from pipelines.model_pipelines import flair_pipeline, hf_pipeline
from utils.helper_functions import create_output_dir, set_seed, setup_loguru

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
import logging 

#hf
import torch 

from dotenv import load_dotenv
import os
import wandb 


# os.environ["HYDRA_FULL_ERROR"]="1"

def init_wandb_run(cfg:DictConfig):
    plain_cfg = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(
        name=f"{cfg.model.name}-{wandb.util.generate_id()}",
        config=plain_cfg,
        job_type="Model-Training",
        **cfg.logging.wandb
    )
    artifact = wandb.Artifact(
        name=f"{cfg.model.name}-Training",
        description=f"NER model training for {cfg.model.name} using {cfg.data.name}",
        type="model",
        metadata={"Model architecture": cfg.model.name, "Dataset": cfg.data.name}
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

    #init logger
    setup_loguru(cfg.logging)

    #seed reproducibility seed
    set_seed(cfg.seed) 
    logger.info(f"Reproducibility seed set to {cfg.seed}")

    #init device
    device = "cuda" if torch.cuda.is_available() else "cpu" #set device options["cpu", "gpu"]

    logger.info(f"Current device set as: {device}")


    if cfg.enable_wandb:
      import wandb 
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
    output_dir = create_output_dir(base_path=BASE_PATH, name=f"{model_name}_{dataset_name}")
    

    logger.info(f"Current Hydra output dir set at: {HydraConfig.get().runtime.output_dir}")

    
    
    if model_name == "flair" and cfg.enable_wandb:
        flair_pipeline.flair_trainer(cfg, wandb_run, run_artifact, output_dir)

    elif model_name == "hf":
        hf_pipeline.hf_trainer(cfg, wandb_run, run_artifact, output_dir, device)
    else:
        raise ValueError(f"Unsupported model type: {model_name}. Choose 'flair' or 'hf'.")

    wandb_run.finish() if cfg.enable_wandb else None



if __name__ == "__main__": train_model()