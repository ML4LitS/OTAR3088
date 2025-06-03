from email.mime import base
from pipelines.model_pipelines import flair_pipeline, hf_pipeline
from utils.helper_functions import create_output_dir, set_seed

from omegaconf import DictConfig, OmegaConf
import hydra
import logging

from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.trainers.plugins.loggers.wandb import WandbLogger


from dotenv import load_dotenv
import os
import wandb 


# os.environ["HYDRA_FULL_ERROR"]="1" #prints full error trace useful for debugging

#load wandb key
load_dotenv()
wandb_token = os.environ.get("WANDB_TOKEN")

#script to do: 
"""
1. set up logger
2. complete hf pipeline
3. harmonise hydra logging and central logging ----> for easy debugging


"""

@hydra.main(config_path="config", config_name="common_config", version_base=None)
def train_model(cfg: DictConfig):
    #login to wandb
    wandb.login(key=wandb_token)

    #set set for reproducibility
    set_seed(cfg.seed) 


    # logging.info(f"Current reproductiblity seed set to {cfg.seed}")
    model_name = cfg.model.name.lower()

    # Convert Hydra config to a plain dict for wandb compatibility
    plain_cfg = OmegaConf.to_container(cfg, resolve=True)

   
    # Apply custom patch for wandb plugin
    WandbLogger.attach_to = flair_pipeline.patched_attach_to

    #init wandb run details and plain_config
    wandb_run = wandb.init(
            project=cfg.logging.wandb.project,
            # entity=cfg.logging.wandb.entity,
            config=plain_cfg, 
            tags=cfg.logging.wandb.tags,
            # mode=cfg.logging.wandb.mode,
            name=f"flair-{wandb.util.generate_id()}",  # custom run name
        )


    if model_name == "flair":
        # Create model output directory ---> Future feature not yet implemented: will be used to save logs once it is setup
        output_dir = create_output_dir(base_path ="./", model_name=model_name)

        #Initialise WandbLogger plugin for flair
        wb_plugin = WandbLogger(
            wandb=wandb,
            emit_alerts=True,
            alert_level=logging.WARNING
        )

        # Create label dictionary and corpus
        label_dict, corpus = flair_pipeline.create_label_dict_corpus(
            label_type=cfg.model.columns[1],
            target_column=cfg.model.columns[0],
            data_folder=cfg.data.data_folder,
            train_file=cfg.model.corpus.train_file,
            dev_file=cfg.model.corpus.dev_file,
            test_file=cfg.model.corpus.test_file,
        )

        # Load embeddings
        embeddings = flair_pipeline.get_embeddings(cfg.model.embeddings)

        # Initialize tagger
        tagger = SequenceTagger(
            embeddings=embeddings,
            tag_dictionary=label_dict,
            **cfg.model.sequence_tagger
        )

        # Train
        trainer = ModelTrainer(tagger, corpus)
        trainer.fine_tune(base_path=output_dir,
                          plugins=[wb_plugin],
                          **cfg.model.fine_tune)
        
        wandb_run.finish()

    elif model_name == "hf":
        raise NotImplementedError(f"This pipeline is yet to be implemented")

    else:
        raise ValueError(f"Unsupported model_type: {model_name}. Choose either 'flair' or 'hf'.")

if __name__ == "__main__":
    train_model()
