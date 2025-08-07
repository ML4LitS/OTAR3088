#from steps.preprocess import data_splitter, process_dataset
import os
import hydra
import wandb
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv
from utils.helper_functions import create_output_dir
from steps.tokenize_preprocess import data_splitter_brat, process_dataset_brat


#load environment variables
load_dotenv()
BASE_PATH = os.environ.get("BASE_PATH")
WANDB_TOKEN = os.environ.get("WANDB_TOKEN")

#To-do

"""
Integrate loguru logging
"""


@hydra.main(config_path=F"{BASE_PATH}/config", config_name="common_config", version_base=None)
def brat_preprocessor(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))

    #login wandb
    wandb.login(key=WANDB_TOKEN)

    #init wandb run

    wandb_run = wandb.init(
        project = cfg.logging.wandb.project,
        entity = cfg.logging.wandb.entity,
        name = cfg.data.version_name,
        tags = [cfg.data.version_name, "ner", "cleaned_data"],
        dir = f"{BASE_PATH}/logs/{cfg.data.name}",
        #reinit = "finish_previous"
        
    )

    wandb_artifact = wandb.Artifact(
        name=cfg.data.version_name,
        description="Preprocessed Cell-finder dataset to ML ready format",
        type="dataset",
        metadata= {
            "data-source": "cell-finder",
            "entity-collection-in-preprocessed-version": ["cell-line", "cell-type",
                                                     "tissue"],
            "total number of sentences": 2100,
            "total number of tokens": 65000,
            "article_type": "full_text",
            "total number of articles": 10, 
            #"total number of annotated entities": null
            }

    )

    BASE_PATH = os.environ.get("BASE_PATH")
    output_dir = create_output_dir(BASE_PATH, cfg.data.version_name, is_model=False, is_datasets=True)
    split_dict = data_splitter_brat(cfg.data.input_dir, cfg.data.train_ratio, cfg.data.seed)
    process_dataset_brat(split_dict,cfg.data.input_dir, output_dir)
    wandb_run.log_artifact(wandb_artifact)
    wandb_run.finish()



if __name__ == "__main__": brat_preprocessor()