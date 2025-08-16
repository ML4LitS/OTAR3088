import os
import hydra
import wandb
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv
from utils.helper_functions import create_output_dir, setup_loguru
from steps.tokenize_preprocess import process_mult_file_brat
from loguru import logger
import glob

#load environment variables
load_dotenv()
BASE_PATH = os.environ.get("BASE_PATH")

@logger.catch
@hydra.main(config_path=F"{BASE_PATH}/config", config_name="common_config", version_base=None)
def brat_converter(cfg:DictConfig):
    #init logger
    setup_loguru(cfg.logging) #logging params changed at runtime to dataset instead of model

    BASE_PATH = os.environ.get("BASE_PATH")
    output_dir = create_output_dir(BASE_PATH, cfg.data.version_name, is_model=False, is_datasets=True)


    file_ids = [os.path.basename(i) for i in glob.glob(f"{cfg.data.input_dir}/*.txt")]
    logger.info(f"File ids with extension: {file_ids}")
    file_ids = [i.split(".")[0] for i in file_ids]
    logger.info(f"File ids without extension: {file_ids}")

    process_mult_file_brat(file_ids, text_col=cfg.data.text_col, 
                    label_col=cfg.data.label_col, input_dir=cfg.data.input_dir,
                    filename=cfg.data.conll_filename, 
                    output_dir=output_dir
                    )
    logger.info(f"Generated conll file saved at: {output_dir}")



if __name__ == "__main__": brat_converter()