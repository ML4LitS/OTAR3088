
from .core.training_components import (
                                            build_base_training_components,
                                            build_reinit_llrd_components, 
                                            )

from .core.training_utils import CustomCallback, CustomTrainer
from utils.helper_functions import set_seed
from loguru import logger


#define a strategy registry for all available training strategies

STRATEGIES = {
    "base": build_base_training_components,
    "reinit_only": build_reinit_llrd_components,
    "llrd_only": build_reinit_llrd_components,
    "reinit_llrd": build_reinit_llrd_components
}



def hf_trainer(cfg, wandb_run, run_artifact, output_dir, device):
  
  strategy = cfg.training_strategy.lower()
  if strategy not in STRATEGIES:
    raise ValueError(f"Training strategy {strategy} not recognised. Current available strategies are: {list(STRATEGIES.keys())}")
  
  build_hf_training_components = STRATEGIES[strategy]
  logger.info(f"Training strategy for this run is set to: {strategy}")
  set_seed(cfg.seed)
  components = build_hf_training_components(cfg, output_dir, device, cfg.use_wandb, wandb_run, run_artifact)
  
  #initialise trainer based on defined training strategy
  trainer = CustomTrainer(**components["trainer_kwargs"], 
                          **components["strategy_kwargs"])

  #add custom callbacks if defined
  for callback in components.get("callbacks", []):
    trainer.add_callback(callback(trainer=trainer))
  logger.success("Trainer initialised")
  logger.info("Training commencing...")
  trainer.train()
  logger.info("Training completed...")
  logger.info("Running final evaluation on validation set...")
  results = trainer.evaluate()
  logger.info(f"Final evaluation results: {results}")

  if cfg.use_wandb:
    wandb_run.log({"Training results on this run": trainer.evaluate(components["trainer_kwargs"]["train_dataset"]),
                  "Validation results on this run": results,
                  "Training strategy used": strategy,
                  "Best model checkpoint path": trainer.state.best_model_checkpoint,              
                  })
    if cfg.reinit_classifier:
      wandb_run.log({"Classifier reinitialised for this run"})
    if strategy in ["llrd_only", "reinit_llrd"]:
      wandb_run.log({
        "LLrd value used for this run": cfg.llrd,
        "Extra run metadata": components["metadata"]
      })
    
    #logging metadata to wandb artifact
    run_artifact.add_dir(local_path=trainer.state.best_model_checkpoint, name="best_model_checkpoint_path_for_run")
    run_artifact.save()
    wandb_run.log_artifact(run_artifact)
    
    logger.info("Linking run to wandb registry")
    wandb_run.log_artifact(run_artifact)
    checkpoint_size = cfg.model.model_name_or_path.split("/")[0]
    target_save_path=f"{cfg.logging.wandb.run.entity}/{cfg.logging.wandb.registry.registry_name}/{cfg.logging.wandb.registry.collection_name}"
    logger.info(f"Target wandb registry path for this run is set at: {target_save_path}")
    wandb_run.link_artifact(artifact=run_artifact,
                            target_path=target_save_path,
                             aliases=[cfg.model.name, cfg.data.name, f"{cfg.data.name}_{cfg.data.version_name}", checkpoint_size, strategy]
    )
                           
    logger.success("Artifact logged to registry")



