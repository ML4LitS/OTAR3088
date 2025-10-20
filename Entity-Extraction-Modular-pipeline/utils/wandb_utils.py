from typing import Optional, Dict
from omegaconf import DictConfig, OmegaConf
import wandb


def init_wandb_run_train(
    cfg: DictConfig,
    run_name: Optional[str] = None,
    job_type: str = "Training",
    artifact_name: Optional[str] = None,
    artifact_description: Optional[str] = None,
    artifact_type: str = "model",
    artifact_metadata: Optional[Dict] = None,
):
    """Initialize W&B run + artifact for training."""
    plain_cfg = OmegaConf.to_container(cfg, resolve=True)

    model_name = cfg.model.name.lower()
    dataset_name = cfg.data.name.lower()
    version_name = cfg.data.version_name
    training_strategy = cfg.training_strategy

    default_run_name = run_name or f"{model_name}_{dataset_name}-{wandb.util.generate_id()}"

    run = wandb.init(
        name=default_run_name,
        reinit=True,
        job_type=job_type,
        config=plain_cfg,
        **cfg.logging.wandb.run,
        sync_tensorboard=True if model_name == "flair" else False,
    )

    artifact = wandb.Artifact(
        name=artifact_name or f"{model_name}_{dataset_name}_{training_strategy}-Training",
        description=artifact_description
        or f"NER model training for {model_name} using {dataset_name}, version {version_name} and strategy: {training_strategy}",
        type=artifact_type,
        metadata=artifact_metadata
        or {"Model architecture": model_name, 
        "Dataset": dataset_name, 
        "Mode": "train", 
        "Strategy" : training_strategy
        },
    )

    return run, artifact


def init_wandb_run_inference(
    run_name: Optional[str] = None,
    job_type: str = "Inference",
    project: Optional[str] = None,
    entity: Optional[str] = None,
):
    """Initialize W&B run + artifact for inference."""
    default_run_name = run_name or f"inference-{wandb.util.generate_id()}"

    run = wandb.init(
        name=default_run_name,
        job_type=job_type,
        reinit=True,
        project=project,
        entity=entity
    )

    return run


def init_wandb_run(mode: str, **kwargs):
    """
    Wrapper for initializing W&B runs in either training or inference mode.

    Args:
        mode (str): "train" or "inference".
        **kwargs: Arguments passed to the underlying train/inference init functions.

    Returns:
        run (wandb.Run): W&B run object.
        artifact (wandb.Artifact): W&B artifact object.
    """
    mode = mode.lower()
    if mode == "train":
        if "cfg" not in kwargs:
            raise ValueError("cfg must be provided when mode='train'")
        return init_wandb_run_train(**kwargs)
    elif mode == "inference":
        return init_wandb_run_inference(**kwargs)
    else:
        raise ValueError("Invalid mode. Must be 'train' or 'inference'.")
