from loguru import logger
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

#from ..common.training_utils import get_encoder_layers, get_model_backbone
from ..core.training_utils import (
                            prepare_datasets, 
                            prepare_training_args,
                            get_model,
                            get_encoder_layers,
                            get_model_backbone,
                            get_training_steps,
                            custom_warmup_steps
                            )




def __reinit_modules(module:nn.Module, initializer_range:float):
  "reinitialises layers to their default pretraining state"
  for layer in module.modules():
    if isinstance(layer, nn.Linear):
        layer.weight.data.normal_(mean=0.0, std=initializer_range)
        if layer.bias is not None:
            layer.bias.data.zero_()
    elif isinstance(layer, nn.LayerNorm):
        layer.bias.data.zero_()
        layer.weight.data.fill_(1.0)


def reinit_last_k_layers(model:nn.Module, k:int, reinit_classifier:bool=False, reinit_pooler:bool=False) -> nn.Module:
    """
    Reinitialize the last k layers of a BERT model.
    Args:
        model (nn.Module): The BERT model to be reinitialised.
        k (int): The number of layers to be reinitialised.
        reinit_classifier (bool): Whether to reinitialise the classifier head.
        reinit_pooler (bool): Whether to reinitialise the pooler(not applicable to a Token Classification problem).
    """
    if reinit_classifier:
        _reinit_classifier(model)
    if reinit_pooler:
        _reinit_pooler(model)
    encoder_layers = get_encoder_layers(model)
    total_layers = len(encoder_layers)
    
    if k <= 0 or k >= total_layers:
      raise ValueError(f"num_layers_to_reinit must be >0 and < total encoder layers {total_layers}")
    
    logger.info(f"This model has {total_layers} encoder layers")
    logger.info(f"Reinitialising top {k} layer(s)")

    top_k_layers = encoder_layers[-k:]  # Slice the top k layers
    for layer in top_k_layers:
        __reinit_modules(layer, initializer_range=model.config.initializer_range)

    logger.info(f"Last {k} layer(s) reinitialized to random weights")
    return model


def _reinit_classifier(model:nn.Module):
  """
  Reinitialises the classifier head of a Encoder Model(E.g BERT)
  """

  for module in model.classifier.modules():
    if isinstance(module, nn.Linear):
      torch.nn.init.xavier_uniform_(module.weight) #reinit weight to a random
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
  logger.info("Classifier head reinitialised to random weights")


def _reinit_pooler(model:nn.Module):
    _, model_backbone = get_model_backbone(model)
    if hasattr(model_backbone, "pooler"):
      dense = model_backbone.pooler.dense
      dense.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
      dense.bias.data.zero_()
      logger.info("Pooler reinitialised to random weights")

    else:
      logger.info("This model has no pooler layer")



def get_optimizer_grouped_parameters(model, llrd:float, weight_decay:float, lr:float):
  """

  Args:
    llrd: Layerwise Learning rate decay factor(default=0.9) options: [0.9, 0.95, 1.0] as indicated in the paper
    weight_decay: weight decay coefficient(default=0.01) options: [0.0, 0.01, 0.001] as indicated in the paper
    lr: base learning rate (default=5e-5)
  """

  no_decay = ["bias", "LayerNorm.weight"]
  model_type = get_model_backbone(model)[0]
  logger.info(f"This model has backbone {model_type.upper()}")
  #no weight decay applied-->same as standard finetuning
  if llrd == 1.0:
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
  else:
    #apply layerwise learning rate decay
    #apply to classifier head/pooler head
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    #apply to embeddings and encoder layer
    if model_type in ["bert", "roberta", "electra"]:
      num_layers = model.config.num_hidden_layers
      layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
      layers.reverse()
      curr_lr = lr
      for layer in layers:
        curr_lr *= llrd
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": curr_lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": curr_lr,
            },
        ]
    else:
      raise NotImplementedError
  return optimizer_grouped_parameters





def apply_reinit(model, config:DictConfig):
  #Reinitialize layers if specified
  if config.reinit_k_layers > 0:
      model = reinit_last_k_layers(model,
                                  k = config.reinit_k_layers,
                                  reinit_classifier=config.reinit_classifier
                                  )
  return model


def apply_llrd(config:DictConfig, model, training_args, train_dataset):
  # Update optimizer parameters if LLRD is specified
  if config.llrd < 1.0:
    optimizer_grouped_params = get_optimizer_grouped_parameters(
        model, 
        llrd=config.llrd, 
        weight_decay=config.weight_decay,
        lr=config.lr
    )
    #init optimizer
    optimizer = AdamW(optimizer_grouped_params, lr=config.lr)

    #get training steps and warmup steps
    training_steps = get_training_steps(
        training_args,
        train_dataset   
    )
    warmup_steps = custom_warmup_steps(training_steps)
    custom_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=training_steps
    )
    return optimizer, custom_scheduler
  return None, None
