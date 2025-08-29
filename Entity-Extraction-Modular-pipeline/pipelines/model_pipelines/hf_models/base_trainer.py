
from loguru import logger

from steps.load_ner_dataset_hf import data_loader
from steps.tokenize_preprocess import tokenize_and_align, cast_to_class_labels

from transformers import AutoModelForTokenClassification, TrainingArguments


from utils.hf_utils import (
                            get_label2id_id2label,
                            get_model,
                            count_entity_labels,
                            count_trainable_params,
                            init_tokenizer_data_collator,
                            CustomCallback,
                            CustomTrainer,
                            prepare_metrics_hf,
                            )


# TODO - Generalise labels/words using data config?
# TODO - Will this pipeline always use hf? Maybe refactor naming of vars

def hf_trainer(cfg,
               wandb_run,
               run_artifact,
               output_dir,
               device):
  hf_checkpoint_name = cfg.model.model_name_or_path
  logger.info(f"Model checkpoint used for this run is: {hf_checkpoint_name}")
  
  # Load datasets:
  train_dataset, val_dataset = data_loader(cfg.data)
  logger.success("Datasets loaded")
  logger.info(f"Train Dataset: {train_dataset}")
  logger.info(f"Sample train dataset: {next(iter(train_dataset))}")

  label_col, text_col = cfg.model.label_col, cfg.model.text_col


  train_entity_count_iob, train_entity_count_wo_iob = count_entity_labels(train_dataset, "labels")
  val_entity_count_iob, val_entity_count_wo_iob = count_entity_labels(val_dataset, "labels")

  train_labels, val_labels = train_entity_count_iob.keys(), val_entity_count_iob.keys()


  unique_tags = list(set(train_labels | val_labels))
  label2id, id2label = get_label2id_id2label(unique_tags)

  # One-hot encode labels to integer values
  train_dataset = cast_to_class_labels(train_dataset, "labels", "words", unique_tags)
  val_dataset = cast_to_class_labels(val_dataset, "labels", "words", unique_tags)

  if cfg.use_wandb:
    wandb_run.log({"Model checkpoint used for this run is": hf_checkpoint_name})
    wandb_run.log({
        "Text column in dataset": text_col,
        "Labels column in dataset": label_col,
        "Unique labels in dataset": list(unique_tags),
        "Labels count in train dataset": dict(train_entity_count_wo_iob),
        "Labels count in val dataset": dict(val_entity_count_wo_iob),
        "Num classes to predict": len(unique_tags)
    })
    wandb_run.log({"Train Label counts in IOB":dict(train_entity_count_iob)}) 
    wandb_run.log({"Validation Label counts in IOB":dict(train_entity_count_iob)})

  # Init tokenizer and data collator
  tokenizer, data_collator = init_tokenizer_data_collator(hf_checkpoint_name)

  tokenize_fn = lambda batch: tokenize_and_align(batch, tokenizer=tokenizer)

  tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
  tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=val_dataset.column_names)

  checkpoint_size = hf_checkpoint_name.split("/")[0]

  # Init training args
  args = TrainingArguments(output_dir=f"{output_dir}/{checkpoint_size}",
                            logging_dir=f"{output_dir}/{checkpoint_size}/{cfg.model.name}-{cfg.data.name}_base_model-trainer_logs",
                            **cfg.model.args)
  logger.info(f"Current training argument: {args}")

  model = get_model(hf_checkpoint_name, 
                    num_labels=len(unique_tags), 
                    label2id=label2id, 
                    id2label=id2label, 
                    device=device)
  logger.info(f"Model initialised as : {model}")

  trainable_params = count_trainable_params(model)
  logger.info(f"Model trainiable params: {trainable_params}")

  #init compute metrics
  compute_metrics = prepare_metrics_hf(unique_tags)
  
  def model_init():
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_checkpoint,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    model.resize_token_embeddings(len(tokenizer)) # Ensure embedding length = that of tokenizer
    return model

  trainer = CustomTrainer(model_init=model_init,
          args=args,
          train_dataset=tokenized_train,
          eval_dataset=tokenized_val,
          processing_class=tokenizer,
          data_collator=data_collator,
          compute_metrics=compute_metrics,
          id2label=id2label, # Explicitly state to avoid NoneType break
          )
  
  trainer.add_callback(CustomCallback(trainer=trainer))

  # Optional hyperparameter tuning

  if cfg.fine_tune:
    logger.info("Will complete train with hyperparameter finetuning")

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        }

    def compute_objective(metrics):
        return metrics["eval_f1"]

    best_run = trainer.hyperparameter_search(
        hp_space=hp_space,
        direction="maximize",
        backend="optuna",  # TODO - change for "ray"/"wandb"?
        n_trials=cfg.tune_trials if hasattr(cfg, "tune_trials") else 10,
        compute_objective=compute_objective,
    )
    logger.info(f"Best hyperparameters found: {best_run}")
    if cfg.use_wandb:
        wandb_run.log({"Best hyperparameters": best_run.hyperparameters})

    # Retrain with the best hyperparameters
    for param, value in best_run.hyperparameters.items():
        setattr(trainer.args, param, value)

    logger.info("Retraining with best hyperparameters...")
    trainer.train()

    # Pick up from training w.o. finetuning
    # TODO - Deduplicate code
    wandb_run.log({"training results": trainer.evaluate(tokenized_train)})
    results = trainer.evaluate()
    best_ckpt_path = trainer.state.best_model_checkpoint
    run_artifact.add_dir(local_path=best_ckpt_path, name="Best Model Checkpoint path for this run")
    
    # TODO - Testing out test review
    # Ideal to reload model form best checkpoint - ensure hf is behaving
    test_model = AutoModelForTokenClassification.from_pretrained(best_ckpt_path, num_labels=len(id2label),
                                                                 id2label=id2label, label2id=label2id)
    trainer = Trainer(model = test_model,
                      args = args,
                      compute_metrics = compute_metrics,
                      tokenizer = tokenizer)
    test_results = trainer.predict(tokenized_datasets["test"])
    # test_results = trainer.evaluate(test_data?)
    
    logger.info(f"Best model checkpoint saved: {best_ckpt_path}")
    #run_artifact.add(results, name="Validation resuls")
    run_artifact.save()
    logger.info("Linking run to wandb registry")
    wandb_run.log_artifact(run_artifact)
    target_save_path=f"{cfg.logging.wandb.run.entity}/{cfg.logging.wandb.registry.registry_name}/{cfg.logging.wandb.registry.collection_name}"
    logger.info(f"Target wandb registry path for this run is set at: {target_save_path}")
    wandb_run.link_artifact(artifact=run_artifact,
                            target_path=target_save_path,
                            aliases=[cfg.model.name, cfg.data.name, checkpoint_size, "base model"]
    )
                          
    logger.success("Artifact logged to registry")

  else:
    logger.info("Training commencing")
    trainer.train()
    logger.info("Training completed....")
    if cfg.use_wandb:
      wandb_run.log({"training results": trainer.evaluate(tokenized_train)})
      results = trainer.evaluate()
      best_ckpt_path = trainer.state.best_model_checkpoint
      run_artifact.add_dir(local_path=best_ckpt_path, name="Best Model Checkpoint path for this run")
      #run_artifact.add(results, name="Validation resuls")
      run_artifact.save()
      logger.info("Linking run to wandb registry")
      wandb_run.log_artifact(run_artifact)
      target_save_path=f"{cfg.logging.wandb.run.entity}/{cfg.logging.wandb.registry.registry_name}/{cfg.logging.wandb.registry.collection_name}"
      logger.info(f"Target wandb registry path for this run is set at: {target_save_path}")
      wandb_run.link_artifact(artifact=run_artifact,
                              target_path=target_save_path,
                              aliases=[cfg.model.name, cfg.data.name, checkpoint_size, "base model"]
      )
                            
      logger.success("Artifact logged to registry")

