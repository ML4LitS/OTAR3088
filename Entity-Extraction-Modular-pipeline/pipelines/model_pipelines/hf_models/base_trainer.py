
from loguru import logger

from steps.load_ner_dataset_hf import data_loader
from steps.tokenize_preprocess import tokenize_and_align, cast_to_class_labels

from transformers import TrainingArguments


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







def hf_trainer(cfg,
               wandb_run,
               run_artifact,
               output_dir,
               device):
  hf_checkpoint_name = cfg.model.model_name_or_path
  logger.info(f"Model checkpoint used for this run is: {hf_checkpoint_name}")
  
  #load datasets:
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

  #one-hot encode labels to integer values
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
    wandb_run.log({"Validatio Label counts in IOB":dict(train_entity_count_iob)})

  #init tokenizer and data collator
  tokenizer, data_collator = init_tokenizer_data_collator(hf_checkpoint_name)

  tokenize_fn = lambda batch: tokenize_and_align(batch, tokenizer=tokenizer)

  tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
  tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=val_dataset.column_names)

  checkpoint_size = hf_checkpoint_name.split("/")[0]
  #init training args
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


  trainer = CustomTrainer(model=model,
          args=args,
          train_dataset=tokenized_train,
          eval_dataset=tokenized_val,
          processing_class=tokenizer,
          data_collator=data_collator,
          compute_metrics=compute_metrics,
          id2label=model.config.id2label
          )
  
  trainer.add_callback(CustomCallback(trainer=trainer))
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

