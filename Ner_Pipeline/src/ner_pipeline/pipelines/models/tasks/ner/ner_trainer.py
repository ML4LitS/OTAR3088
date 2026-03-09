from dataclasses import asdict
from loguru import logger
from transformers.trainer_callback import EarlyStoppingCallback

from ...shared.trainer_base import HFTrainingOrchestrator
from .modelling import CustomTrainer, CustomCallback, WeightedCustomTrainer
from .metrics import (NervaluateEvaluator, 
                    SeqevalLogger,
                    NervaluateLogger, 
                    decode_all_predictions)

from .trainer_config import NerPredictions

from ner_pipeline.utils.common import set_seed



class NerTrainingOrchestrator(HFTrainingOrchestrator):
    def __init__(self, runner_conf):
        super().__init__(runner_conf)


    def _build_trainer(self):
        set_seed(self.builder.cfg.seed)
        logger.info("Building Trainer----->")
        trainer_kwargs = self.components.trainer_kwargs
        train_dataset, eval_dataset, id2label, compute_metrics = (trainer_kwargs.train_dataset, 
                                                                  trainer_kwargs.eval_dataset,
                                                                  trainer_kwargs.id2label,
                                                                  trainer_kwargs.compute_metrics
                                                                      )
        model, args, processing_class, data_collator = (trainer_kwargs.model,
                                                        trainer_kwargs.args,
                                                        trainer_kwargs.processing_class,
                                                        trainer_kwargs.data_collator

                                                        )
        if self.builder.cfg.task.use_weighted_trainer:
            logger.info("Using Weighted Trainer with SWA")
            label2id = {v:k for k,v in id2label.items()}
            self.trainer = WeightedCustomTrainer(
                                **self.components.strategy_kwargs,
                            train_dataset = train_dataset,
                            eval_dataset = eval_dataset,
                            model = model,
                            args = args,
                            processing_class = processing_class,
                            data_collator = data_collator,
                            id2label = id2label,
                            label2id = label2id,
                            compute_metrics = compute_metrics
            )
        else:
            self.trainer = CustomTrainer(
                                **self.components.strategy_kwargs,
                                train_dataset = train_dataset,
                                eval_dataset = eval_dataset,
                                model = model,
                                args = args,
                                processing_class = processing_class,
                                data_collator = data_collator,
                                id2label = id2label,
                                compute_metrics = compute_metrics,
                                        )
        early_stopping_callback = EarlyStoppingCallback(3)
        self.trainer.add_callback(early_stopping_callback)

        for cb in self.components.callbacks:
            if isinstance(cb, type):
                self.trainer.add_callback(cb(trainer=self.trainer))
            else:
                self.trainer.add_callback(cb)
        
        logger.success("Trainer built Successfully")
        logger.info("Initialising Trainer------->")

    def _compute_ner_metrics_wandb(self):
        self._validate_trainer_built()
        
        logits, label_ids = self.trainer.eval_predictions, self.trainer.eval_label_ids
        true_labels, pred_labels = decode_all_predictions(
                                            logits=logits,
                                            label_ids=label_ids,
                                            id2label=self.trainer.model.config.id2label
                                            )
        
        ner_predictions = NerPredictions(
                                true_labels=true_labels,
                                pred_labels=pred_labels,
                                label_names=self.builder.cfg.task.label_names
                                )                              
        self.seqeval_logger = SeqevalLogger(ner_predictions, self.wandb_run)

        #nervaluate results
        evaluator = NervaluateEvaluator(ner_predictions)
        nervaluate_results = evaluator.run_evaluation()
        #nervaluate logger
        self.nervaluate_logger = NervaluateLogger(nervaluate_results, self.wandb_run)


    def _log_to_wandb(self):
        if not hasattr(self, "seqeval_logger") or not hasattr(self, "nervaluate_logger"):
            self._compute_ner_metrics_wandb()

        #log metrics to wandb
        self.seqeval_logger.log()
        self.nervaluate_logger.log()

        #log model artifacts
        if self.wandb_artifact is not None: 
            logger.info("Linking run to wandb registry")
            self.wandb_artifact.add_dir(local_path=self.best_ckpt_path,
                                        name="best_model_checkpoint_path_for_run")
            self.wandb_artifact.save()
        
            
            self.wandb_run.log_artifact(self.wandb_artifact)
            parts = [
                    self.builder.cfg.logging.wandb.run.entity,
                    self.builder.cfg.logging.wandb.registry.registry_name,
                    self.builder.cfg.logging.wandb.registry.collection_name
                                                        ]
            target_save_path = "/".join(parts)
            logger.info(f"Target wandb registry path for this run is set at: {target_save_path}")

            self.wandb_run.link_artifact(artifact=self.wandb_artifact,
                                target_path=target_save_path,
                                aliases=list(self.wandb_run.tags)
                                )
                            
            logger.success("Artifact logged to registry")
