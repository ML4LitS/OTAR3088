from dataclasses import asdict
from loguru import logger
from ...shared.trainer_base import HFTrainingOrchestrator
from .modelling import CustomTrainer, CustomCallback, WeightedCustomTrainer


# if __name__ == "__main__":

class NerTrainingOrchestrator(HFTrainingOrchestrator):
    def __init__(self, runner_conf):
        super().__init__(runner_conf)


    def _build_trainer(self):
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
        if self.builder.cfg.model.use_weighted_trainer:
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

        for cb in self.components.callbacks:
            if isinstance(cb, type):
                self.trainer.add_callback(cb(trainer=self.trainer))
            else:
                self.trainer.add_callback(cb)
        
        logger.success("Trainer built Successfully")
        logger.info("Initialising Trainer------->")

    def _log_to_wandb(self):
        pass