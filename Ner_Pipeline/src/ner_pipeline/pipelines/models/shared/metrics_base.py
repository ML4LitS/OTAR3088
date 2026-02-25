from typing import Dict, Any



class MetricsLogger:
    """
    Utility class responsible for logging and persisting training and evaluation metrics.

    This class abstracts metric persistence logic from the training runner,
    ensuring metrics are consistently recorded using the Hugging Face Trainer's
    logging and checkpointing mechanisms.


    Methods:
        log_training_metrics:
            Logs and saves training metrics generated during trainer.train().

        log_eval_metrics:
            Logs and saves evaluation metrics generated during trainer.evaluate().

    Notes:
        This class is designed to be stateless. All required runtime objects
        (trainer, train_results, eval_results) must be passed explicitly to methods.
    """

    def log_training_metrics(self, trainer, train_results, metrics_name="train"):
        """
        Logs training metrics.

        Args:
            trainer (Trainer):
                Hugging Face Trainer instance used for training.

            train_results (TrainOutput):
                Output object returned by trainer.train(), containing training metrics.

            metrics_name (str, optional):
                Namespace used when logging metrics. Defaults to "train".

        """
        self.metrics = train_results.metrics
        self.metrics["train_samples"] = len(trainer.train_dataset)
        trainer.log_metrics(metrics_name, self.metrics)
        trainer.save_metrics(metrics_name, self.metrics)

    def log_eval_metrics(self, trainer, eval_results, metrics_name="eval"):
        """
        Log and persist evaluation metrics.

        Args:
            trainer (Trainer):
                Hugging Face Trainer instance used for evaluation.

            eval_results (Dict[str, float]):
                Dictionary containing evaluation metrics.

            metrics_name (str, optional):
                Namespace used when logging metrics. Defaults to "eval".

        """
        self.metrics.update(eval_results)
        if trainer.eval_dataset is not None:
            self.metrics["eval_samples"] = len(trainer.eval_dataset)
        trainer.log_metrics(metrics_name, self.metrics)
        trainer.save_metrics(metrics_name, self.metrics)