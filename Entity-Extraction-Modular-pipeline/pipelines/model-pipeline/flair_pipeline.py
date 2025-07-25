


from typing import List, Union, Dict, Tuple
import logging
from loguru import logger

from omegaconf import DictConfig


#flair
from flair.data import Corpus, Dictionary
from flair.datasets import ColumnCorpus

from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from flair.trainers.plugins.loggers.wandb import WandbLogger


# from zenml import step 


import utils



#fixing patching error with flair v 0.15 and wandb compatiblity
def patched_attach_to(self, trainer):
        valid_events = {
            "after_training_batch",
            "training_interrupt",
            "before_training_batch",
            "_training_exception",
            "after_training_loop",
            "after_evaluation",
            "before_training_optimizer_step",
            "before_training_epoch",
            "_training_finally",
            "metric_recorded",
            "after_training",
            "after_training_epoch",
            "after_setup",
        }

        for name, method in self.__class__.__dict__.items():
            if hasattr(method, "_plugin_events"):
                for event in method._plugin_events:
                    if event in valid_events:
                        self.register_hook(getattr(self, name), event)


#@step
def create_label_dict_corpus(label_type: str,
                              target_column: str,
                              data_folder: str,
                              train_file: str,
                              dev_file: str,
                              test_file: str) -> tuple[Dictionary, Corpus]:
    columns = {0: target_column, 1: label_type}
    corpus: Corpus = ColumnCorpus(
        data_folder,
        column_format=columns,
        train_file=train_file,
        dev_file=dev_file,
        test_file=test_file
    )
    label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
    return label_dict, corpus




#@step
def get_embeddings(config: DictConfig):
    return TransformerWordEmbeddings(
        model=config.model,
        layers=config.layers,
        subtoken_pooling=config.subtoken_pooling,
        fine_tune=config.fine_tune,
        use_context=config.use_context,
        allow_long_sentences=config.allow_long_sentences,
        respect_document_boundaries=config.respect_document_boundaries,
        context_dropout=config.context_dropout,
        force_max_length=config.force_max_length,
        use_context_separator=config.use_context_separator,
        transformers_tokenizer_kwargs=config.transformers_tokenizer_kwargs,
        transformers_config_kwargs=config.transformers_config_kwargs,
        transformers_model_kwargs=config.transformers_model_kwargs,
        peft_config=config.peft_config,
        peft_gradient_checkpointing_kwargs=config.peft_gradient_checkpointing_kwargs,
    )



def flair_trainer(cfg, wandb_run, run_artifact, wandb, output_dir):
    
    # Init custom flair wandb patch and apply to Flair WandbLogger plugin
    WandbLogger.attach_to = patched_attach_to

    wb_plugin = WandbLogger(wandb=wandb, emit_alerts=True)

    label_dict, corpus = create_label_dict_corpus(
        label_type=cfg.model.columns[1],
        target_column=cfg.model.columns[0],
        data_folder=cfg.data.data_folder,
        train_file=cfg.model.corpus.train_file,
        dev_file=cfg.model.corpus.dev_file,
        test_file=cfg.model.corpus.test_file,
    )

    wandb_run.log({"Flair label dict": str(label_dict)})
    logger.info(f"Flair label dict: {str(label_dict)}")

    logger.info(f"Initialising embeddings")
    embeddings = get_embeddings(cfg.model.embeddings)
    logger.info(f"Embeddings: {embeddings}")

    tagger = SequenceTagger(
        embeddings=embeddings,
        tag_dictionary=label_dict,
        **cfg.model.sequence_tagger
    )
    logger.info(f"Initialising flair trainer class")
    trainer = ModelTrainer(tagger, corpus)
    logger.info(f"Starting flair model training...")
    trainer.fine_tune(base_path=output_dir, plugins=[wb_plugin], **cfg.model.fine_tune)

    wandb_run.log_artifact(run_artifact)
    wandb_run.log({"Run Test results": trainer.test_results.detailed_results})
    logger.success(f"Flair model training completed. Checkpoint saved in {output_dir}")
