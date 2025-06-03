


from typing import List, Union, Dict, Tuple
import logging

from omegaconf import DictConfig

from flair.data import Corpus, Dictionary
from flair.datasets import ColumnCorpus

from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger



from zenml import step 

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
    # logging.info(f"Corpus label dictionary looks like this: {label_dict}")
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


