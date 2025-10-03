import json

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@misc{sun2022phee,
      title={PHEE: A Dataset for Pharmacovigilance Event Extraction from Text}, 
      author={Zhaoyue Sun and Jiazheng Li and Gabriele Pergola and Byron C. Wallace and Bino John and Nigel Greene and Joseph Kim and Yulan He},
      year={2022},
      eprint={2210.12560},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
Data and Code for [``PHEE: A Dataset for Pharmacovigilance Event Extraction from Text``](https://arxiv.org/abs/2210.12560/)\
"""

_URL = "https://raw.githubusercontent.com/ZhaoyueSun/PHEE/ceea192bc1f1da306980c39e53767176b1f8caec/data/json/"
_URLS = {
    "train": _URL + "train.json",
    "test": _URL + "test.json",
    "dev": _URL + "dev.json",
}


class PHEEConfig(datasets.BuilderConfig):
    """BuilderConfig for PHEE."""

    def __init__(self, **kwargs):
        """BuilderConfig for PHEE.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PHEEConfig, self).__init__(**kwargs)


class PHEE(datasets.GeneratorBasedBuilder):
    """PHEE: A Dataset for Pharmacovigilance Event Extraction from Text"""

    BUILDER_CONFIGS = [
        PHEEConfig(
            name="json",
            version=datasets.Version("1.0.0", ""),
            description="processed structured data",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            homepage="https://github.com/ZhaoyueSun/PHEE",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "is_mult_event": datasets.Value("bool"),
                    "annotations": [
                        {
                            "events": [
                                {
                                    "event_id": datasets.Value("string"),
                                    "event_type": datasets.Value("string"),
                                    "event_data": datasets.Value("string"),
                                }
                            ]
                        }
                    ],
                }
            ),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
            datasets.SplitGenerator(name='dev', gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples parsed from json."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            for key, line in enumerate(f):
                obs = json.loads(line)
                yield key, {
                    "id": obs["id"],
                    "context": obs["context"],
                    "is_mult_event": obs["is_mult_event"],
                    "annotations": [
                        {
                            "events": [
                                {
                                    "event_id": event["event_id"],
                                    "event_type": event["event_type"],
                                    "event_data": json.dumps(event),
                                }
                            for event in annotation["events"]]
                        }
                    for annotation in obs["annotations"]],
                }