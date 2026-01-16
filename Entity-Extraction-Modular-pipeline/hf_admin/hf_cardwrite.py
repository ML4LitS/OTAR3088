import yaml
from huggingface_hub import DatasetCardData, DatasetCard

# Configuration for the specific dataset
REPO_ID = "OTAR3088/PHEE_train_v1.conll"
COMMIT_MESSAGE = "Upload Dataset Card"

card_data = DatasetCardData(
    language="en",
    license="mit",
    task_categories=["token-classification"],
    task_ids=["named-entity-recognition"],
    pretty_name="PHEE train data",
    tags=["pharmacovigilance", "adverse-event", "medical", "ner"]
)

markdown_content = """
# PHEE train data

## Dataset Description

This dataset contains sentences derived from medical case report abstracts 
curated for adverse events. Split data and CoNLL formatting allows for the
**training of language models**, for **named entity recognition.** The dataset
includes entity annotations or labels. This subsect is the train split.

The creation of the original PHEE dataset is detailed at:

> Sun, Z., Li, J., Pergola, G., Wallace, B. C., John, B., Greene, N., Kim, J.,
> & He, Y. (2022). PHEE: A dataset for pharmacovigilance event extraction from
> text. arXiv preprint arXiv:2210.12560.
> https://arxiv.org/pdf/2210.12560.

---

## Source Data

The port of the original PHEE dataset used for our purposes is detailed here:

Original source repository:  
https://huggingface.co/datasets/sarus-tech/phee

---

## Intended Use

### Primary Use
- Supervised NER training for biomedical NLP tasks

### Not Intended For
- Clinical or patient-level decision making

---

## Dataset Structure

- **Language:** English
- **Splits:** Train / Test / Validation
- **Features:** Text field, BIO label
- **Labels:** Adev ~ 'Adverse Event'

---

## Preprocessing

- Sentence-level segmentation is enforced
- Annotations carried out by 15 annotators in data's original creation
- Present dataset split into train / test / val
- Present dataset labeled in the IOB CoNLL format

---

## Limitations

- Relatively small corpus size compared to large-scale pretraining datasets
- Specific to medical case report abstracts only

---

## Ethical Considerations

- All content originates from publicly available, open-access scientific datasets
- No personal, clinical, or identifiable patient information is included

---

## Citation

If you use this dataset, please cite the original publication:

```bibtex
@article{sun2022phee,
  title   = {PHEE: A dataset for pharmacovigilance event extraction from text},
  author  = {Sun, Z., Li, J., Pergola, G., Wallace, B. C., John, B., Greene, N., Kim, J., & He, Y.},
  journal = {arXiv},
  year    = {2022},
  doi     = {preprint arXiv:2210.12560}
}
```
"""

if __name__ == "__main__":
    print(f"Processing {REPO_ID}...")

    # Manual construction to avoid Jinja2 dependency
    # Convert card_data to dict if needed (DatasetCardData is dict-like)
    data_dict = card_data.to_dict() if hasattr(card_data, "to_dict") else dict(card_data)
    
    # Create YAML string
    yaml_header = yaml.dump(data_dict, sort_keys=False).strip()
    
    # Combine content
    full_content = f"---\n{yaml_header}\n---\n{markdown_content}"
    
    # Create card
    card = DatasetCard(full_content)
    
    # Push to Hugging Face Hub
    try:
        card.push_to_hub(REPO_ID, commit_message=COMMIT_MESSAGE)
        print(f"Successfully pushed dataset card to {REPO_ID}")
    except Exception as e:
        print(f"Error pushing to {REPO_ID}: {e}")