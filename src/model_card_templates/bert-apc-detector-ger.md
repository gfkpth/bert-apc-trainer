---
pipeline_tag: token-classification
language:
- de
license: apache-2.0
repo_url: https://github.com/yourusername/bert-apc-trainer
---

# BERT APC Detector (German)

## Model Details
- **Base Model**: bert-base-german-cased
- **Task**: Aboutness Person Coreference (APC) detection via token classification
- **Label Scheme**: BILOU (Begin, Inside, Last, Unit, Outside)
- **Training Data**: DWDS corpus (German)

## Intended Use
Detect mentions of adnominal person constructions ("wir Linguisten", "ihr von der Sonne verwöhnten Inselbewohner:innen", "du Trottel") in German text.

## Training Procedure
- **Optimizer**: AdamW
- **Learning Rate**: {learning_rate}
- **Batch Size**: {batch_size}
- **Epochs**: {num_epochs}
- **Hardware**: GPU

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("gfkpth/bert-apc-detector-ger")
model = AutoModelForTokenClassification.from_pretrained("gfkpth/bert-apc-detector-ger")
```


## Evaluation Results

| Metric | Score |
|--------|-------|
| Accuracy | {accuracy:.4f} |
| F1 | {f1:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |

Evaluated on held-out test set with {num_test_samples} example sentences.

## Repository

[GitHub](https://github.com/gfkpth/bert-apc-trainer)

## Citation

If you use this model, please cite:

```
@software{hoehn2026bert-apc-ger,
  title={BERT APC Detector German},
  author={Georg F.K. Höhn},
  year={2026},
  url={https://huggingface.co/gfkpth/bert-apc-detector-ger}
}
```