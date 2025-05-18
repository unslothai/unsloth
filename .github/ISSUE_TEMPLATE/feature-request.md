---
name: Feature Request
about: New features, model support, ideas
title: "[Feature]"
labels: feature request
assignees: ''

---

For new models, have you tried:
```python
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    "microsoft/Phi-4-multimodal-instruct",
    trust_remote_code = True,
)
from transformers import AutoModelForSequenceClassification
model, tokenizer = FastModel.from_pretrained(
    auto_model = AutoModelForSequenceClassification,
)
```
