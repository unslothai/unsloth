---
name: "\U0001F680 Feature request"
about: New features, model support, ideas
title: "[Feature]"
labels: feature request
assignees: ''

---

1. For new models, have you tried:
```python
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    "microsoft/Phi-4-multimodal-instruct",
    trust_remote_code = True,
)
```
If that doesn't work, try using the exact `AutoModel` class:
```python
from transformers import WhisperForConditionalGeneration
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/whisper-large-v3",
    auto_model = WhisperForConditionalGeneration,
)
```
For Sequence Classification / other `AutoModel` classes:
```python
from transformers import AutoModelForSequenceClassification
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/whisper-large-v3",
    auto_model = AutoModelForSequenceClassification,
)
```
2. Otherwise, ask away!
