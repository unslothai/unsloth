## QLoRA Train and Merge Tests

### Overview
Tests that performing QLoRA training and merging weights to 16-bits post-training maintains same behavior as trained model.

- `test_unsloth_qlora_train_and_merge.py`: Test Unsloth QLoRA train and merge using `FastLanguageModel.from_pretrained`, `FastLanguageModel.get_peft_model`, and `FastLanguageModel.save_pretrained_merged` apis
- `test_hf_qlora_train_and_merge.py`: Test Hugging Face QLoRA train and merge using `from_pretrained`, `get_peft_model`, and `merge_and_unload` apis.
   - Demonstrates that `peft`'s `merge_and_unload` results in loss of accuracy as it requantizes the base layer after merging adapter weights so that the model still contains `Linear4Bit` layers post merging.
   - I (@jeromeku) implemented a custom merge function that maintains the original dtype of the base layer by dequantizing base layer weights, merging adapter weights, and ensuring the original dtype is used for the base layer.

### Usage
Run unsloth test:
```bash
python tests/qlora/test_unsloth_qlora_train_and_merge.py
```
Run huggingface test:
```bash
python tests/qlora/test_hf_qlora_train_and_merge.py
```

### Results
The tests train a QLoRA model on a single prompt dataset
```
QUESTION = "What day was I born?"
ANSWER = "January 1, 2058"
USER_MESSAGE = {"role": "user", "content": QUESTION}
ASSISTANT_MESSAGE = {"role": "assistant", "content": ANSWER}
```

Given that the answer is impossible to answer accurately without finetuning, we can only expect the model to answer the question correctly if the model has been trained on the question.

To check this behavior, we print the model's response to the question before and after training, checking whether the model's response is the same as the answer.


