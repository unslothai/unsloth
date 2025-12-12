# Troubleshooting & FAQs

If you're still encountering any issues with versions or depencies, please use our [Docker image](https://docs.unsloth.ai/get-started/install-and-update/docker) which will have everything pre-installed.

{% hint style="success" %}
**Try always to update Unsloth if you find any issues.**

`pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo`
{% endhint %}

### Running in Unsloth works well, but after exporting & running on other platforms, the results are poor

You might sometimes encounter an issue where your model runs and produces good results on Unsloth, but when you use it on another platform like Ollama or vLLM, the results are poor or you might get gibberish, endless/infinite generations *or* repeated output&#x73;**.**

* The most common cause of this error is using an <mark style="background-color:blue;">**incorrect chat template**</mark>**.** It’s essential to use the SAME chat template that was used when training the model in Unsloth and later when you run it in another framework, such as llama.cpp or Ollama. When inferencing from a saved model, it's crucial to apply the correct template.
* It might also be because your inference engine adds an unnecessary "start of sequence" token (or the lack of thereof on the contrary) so ensure you check both hypotheses!
* <mark style="background-color:green;">**Use our conversational notebooks to force the chat template - this will fix most issues.**</mark>
  * Qwen-3 14B Conversational notebook [**Open in Colab**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_\(14B\)-Reasoning-Conversational.ipynb)
  * Gemma-3 4B Conversational notebook [**Open in Colab**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_\(4B\).ipynb)
  * Llama-3.2 3B Conversational notebook [**Open in Colab**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_\(1B_and_3B\)-Conversational.ipynb)
  * Phi-4 14B Conversational notebook [**Open in Colab**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb)
  * Mistral v0.3 7B Conversational notebook [**Open in Colab**](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_\(7B\)-Conversational.ipynb)
  * **More notebooks in our** [**notebooks docs**](https://docs.unsloth.ai/get-started/unsloth-notebooks)

### Saving to GGUF / vLLM 16bit crashes

You can try reducing the maximum GPU usage during saving by changing `maximum_memory_usage`.

The default is `model.save_pretrained(..., maximum_memory_usage = 0.75)`. Reduce it to say 0.5 to use 50% of GPU peak memory or lower. This can reduce OOM crashes during saving.

### How do I manually save to GGUF?

First save your model to 16bit via:

```python
model.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit",)
```

Compile llama.cpp from source like below:

```bash
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggerganov/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli
cp llama.cpp/build/bin/llama-* llama.cpp
```

Then, save the model to F16:

```bash
python llama.cpp/convert_hf_to_gguf.py merged_model \
    --outfile model-F16.gguf --outtype f16 \
    --split-max-size 50G
```

```bash
# For BF16:
python llama.cpp/convert_hf_to_gguf.py merged_model \
    --outfile model-BF16.gguf --outtype bf16 \
    --split-max-size 50G
    
# For Q8_0:
python llama.cpp/convert_hf_to_gguf.py merged_model \
    --outfile model-Q8_0.gguf --outtype q8_0 \
    --split-max-size 50G
```

## :question:Why is Q8\_K\_XL slower than Q8\_0 GGUF?

On Mac devices, it seems like that BF16 might be slower than F16. Q8\_K\_XL upcasts some layers to BF16, so hence the slowdown, We are actively changing our conversion process to make F16 the default choice for Q8\_K\_XL to reduce performance hits.

## :question:How to do Evaluation

To set up evaluation in your training run, you first have to split your dataset into a training and test split. You should <mark style="background-color:green;">**always shuffle the selection of the dataset**</mark>, otherwise your evaluation is wrong!

```python
new_dataset = dataset.train_test_split(
    test_size = 0.01, # 1% for test size can also be an integer for # of rows
    shuffle = True, # Should always set to True!
    seed = 3407,
)

train_dataset = new_dataset["train"] # Dataset for training
eval_dataset = new_dataset["test"] # Dataset for evaluation
```

Then, we can set the training arguments to enable evaluation. Reminder evaluation can be very very slow especially if you set `eval_steps = 1` which means you are evaluating every single step. If you are, try reducing the eval\_dataset size to say 100 rows or something.

```python
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    args = SFTConfig(
        fp16_full_eval = True,         # Set this to reduce memory usage
        per_device_eval_batch_size = 2,# Increasing this will use more memory
        eval_accumulation_steps = 4,   # You can increase this include of batch_size
        eval_strategy = "steps",       # Runs eval every few steps or epochs.
        eval_steps = 1,                # How many evaluations done per # of training steps
    ),
    train_dataset = new_dataset["train"],
    eval_dataset = new_dataset["test"],
    ...
)
trainer.train()
```

## :question:Evaluation Loop - Out of Memory or crashing.

A common issue when you OOM is because you set your batch size too high. Set it lower than 2 to use less VRAM. Also use `fp16_full_eval=True` to use float16 for evaluation which cuts memory by 1/2.

First split your training dataset into a train and test split. Set the trainer settings for evaluation to:

```python
new_dataset = dataset.train_test_split(test_size = 0.01)

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    args = SFTConfig(
        fp16_full_eval = True,
        per_device_eval_batch_size = 2,
        eval_accumulation_steps = 4,
        eval_strategy = "steps",
        eval_steps = 1,
    ),
    train_dataset = new_dataset["train"],
    eval_dataset = new_dataset["test"],
    ...
)
```

This will cause no OOMs and make it somewhat faster. You can also use `bf16_full_eval=True` for bf16 machines. By default Unsloth should have set these flags on by default as of June 2025.

## :question:How do I do Early Stopping?

If you want to stop the finetuning / training run since the evaluation loss is not decreasing, then you can use early stopping which stops the training process. Use `EarlyStoppingCallback`.

As usual, set up your trainer and your evaluation dataset. The below is used to stop the training run if the `eval_loss` (the evaluation loss) is not decreasing after 3 steps or so.

```python
from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    args = SFTConfig(
        fp16_full_eval = True,
        per_device_eval_batch_size = 2,
        eval_accumulation_steps = 4,
        output_dir = "training_checkpoints", # location of saved checkpoints for early stopping
        save_strategy = "steps",             # save model every N steps
        save_steps = 10,                     # how many steps until we save the model
        save_total_limit = 3,                # keep ony 3 saved checkpoints to save disk space
        eval_strategy = "steps",             # evaluate every N steps
        eval_steps = 10,                     # how many steps until we do evaluation
        load_best_model_at_end = True,       # MUST USE for early stopping
        metric_for_best_model = "eval_loss", # metric we want to early stop on
        greater_is_better = False,           # the lower the eval loss, the better
    ),
    model = model,
    tokenizer = tokenizer,
    train_dataset = new_dataset["train"],
    eval_dataset = new_dataset["test"],
)
```

We then add the callback which can also be customized:

```python
from transformers import EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience = 3,     # How many steps we will wait if the eval loss doesn't decrease
                                     # For example the loss might increase, but decrease after 3 steps
    early_stopping_threshold = 0.0,  # Can set higher - sets how much loss should decrease by until
                                     # we consider early stopping. For eg 0.01 means if loss was
                                     # 0.02 then 0.01, we consider to early stop the run.
)
trainer.add_callback(early_stopping_callback)
```

Then train the model as usual via `trainer.train() .`

## :question:Downloading gets stuck at 90 to 95%

If your model gets stuck at 90, 95% for a long time before you can disable some fast downloading processes to force downloads to be synchronous and to print out more error messages.

Simply use `UNSLOTH_STABLE_DOWNLOADS=1` before any Unsloth import.

```python
import os
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"

from unsloth import FastLanguageModel
```

## :question:RuntimeError: CUDA error: device-side assert triggered

Restart and run all, but place this at the start before any Unsloth import. Also please file a bug report asap thank you!

```python
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
```

If you are training an embedding / bi‑encoder model, you will usually want this flag enabled. See `/basics/embedding-model-fine-tuning` for a full end‑to‑end example.

## :question:All labels in your dataset are -100. Training losses will be all 0.

This means that your usage of `train_on_responses_only` is incorrect for that particular model. train\_on\_responses\_only allows you to mask the user question, and train your model to output the assistant response with higher weighting. This is known to increase accuracy by 1% or more. See our [**LoRA Hyperparameters Guide**](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) for more details.

For Llama 3.1, 3.2, 3.3 type models, please use the below:

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

For Gemma 2, 3. 3n models, use the below:

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)
```

## :question:Some weights of Gemma3nForConditionalGeneration were not initialized from the model checkpoint

This is a critical error, since this means some weights are not parsed correctly, which will cause incorrect outputs. This can normally be fixed by upgrading Unsloth

`pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo`

Then upgrade transformers and timm:

`pip install --upgrade --force-reinstall --no-cache-dir --no-deps transformers timm`

However if the issue still persists, please file a bug report asap!

## :question:NotImplementedError: A UTF-8 locale is required. Got ANSI

See <https://github.com/googlecolab/colabtools/issues/3409>

In a new cell, run the below:

```python
import locale
locale.getpreferredencoding = lambda: "UTF-8"
```

## :green\_book:Citing Unsloth

If you are citing the usage of our model uploads, use the below Bibtex. This is for Qwen3-30B-A3B-GGUF Q8\_K\_XL:

```
@misc{unsloth_2025_qwen3_30b_a3b,
  author       = {Unsloth AI and Han-Chen, Daniel and Han-Chen, Michael},
  title        = {Qwen3-30B-A3B-GGUF:Q8\_K\_XL},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF}}
}
```

To cite the usage of our Github package or our work in general:

```
@misc{unsloth,
  author       = {Unsloth AI and Han-Chen, Daniel and Han-Chen, Michael},
  title        = {Unsloth},
  year         = {2025},
  publisher    = {Github},
  howpublished = {\url{https://github.com/unslothai/unsloth}}
}
```
