## Model Registry

### Structure
```
unsloth
    -registry
        __init__.py
        registry.py
        _llama.py
        _mistral.py
        _phi.py
        ...
```

Each model is registered in a separate file within the `registry` module (e.g. `registry/_llama.py`).

Within each model registration file, a high-level `ModelMeta` is created for each model version, with the following structure:
```python
@dataclass
class ModelMeta:
    org: str
    base_name: str
    model_version: str
    model_info_cls: type[ModelInfo]
    model_sizes: list[str] = field(default_factory=list)
    instruct_tags: list[str] = field(default_factory=list)
    quant_types: list[QuantType] | dict[str, list[QuantType]] = field(default_factory=list)
    is_multimodal: bool = False
```

Each model then instantiates a global `ModelMeta` for its specific model version, defining how the model path (e.g. `unsloth/Llama-3.1-8B-Instruct`) is constructed since each model type has a different naming convention.
```python
LlamaMeta_3_1 = ModelMeta(
    org="meta-llama",
    base_name="Llama",
    instruct_tags=[None, "Instruct"],
    model_version="3.1",
    model_sizes=["8"],
    model_info_cls=LlamaModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
)
```

`LlamaModelInfo` is a subclass of `ModelInfo` that defines the model path for each model size and quant type.
```python
class LlamaModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}-{size}B"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)
```

Once these constructs are defined, the model is registered by writing a register_xx_models function.
```python
def register_llama_3_1_models(include_original_model: bool = False):
    global _IS_LLAMA_3_1_REGISTERED
    if _IS_LLAMA_3_1_REGISTERED:
        return
    _register_models(LlamaMeta_3_1, include_original_model=include_original_model)
    _IS_LLAMA_3_1_REGISTERED = True
```

`_register_models` is a helper function that registers the model with the registry.  The global `_IS_XX_REGISTERED` is used to prevent duplicate registration.

Once a model is registered, registry.registry.MODEL_REGISTRY is updated with the model info and can be searched with `registry.search_models`.

### Tests

The `tests/test_model_registry.py` file contains tests for the model registry.

Also, each model registration file is an executable module that checks that all registered models are available on `huggingface_hub`.
```python
python unsloth.registry._llama.py
```

Prints the following (abridged) output:
```bash
✓ unsloth/Llama-3.1-8B
✓ unsloth/Llama-3.1-8B-bnb-4bit
✓ unsloth/Llama-3.1-8B-unsloth-bnb-4bit
✓ meta-llama/Llama-3.1-8B
✓ unsloth/Llama-3.1-8B-Instruct
✓ unsloth/Llama-3.1-8B-Instruct-bnb-4bit
✓ unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit
✓ meta-llama/Llama-3.1-8B-Instruct
✓ unsloth/Llama-3.2-1B
✓ unsloth/Llama-3.2-1B-bnb-4bit
✓ unsloth/Llama-3.2-1B-unsloth-bnb-4bit
✓ meta-llama/Llama-3.2-1B
...
```

### TODO
- Model Collections
    - [x] Gemma3
    - [ ] Llama3.1
    - [x] Llama3.2
    - [x] MistralSmall
    - [x] Qwen2.5
    - [x] Qwen2.5-VL
    - [ ] Qwen2.5 Coder
    - [x] QwenQwQ-32B
    - [x] Deepseek v3
    - [x] Deepseek R1
    - [x] Phi-4
    - [ ] Unsloth 4-bit Dynamic Quants
    - [ ] Vision/multimodal models
- Sync model uploads with registry
- Add utility methods for tracking model stats