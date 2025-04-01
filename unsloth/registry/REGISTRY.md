## Model Registry

### Structure

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

Once these constructs are defined, the model is registered in the `registry` module by calling `register_models` with the `ModelMeta` and `ModelInfo` classes.

