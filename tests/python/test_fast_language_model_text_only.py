"""Text-only FastLanguageModel routing for vision-capable configs."""

import ast
import copy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LOADER_PATH = REPO_ROOT / "unsloth" / "models" / "loader.py"
VISION_PATH = REPO_ROOT / "unsloth" / "models" / "vision.py"
UTILS_PATH = REPO_ROOT / "unsloth" / "models" / "_utils.py"


def _source(path):
    return path.read_text()


def _class_method(tree, class_name, method_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def _assigns_name(method, target_name, predicate):
    """True when the method contains `target_name = <value>` and predicate(value)."""
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == target_name:
                if predicate(node.value):
                    return True
    return False


def _calls_function(method, func_name):
    """True when the method calls `func_name(...)` (bare name, not attribute)."""
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == func_name
        ):
            return True
    return False


def _names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _param_default(method, name):
    # Default-value AST node for a named parameter, or None.
    args = method.args
    params = list(args.args) + list(args.kwonlyargs)
    defaults = list(args.defaults) + list(args.kw_defaults)
    return dict(zip([p.arg for p in params][-len(defaults) :], defaults)).get(name)


def _load_text_only_namespace():
    # Exec the _utils text-only helpers into one namespace (no unsloth import),
    # in dependency order so cross-references resolve.
    source = _source(UTILS_PATH)
    import transformers
    from packaging.version import Version

    ns = {
        "copy": copy,
        "Version": Version,
        "transformers_version": transformers.__version__,
    }
    funcs = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    for name in (
        "resolve_model_class",
        "_is_family_text_decoder",
        "_remap_text_only_skip_modules",
        "_get_text_only_config",
        "_get_text_only_key_mapping",
        "_apply_text_only_key_mapping",
    ):
        if name in funcs:
            exec(funcs[name], ns)
    return ns


def _load_text_only_helper():
    return _load_text_only_namespace()["_get_text_only_config"]


def test_gemma3_vision_config_resolves_to_text_config():
    transformers = pytest.importorskip("transformers")
    helper = _load_text_only_helper()

    config = transformers.Gemma3Config()
    text_config = helper(config, "google/gemma-3-27b-it")

    assert isinstance(text_config, transformers.Gemma3TextConfig)
    assert text_config.model_type == "gemma3_text"
    model_class = transformers.AutoModelForCausalLM._model_mapping[type(text_config)]
    assert model_class.__name__ == "Gemma3ForCausalLM"


def test_text_only_helper_rejects_configs_without_text_submodel():
    helper = _load_text_only_helper()

    class VisionOnlyConfig:
        vision_config = object()

    with pytest.raises(ValueError, match = "Cannot load vision-only as text-only"):
        helper(VisionOnlyConfig(), "vision-only")


def test_fast_language_model_forwards_text_only_to_fast_model():
    source = _source(LOADER_PATH)
    method = _class_method(ast.parse(source), "FastLanguageModel", "from_pretrained")

    # text_only defaults False (opt-in); both FastModel delegations forward it.
    text_only_default = _param_default(method, "text_only")
    assert isinstance(text_only_default, ast.Constant) and text_only_default.value is False

    fast_model_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "FastModel"
    ]
    assert len(fast_model_calls) == 2
    for call in fast_model_calls:
        kw = [k for k in call.keywords if k.arg == "text_only"]
        assert len(kw) == 1
        assert isinstance(kw[0].value, ast.Name) and kw[0].value.id == "text_only"


def test_fast_model_text_only_does_not_override_explicit_auto_model():
    # AST-based so formatting/refactors that keep the structure do not break it.
    source = _source(LOADER_PATH)
    method = _class_method(ast.parse(source), "FastModel", "from_pretrained")

    text_only_default = _param_default(method, "text_only")
    assert isinstance(text_only_default, ast.Constant) and text_only_default.value is False

    # load_text_only is text_only AND a check that the caller did not pass auto_model.
    def _is_guarded_bool(value):
        names = _names_in(value)
        has_none_check = any(
            isinstance(n, ast.Compare) and any(isinstance(op, (ast.Is, ast.IsNot)) for op in n.ops)
            for n in ast.walk(value)
        )
        return "text_only" in names and "auto_model" in names and has_none_check

    assert _assigns_name(method, "load_text_only", _is_guarded_bool)

    assert _calls_function(method, "_get_text_only_config")

    def _forwards_kwarg(node):
        return any(
            isinstance(n, ast.Call)
            and any(
                kw.arg == "text_only"
                and isinstance(kw.value, ast.Name)
                and kw.value.id == "load_text_only"
                for kw in n.keywords
            )
            for n in ast.walk(node)
        )

    assert _forwards_kwarg(method)
    # Falls back to the full model unless the family has its own text decoder.
    assert _calls_function(method, "_is_family_text_decoder")
    assert _assigns_name(
        method,
        "load_text_only",
        lambda v: isinstance(v, ast.Constant) and v.value is False,
    )


def test_fast_base_model_text_only_bypasses_vision_auto_model():
    source = _source(VISION_PATH)
    method = _class_method(ast.parse(source), "FastBaseModel", "from_pretrained")

    text_only_default = _param_default(method, "text_only")
    assert isinstance(text_only_default, ast.Constant) and text_only_default.value is False

    assert _assigns_name(
        method,
        "auto_model",
        lambda v: isinstance(v, ast.Name) and v.id == "AutoModelForCausalLM",
    )
    # Text-only path: strip config, apply the family guard, inject the key remap.
    assert _calls_function(method, "_get_text_only_config")
    assert _calls_function(method, "_is_family_text_decoder")
    assert _calls_function(method, "_apply_text_only_key_mapping")


def test_gemma3_text_only_model_class_resolves_and_has_no_vision_tower():
    """End-to-end: a tiny Gemma3 text-only model instantiates with text LM attrs and no vision tower."""
    transformers = pytest.importorskip("transformers")
    helper = _load_text_only_helper()

    full_config = transformers.Gemma3Config()
    text_config = helper(full_config, "google/gemma-3-27b-it")

    # Shrink for cheap CPU instantiation.
    text_config.num_hidden_layers = 1
    text_config.hidden_size = 32
    text_config.intermediate_size = 32
    text_config.num_attention_heads = 2
    text_config.num_key_value_heads = 1
    text_config.head_dim = 16
    text_config.vocab_size = 128

    model_class = transformers.AutoModelForCausalLM._model_mapping[type(text_config)]
    model = model_class(text_config)

    assert hasattr(model, "lm_head"), "text-only Gemma3 model should expose lm_head"

    # No vision tower / multimodal projector remains.
    assert not hasattr(
        model, "vision_tower"
    ), "text-only Gemma3 model should not have a vision_tower"
    assert not hasattr(
        model, "multi_modal_projector"
    ), "text-only Gemma3 model should not have a multi_modal_projector"


def test_helper_defined_once_in_utils_and_imported():
    # _get_text_only_config defined only in _utils, imported by loader + vision.
    def _defines(path):
        return any(
            isinstance(n, ast.FunctionDef) and n.name == "_get_text_only_config"
            for n in ast.parse(_source(path)).body
        )

    def _imports(path):
        return any(
            isinstance(n, ast.ImportFrom)
            and n.module == "_utils"
            and any(a.name == "_get_text_only_config" for a in n.names)
            for n in ast.walk(ast.parse(_source(path)))
        )

    assert _defines(UTILS_PATH)
    assert not _defines(LOADER_PATH) and _imports(LOADER_PATH)
    assert not _defines(VISION_PATH) and _imports(VISION_PATH)


def _load_util_func(name):
    ns = _load_text_only_namespace()
    if name not in ns:
        raise AssertionError(f"{name} not found")
    return ns[name]


def test_text_only_guard_predicate_across_vlm_families():
    # Text-only taken only when the resolved class remaps VLM weights.
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    resolve = _load_util_func("resolve_model_class")
    is_family = _load_util_func("_is_family_text_decoder")
    helper = _load_text_only_helper()

    def takes_text_only(cfg):
        text = helper(cfg, "x")
        return resolve(AutoModelForCausalLM, text) is not None and is_family(
            getattr(cfg, "model_type", ""), getattr(text, "model_type", "")
        )

    # Dedicated text decoder remaps language_model.* -> strip vision.
    assert takes_text_only(transformers.Gemma3Config()) is True

    # No text class (Qwen2-VL/Mllama) or a generic reused decoder that would
    # load random weights (Llava/PaliGemma/Idefics3/InternVL) -> keep full model.
    for name in [
        "Qwen2VLConfig",
        "Qwen2_5_VLConfig",
        "MllamaConfig",
        "LlavaConfig",
        "PaliGemmaConfig",
        "Idefics3Config",
        "InternVLConfig",
    ]:
        cfg_cls = getattr(transformers, name, None)
        if cfg_cls is None:
            continue
        assert takes_text_only(cfg_cls()) is False, name


def test_text_only_helper_preserves_quantization_config():
    # quantization_config must survive the strip so pre-quantized repos load. A
    # sentinel object avoids a bitsandbytes dependency on transformers 4.51.3.
    transformers = pytest.importorskip("transformers")
    helper = _load_text_only_helper()
    config = transformers.Gemma3Config()
    sentinel = object()
    config.quantization_config = sentinel
    text_config = helper(config, "google/gemma-3-27b-it")
    assert getattr(text_config, "quantization_config", None) is sentinel
    # The parent's shared text sub-config must not be mutated.
    assert getattr(config.get_text_config(), "quantization_config", None) is None


def test_text_only_key_mapping_targets_published_prefixes():
    # Remap the published VLM decoder prefixes, applying only on transformers >=5
    # (on 4.x base_model_prefix handles it and a mapping hurts).
    transformers = pytest.importorskip("transformers")
    get_key_mapping = _load_util_func("_get_text_only_key_mapping")
    mapping = get_key_mapping(transformers.Gemma3Config(), transformers.Gemma3TextConfig())
    if int(transformers.__version__.split(".")[0]) < 5:
        assert mapping is None
    else:
        assert isinstance(mapping, dict)
        assert mapping.get(r"^language_model\.model\.") == "model."  # gemma3
        assert mapping.get(r"^model\.language_model\.") == "model."  # gemma3n
        assert mapping.get(r"^language_model\.lm_head\.") == "lm_head."


def test_gemma3_text_only_loads_real_language_weights_from_vlm_checkpoint(tmp_path):
    # PR #5816: text-only loading of a Gemma 3 VLM checkpoint must load real
    # language weights, not random ones. Fails on tf >=5 without the key_mapping fix.
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    import shutil
    from safetensors.torch import load_file, save_file

    get_text_config = _load_text_only_helper()
    get_key_mapping = _load_util_func("_get_text_only_key_mapping")

    sentinel = 0.1234
    text_cfg = transformers.Gemma3TextConfig(
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        vocab_size = 128,
        max_position_embeddings = 128,
        sliding_window = 64,
    )
    vision_cfg = transformers.SiglipVisionConfig(
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        image_size = 16,
        patch_size = 8,
        num_channels = 3,
    )
    full_config = transformers.Gemma3Config(
        text_config = text_cfg.to_dict(),
        vision_config = vision_cfg.to_dict(),
    )
    full_model = transformers.Gemma3ForConditionalGeneration(full_config)

    state = full_model.state_dict()
    text_q = [
        k
        for k in state
        if "language_model" in k
        and "vision" not in k
        and k.endswith("layers.0.self_attn.q_proj.weight")
    ]
    assert text_q, [k for k in state if "q_proj" in k][:5]
    with torch.no_grad():
        for k in text_q:
            state[k].fill_(sentinel)

    save_dir = tmp_path / "vlm"
    full_model.save_pretrained(save_dir, safe_serialization = True)

    # tf >=5 saves under an outer "model." prefix; strip it to reproduce the
    # language_model.model.* layout the published Gemma 3 checkpoints use.
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    weights = {}
    for f in save_dir.glob("*.safetensors"):
        weights.update(load_file(str(f)))
    for f in save_dir.glob("*.bin"):
        weights.update(torch.load(f, map_location = "cpu", weights_only = True))
    weights = {
        (k[len("model.") :] if k.startswith("model.") else k): v.contiguous()
        for k, v in weights.items()
    }
    for p in save_dir.iterdir():
        if not p.name.endswith((".safetensors", ".bin", ".index.json")):
            shutil.copy(p, real_dir / p.name)
    save_file(weights, str(real_dir / "model.safetensors"))

    text_config = get_text_config(full_config, "google/gemma-3-27b-it")
    load_kwargs = {}
    key_mapping = get_key_mapping(full_config, text_config)
    if key_mapping is not None:
        load_kwargs["key_mapping"] = key_mapping
    model = transformers.AutoModelForCausalLM.from_pretrained(
        real_dir,
        config = text_config,
        dtype = torch.float32,
        local_files_only = True,
        **load_kwargs,
    )

    loaded = model.state_dict()
    q_key = [k for k in loaded if k.endswith("model.layers.0.self_attn.q_proj.weight")]
    assert q_key, "text decoder q_proj weight missing from the loaded model"
    assert float(loaded[q_key[0]].flatten()[0]) == pytest.approx(
        sentinel
    ), "text weights were randomly initialized instead of loaded from the checkpoint"
    assert not any(
        "vision_tower" in n for n, _ in model.named_modules()
    ), "vision tower should be skipped on the text-only path"
