"""Regression tests for the export-time remote-code trust decision.

FP8/FP4/INT quantization export re-reads the just-merged checkpoint. It used to enable
trust_remote_code whenever the checkpoint's config carried an ``auto_map`` entry, so a model
that loads fine with built-in classes (and therefore skips the load-time consent scan) could
smuggle unvetted remote code that then runs at export. The export paths now derive
trust_remote_code from ``_loaded_via_remote_code`` - the already approved load decision - instead.

These run on CPU with no torch / unsloth import: they AST-extract the real helper from
unsloth/save.py and exec it in isolation, plus assert the call sites dropped the auto_map trust.
"""

import ast
from pathlib import Path

_SAVE_PY = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"
_SRC = _SAVE_PY.read_text(encoding = "utf-8")


def _load_helper():
    """Exec just `_loaded_via_remote_code` from save.py (no torch import) and return it."""
    tree = ast.parse(_SRC)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_loaded_via_remote_code"
    )
    ns = {}
    exec(compile(ast.Module(body = [fn], type_ignores = []), str(_SAVE_PY), "exec"), ns)
    return ns["_loaded_via_remote_code"]


_loaded_via_remote_code = _load_helper()


def _obj(module_name, **attrs):
    """A throwaway instance whose class __module__ is `module_name`, plus given attributes."""
    cls = type("Fake", (), {})
    cls.__module__ = module_name
    inst = cls()
    for k, v in attrs.items():
        setattr(inst, k, v)
    return inst


def test_builtin_class_is_not_remote_code():
    assert (
        _loaded_via_remote_code(_obj("transformers.models.llama.modeling_llama"))
        is False
    )


def test_transformers_modules_class_is_remote_code():
    assert _loaded_via_remote_code(_obj("transformers_modules.acme.modeling_x")) is True


def test_none_is_not_remote_code():
    assert _loaded_via_remote_code(None) is False


def test_none_module_is_not_remote_code():
    # A class whose __module__ is None must not raise AttributeError.
    assert _loaded_via_remote_code(_obj(None)) is False


def test_auto_map_in_config_alone_does_not_grant_trust():
    # The core bypass: a built-in-loadable model whose config merely declares auto_map must NOT
    # be treated as remote-code-loaded (that is exactly what enabled the consent-gate bypass).
    cfg = type("Cfg", (), {"auto_map": {"AutoModelForCausalLM": "modeling_x.Model"}})()
    assert (
        _loaded_via_remote_code(
            _obj("transformers.models.llama.modeling_llama", config = cfg)
        )
        is False
    )


def test_peft_base_model_is_unwrapped():
    base = _obj("transformers_modules.acme.modeling_x")
    peft = _obj("peft.peft_model", get_base_model = lambda: base)
    assert _loaded_via_remote_code(peft) is True


def test_wrapper_model_attr_is_walked():
    inner = _obj("transformers_modules.acme.modeling_x")
    wrapper = _obj("peft.peft_model", model = inner)
    assert _loaded_via_remote_code(wrapper) is True


def test_wrapper_over_builtin_stays_false():
    inner = _obj("transformers.models.llama.modeling_llama")
    wrapper = _obj("peft.peft_model", model = inner)
    assert _loaded_via_remote_code(wrapper) is False


def test_processor_held_custom_tokenizer_is_detected():
    # A built-in ProcessorMixin can hold an approved custom-code tokenizer; the walk must
    # descend into processor components or the export reload loses that approved trust.
    tok = _obj("transformers_modules.acme.tokenization_x")
    proc = _obj("transformers.processing_utils", tokenizer = tok)
    assert _loaded_via_remote_code(proc) is True


def test_processor_held_custom_image_processor_is_detected():
    ip = _obj("transformers_modules.acme.image_processing_x")
    proc = _obj("transformers.processing_utils", image_processor = ip)
    assert _loaded_via_remote_code(proc) is True


def test_builtin_processor_with_builtin_components_stays_false():
    proc = _obj(
        "transformers.processing_utils",
        tokenizer = _obj("transformers.tokenization_utils_fast"),
        image_processor = _obj("transformers.image_processing_utils"),
    )
    assert _loaded_via_remote_code(proc) is False


def test_cyclic_wrappers_terminate():
    a = _obj("peft.peft_model")
    b = _obj("peft.peft_model", model = a)
    a.model = b
    assert _loaded_via_remote_code(a) is False


# -- call-site assertions: the auto_map-derived trust is gone from every export path -----------


def test_torchao_export_derives_trust_from_load_decision():
    assert "model_trust = _loaded_via_remote_code(model)" in _SRC
    assert "tok_trust = _loaded_via_remote_code(tokenizer)" in _SRC
    assert "trust_remote_code = model_trust" in _SRC
    assert "trust_remote_code = tok_trust" in _SRC
    # The staged-config auto_map scan that granted trust is removed.
    assert 'if "auto_map" in json.load' not in _SRC


def test_compressed_and_gguf_lora_paths_drop_auto_map_trust():
    # No path derives a trust decision straight from config auto_map anymore, and no path
    # collapses model and tokenizer trust into one flag.
    assert 'bool(getattr(model.config, "auto_map", None))' not in _SRC
    assert (
        "_loaded_via_remote_code(model) or _loaded_via_remote_code(tokenizer)"
        not in _SRC
    )
    assert "if _loaded_via_remote_code(model):" in _SRC  # GGUF-LoRA converter flag


def test_compressed_export_keeps_model_and_tokenizer_trust_separate():
    # The subprocess gets one flag per component, so an approved custom tokenizer cannot
    # enable an unapproved model's code during compressed quantization (or vice versa).
    assert 'cmd.append("--trust-remote-code")' in _SRC
    assert 'cmd.append("--trust-remote-code-tokenizer")' in _SRC
    qsrc = (_SAVE_PY.parent / "_compressed_quantize.py").read_text(encoding = "utf-8")
    assert (
        'ap.add_argument("--trust-remote-code-tokenizer", action = "store_true")'
        in qsrc
    )
    assert "trust_remote_code = args.trust_remote_code_tokenizer" in qsrc
    # The model loads keep the model flag only.
    assert "args.model, args.trust_remote_code)" in qsrc
