from types import SimpleNamespace

import unsloth  # noqa: F401
from transformers.utils import import_utils

from unsloth.models import _utils


class SupportsFlexAndSdpa:
    _supports_flash_attn_2 = True
    _supports_flex_attn = True
    _supports_sdpa = True


def _config(model_type, **kwargs):
    values = {"model_type": model_type, "attention_dropout": 0}
    values.update(kwargs)
    return SimpleNamespace(**values)


def _set_flex_available(monkeypatch, available):
    monkeypatch.setenv("UNSLOTH_ENABLE_FLEX_ATTENTION", "1")
    monkeypatch.setattr(
        import_utils,
        "is_torch_flex_attn_available",
        lambda: available,
        raising = False,
    )


def _auto_model(name = "AutoModelForCausalLM"):
    return SimpleNamespace(__name__ = name)


def _routes(*auto_classes):
    return {name: f"modeling_custom.Custom{name}" for name in auto_classes}


def test_remote_code_class_is_detected_whatever_the_config_class_module(monkeypatch):
    # unsloth#7527. A repo may route only the model through auto_map, in which case AutoConfig
    # hands back the *built-in* config class while transformers still builds the repo's model, so
    # the config class cannot be used to tell whether remote code wins.
    config = _config("nemotron_h", auto_map = _routes("AutoModelForCausalLM"))

    assert _utils.builds_remote_code_class(_auto_model(), config, True)


def test_remote_code_class_needs_remote_code_to_be_trusted(monkeypatch):
    config = _config("nemotron_h", auto_map = _routes("AutoConfig", "AutoModelForCausalLM"))

    assert not _utils.builds_remote_code_class(_auto_model(), config, False)


def test_remote_code_class_is_matched_per_auto_class(monkeypatch):
    # auto_factory keys has_remote_code on the auto class name, so a route for another task
    # leaves the built-in class in place and its flags are the correct ones to read.
    config = _config("nemotron_h", auto_map = _routes("AutoConfig", "AutoModelForCausalLM"))

    assert not _utils.builds_remote_code_class(
        _auto_model("AutoModelForImageTextToText"), config, True
    )


def test_config_without_an_auto_map_never_builds_remote_code(monkeypatch):
    assert not _utils.builds_remote_code_class(_auto_model(), _config("llama"), True)


class SupportsSdpaOnly:
    _supports_sdpa = True


class _FakeImport:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, class_reference, model_name, **kwargs):
        self.calls.append((class_reference, model_name, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _patch_dynamic_import(monkeypatch, result):
    from transformers import dynamic_module_utils

    fake = _FakeImport(result)
    monkeypatch.setattr(dynamic_module_utils, "get_class_from_dynamic_module", fake)
    return fake


def test_remote_code_class_is_imported_from_the_route_in_the_config(monkeypatch):
    config = _config("nemotron_h", auto_map = _routes("AutoModelForCausalLM"))
    fake = _patch_dynamic_import(monkeypatch, SupportsSdpaOnly)

    loaded = _utils.load_remote_code_class(
        _auto_model(),
        config,
        "acme/model",
        revision = "abc",
        code_revision = "def",
    )

    assert loaded is SupportsSdpaOnly
    class_reference, model_name, kwargs = fake.calls[0]
    assert class_reference == "modeling_custom.CustomAutoModelForCausalLM"
    assert model_name == "acme/model"
    assert kwargs["revision"] == "abc"
    # the module comes from code_revision, so the flags have to be read from there too
    assert kwargs["code_revision"] == "def"


def test_remote_code_class_is_none_when_the_import_fails(monkeypatch):
    config = _config("nemotron_h", auto_map = _routes("AutoModelForCausalLM"))
    _patch_dynamic_import(monkeypatch, ImportError("missing dependency"))

    assert _utils.load_remote_code_class(_auto_model(), config, "acme/model") is None


def test_remote_class_that_declares_sdpa_keeps_it(monkeypatch):
    # The repo's own class decides: one that declares support is not downgraded just for being
    # remote code.
    config = _config("nemotron_h")

    impl = _utils.resolve_attention_implementation(
        SupportsSdpaOnly,
        config,
        supports_sdpa = True,
    )

    assert impl == "sdpa"


def test_class_that_cannot_be_inspected_resolves_to_eager(monkeypatch):
    # What the loader passes once builds_remote_code_class is true: repo code need not declare
    # support for anything, and transformers raises rather than falling back.
    config = _config("nemotron_h")

    impl = _utils.resolve_attention_implementation(None, config, supports_sdpa = False)

    assert impl == "eager"
    assert config._attn_implementation == "eager"


def test_gpt_oss_uses_eager_instead_of_flash_flex_or_sdpa(monkeypatch):
    _set_flex_available(monkeypatch, True)
    config = _config("gpt_oss")

    impl = _utils.resolve_attention_implementation(
        SupportsFlexAndSdpa,
        config,
        supports_sdpa = True,
    )

    assert impl == "eager"
    assert config._attn_implementation == "eager"


def test_gpt_oss_falls_back_to_eager_when_flex_unavailable(monkeypatch):
    _set_flex_available(monkeypatch, False)
    config = _config("gpt_oss")

    impl = _utils.resolve_attention_implementation(
        SupportsFlexAndSdpa,
        config,
        supports_sdpa = True,
    )

    assert impl == "eager"
    assert config._attn_implementation == "eager"
