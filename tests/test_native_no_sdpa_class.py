# SPDX-License-Identifier: AGPL-3.0-only
"""A transformers class whose _supports_sdpa resolves False (Mamba, FalconMamba) resolves to eager even when
the zoo's source probe reports sdpa: transformers raises "does not support ... scaled_dot_product_attention"."""

from types import SimpleNamespace

import pytest
import unsloth  # noqa: F401

from unsloth.models import _utils


@pytest.fixture(autouse = True)
def _no_flash(monkeypatch):
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", False, raising = False)


@pytest.mark.parametrize(
    "module_name, class_name, model_type",
    [
        ("transformers.models.mamba.modeling_mamba", "MambaForCausalLM", "mamba"),
        (
            "transformers.models.falcon_mamba.modeling_falcon_mamba",
            "FalconMambaForCausalLM",
            "falcon_mamba",
        ),
    ],
)
def test_real_mamba_classes_resolve_to_eager(module_name, class_name, model_type):
    cls = getattr(pytest.importorskip(module_name), class_name)
    if cls._supports_sdpa:
        pytest.skip(reason = f"this transformers gives {class_name} an sdpa path")
    config = SimpleNamespace(model_type = model_type)
    # supports_sdpa = True is what unsloth_compile_transformers reports for these classes.
    impl = _utils.resolve_attention_implementation(cls, config, supports_sdpa = True)
    assert impl == "eager"
    # The resolved value is what the loader stamps on the config and passes to from_pretrained.
    assert config._attn_implementation == "eager"


def test_native_class_with_sdpa_keeps_sdpa():
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    assert LlamaForCausalLM._supports_sdpa
    config = SimpleNamespace(model_type = "llama", attention_dropout = 0)
    assert (
        _utils.resolve_attention_implementation(LlamaForCausalLM, config, supports_sdpa = True)
        == "sdpa"
    )


def test_only_transformers_classes_are_checked():
    # Remote code that does not declare _supports_sdpa keeps the source-probe answer.
    remote = type("RemoteForCausalLM", (), {"__module__": "transformers_modules.repo.modeling"})
    remote._supports_sdpa = False
    assert not _utils._native_class_refuses_sdpa(remote)
    native = type("NativeForCausalLM", (), {"__module__": "transformers.models.x.modeling_x"})
    native._supports_sdpa = False
    assert _utils._native_class_refuses_sdpa(native)
    native._supports_sdpa = True
    assert not _utils._native_class_refuses_sdpa(native)
    assert not _utils._native_class_refuses_sdpa(None)
