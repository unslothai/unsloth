# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for trust_remote_code in the native-template fallback.

``render_native_template`` re-fetches a model's native chat template from its
repo when an Unsloth override template (mistral, gemma-4) dropped the tools
schema. For a model loaded with ``trust_remote_code=True`` whose tokenizer repo
carries custom code, the secondary ``AutoTokenizer.from_pretrained`` must re-use
that same consent or transformers raises (it requires ``trust_remote_code`` to
instantiate a custom tokenizer class), the ``except`` swallows it, and the
request silently keeps the tool-dropping prompt even though the user already
consented to remote code for the model load.

These tests pin that the stored ``trust_remote_code`` is threaded to the reload,
that the reload is skipped (returns ``None`` without executing code) when no
consent is stored, and that both backend ``model_info`` dicts persist the flag at
load time so the read lands on a value ``load_model`` actually set.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# ``chat_template_helpers`` is dependency-light (copy / logging / typing, with the
# transformers import deferred inside the function). Load it directly so the test
# runs without importing the heavy ``core.inference`` package (unsloth / torch).
_HELPERS_PATH = Path(_BACKEND_DIR) / "core" / "inference" / "chat_template_helpers.py"
_spec = importlib.util.spec_from_file_location("_native_tpl_trc_test", _HELPERS_PATH)
chat_template_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chat_template_helpers)

render_native_template = chat_template_helpers.render_native_template


# A native template that emits a tools section only when tools are provided, so the
# with-tools vs no-tools render differs and ``render_native_template`` accepts it.
_NATIVE_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if tools %}[AVAILABLE_TOOLS]{{ tools }}[/AVAILABLE_TOOLS]\n{% endif %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)

_MESSAGES = [{"role": "user", "content": "what is the weather"}]
_TOOLS = [{"type": "function", "function": {"name": "get_weather"}}]


class _JinjaTokenizer:
    """Minimal tokenizer whose ``apply_chat_template`` renders ``self.chat_template``.

    Stands in for the live model tokenizer that ``render_native_template`` shallow-
    copies and re-points at the native template before rendering.
    """

    def __init__(self, chat_template):
        self.chat_template = chat_template

    def apply_chat_template(
        self,
        messages,
        tokenize = False,
        add_generation_prompt = True,
        tools = None,
        **kwargs,
    ):
        from jinja2 import BaseLoader, Environment
        env = Environment(loader = BaseLoader())
        return env.from_string(self.chat_template).render(
            messages = messages,
            tools = tools,
            add_generation_prompt = add_generation_prompt,
        )


def _install_custom_code_tokenizer(monkeypatch):
    """Patch ``AutoTokenizer.from_pretrained`` to mimic a custom-code repo: raise
    unless ``trust_remote_code`` is truthy, else return a tokenizer carrying the
    native template. Records the ``trust_remote_code`` it was called with."""
    pytest.importorskip("jinja2")
    from transformers import AutoTokenizer

    calls = {}

    def fake_from_pretrained(
        model_id,
        *args,
        trust_remote_code = False,
        token = None,
        **kwargs,
    ):
        calls["trust_remote_code"] = trust_remote_code
        calls["model_id"] = model_id
        calls["token"] = token
        if not trust_remote_code:
            # Mirrors transformers.dynamic_module_utils.resolve_trust_remote_code:
            # has_remote_code and not has_local_code and not trust_remote_code -> ValueError.
            raise ValueError(
                f"The repository {model_id} contains custom code which must be executed "
                "to correctly load the model. Please pass the argument "
                "`trust_remote_code=True` to allow custom code to be run."
            )
        return _JinjaTokenizer(_NATIVE_TEMPLATE)

    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", staticmethod(fake_from_pretrained)
    )
    return calls


def _model_info(trust_remote_code):
    return {
        "native_chat_template": None,  # force the repo reload path
        "base_model": None,  # non-LoRA: template_source == active_model_name
        "trust_remote_code": trust_remote_code,
        # Live tokenizer that gets shallow-copied + re-pointed at the native template.
        "tokenizer": _JinjaTokenizer("OVERRIDE-THAT-DROPS-TOOLS"),
    }


def test_native_reload_passes_stored_trust_remote_code(monkeypatch):
    """With ``trust_remote_code`` stored on ``model_info`` the custom-code reload
    succeeds and the tools-advertising native prompt is returned. This FAILS before
    the fix (reload omits the flag, raises, is swallowed, returns None)."""
    calls = _install_custom_code_tokenizer(monkeypatch)
    model_info = _model_info(trust_remote_code = True)

    out = render_native_template(
        model_info = model_info,
        active_model_name = "acme/custom-tokenizer-model",
        messages = _MESSAGES,
        tools = _TOOLS,
    )

    assert (
        out is not None
    ), "native fallback should render the tools prompt with consent"
    assert "[AVAILABLE_TOOLS]" in out
    assert "get_weather" in out
    assert calls["trust_remote_code"] is True  # the stored consent was threaded through
    # A successful fetch is cached so the next tool turn skips the reload.
    assert model_info["native_chat_template"] == _NATIVE_TEMPLATE


def test_native_reload_without_consent_returns_none(monkeypatch):
    """Without stored consent the custom-code reload raises, is swallowed, and
    ``render_native_template`` returns None (no unconsented code execution). Proves
    the stored flag -- not a hard-coded True -- drives the reload."""
    calls = _install_custom_code_tokenizer(monkeypatch)
    model_info = _model_info(trust_remote_code = False)

    out = render_native_template(
        model_info = model_info,
        active_model_name = "acme/custom-tokenizer-model",
        messages = _MESSAGES,
        tools = _TOOLS,
    )

    assert out is None
    assert calls["trust_remote_code"] is False
    # A failed fetch must not be cached as "no template" (would pin the tool drop).
    assert model_info["native_chat_template"] is None


def test_backend_model_info_persists_trust_remote_code():
    """Both backends must store ``trust_remote_code`` on their per-model info dict so
    ``render_native_template`` can source the consent value. Guards against the read
    landing on a key ``load_model`` never sets (which would silently no-op the fix)."""
    inf = (Path(_BACKEND_DIR) / "core" / "inference" / "inference.py").read_text()
    mlx = (Path(_BACKEND_DIR) / "core" / "inference" / "mlx_inference.py").read_text()
    assert '"trust_remote_code": trust_remote_code,' in inf
    assert '"trust_remote_code": trust_remote_code,' in mlx
