# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A date-only system turn replaces Ollama's Modelfile SYSTEM (#10436).

``_prepend_current_date_to_messages`` used to invent ``{role: system, content: date}``
when the payload had no system turn. The chat adapter only unshifts a system turn when
``combinedSystemPrompt`` is non-empty, so Settings -> Chat -> Tell the model today's date
was enough to wipe the model SYSTEM on Ollama's /v1 path. The date goes on the first user
turn instead for ``provider_type == "ollama"``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "studio" / "backend"
INFERENCE = REPO / "studio" / "backend" / "routes" / "inference.py"
DATE_LINE = "The current date is 2026-08-15."

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _inference_src() -> str:
    return INFERENCE.read_text(encoding = "utf-8")


def _date_settings():
    from utils import current_date_prompt_settings as settings

    return settings


def test_only_ollama_preserves_the_server_system():
    date_prompt_preserves_server_system = getattr(
        _date_settings(), "date_prompt_preserves_server_system", None
    )
    assert date_prompt_preserves_server_system is not None
    assert date_prompt_preserves_server_system("ollama") is True
    for provider in (
        "openai",
        "anthropic",
        "vllm",
        "llama_cpp",
        "custom",
        "openai_codex",
        None,
        "",
    ):
        assert date_prompt_preserves_server_system(provider) is False


def _attach():
    fn = getattr(_date_settings(), "attach_date_line_to_first_user_message", None)
    assert fn is not None
    return fn


def test_date_lands_on_the_user_turn_without_a_system_role():
    payload = [{"role": "assistant", "content": "ready"}, {"role": "user", "content": "hi"}]
    out = _attach()(payload, DATE_LINE)
    assert [msg["role"] for msg in out] == ["assistant", "user"]
    assert out[0]["content"] == "ready"
    assert out[1]["content"] == f"{DATE_LINE}\n\nhi"


def test_structured_user_text_is_prefixed_the_same_way():
    payload = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png,x"}},
                {"type": "text", "text": "what is this"},
            ],
        }
    ]
    out = _attach()(payload, DATE_LINE)
    assert out[0]["role"] == "user"
    assert out[0]["content"][0] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png,x"},
    }
    assert out[0]["content"][1]["text"] == f"{DATE_LINE}\n\nwhat is this"


def test_a_user_turn_with_no_text_part_gets_one():
    payload = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png,x"}}],
        }
    ]
    out = _attach()(payload, DATE_LINE)
    assert out[0]["content"][0] == {"type": "text", "text": DATE_LINE}
    assert out[0]["content"][1]["type"] == "image_url"


def test_no_user_turn_does_not_invent_a_system_turn():
    payload = [{"role": "assistant", "content": "ready"}]
    out = _attach()(payload, DATE_LINE)
    assert out == [{"role": "assistant", "content": "ready"}]


def test_external_proxy_passes_the_ollama_preserve_flag():
    src = _inference_src()
    marker = "run_studio_tool_loop = bool(external_studio_tools)"
    window = src[src.index(marker) : src.index(marker) + 600]
    assert (
        "preserve_server_system = date_prompt_preserves_server_system(provider_type)"
        in window
    )


def test_prepend_fallback_attaches_to_the_user_when_preserving():
    src = _inference_src()
    fallback = src[src.index("def _prepend_current_date_to_messages(") :]
    fallback = fallback[: fallback.index("\n\n# Strip leaked tool-call markup")]
    assert "if preserve_server_system:" in fallback
    assert "attach_date_line_to_first_user_message(copied, date_line)" in fallback
    assert 'return [{"role": "system", "content": date_line}, *copied]' in fallback
