# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The lazy native-template reload must run only the custom code reviewed at load time.

It runs on a chat request long after the load, with the load's trust_remote_code. Without a
pin, transformers resolves the branch's current tip and imports whatever code is there now.
"""

import sys
import types

import pytest

from core.inference.chat_template_helpers import resolve_native_chat_template

SHA = "a" * 40


@pytest.fixture
def captured(monkeypatch):
    calls = []

    class _Tok:
        chat_template = "{{ messages }}"

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append((source, kwargs))
            return _Tok()

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return calls


def test_trusted_reload_is_pinned_to_the_reviewed_commit(captured):
    info = {"trust_remote_code": True, "remote_code_revisions": {"org/custom": SHA}}
    assert resolve_native_chat_template(info, "org/custom") == "{{ messages }}"
    ((source, kwargs),) = captured
    assert source == "org/custom"
    assert kwargs.get("revision") == SHA
    assert kwargs.get("trust_remote_code") is True


def test_lora_base_reload_uses_the_base_commit(captured):
    info = {
        "trust_remote_code": True,
        "base_model": "org/base",
        "remote_code_revisions": {"org/adapter": "b" * 40, "org/base": SHA},
    }
    resolve_native_chat_template(info, "org/adapter")
    ((source, kwargs),) = captured
    assert source == "org/base" and kwargs.get("revision") == SHA


def test_trusted_reload_without_a_known_commit_never_fetches_new_code(captured):
    resolve_native_chat_template({"trust_remote_code": True}, "org/custom")
    ((_, kwargs),) = captured
    assert kwargs.get("revision") is not None or kwargs.get("local_files_only") is True


def test_untrusted_reload_is_unchanged(captured):
    resolve_native_chat_template({"trust_remote_code": False}, "org/plain")
    ((_, kwargs),) = captured
    assert "revision" not in kwargs and "local_files_only" not in kwargs
    assert kwargs.get("trust_remote_code") is False


def test_local_path_reload_is_unchanged(captured, tmp_path):
    resolve_native_chat_template({"trust_remote_code": True}, str(tmp_path))
    ((_, kwargs),) = captured
    assert "revision" not in kwargs and "local_files_only" not in kwargs


def test_reviewed_commit_is_read_from_the_cached_snapshot(monkeypatch, tmp_path):
    from core.inference import chat_template_helpers as helpers
    import utils.utils as uu

    repo = tmp_path / "models--org--custom"
    (repo / "snapshots" / SHA).mkdir(parents = True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(SHA)
    monkeypatch.setattr(uu, "_hf_cache_roots", lambda: [tmp_path])
    reviewed = getattr(helpers, "reviewed_code_revision", None)
    assert reviewed is not None
    assert reviewed("org/custom") == SHA
    assert reviewed("org/missing") is None
    assert reviewed(str(tmp_path)) is None
