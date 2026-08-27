# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ``unsloth_<model>_<timestamp>`` directory-name fallback for base model detection.

The last resort in ``get_base_model_from_checkpoint`` / ``get_base_model_from_lora`` and in the
two resolvers in ``utils.transformers_version``. It used to slice a two-segment
``unsloth_<model>`` name down to nothing and return the bogus repo id ``unsloth/``.

The round-trip tests are the real specification: a name we write must parse back to the model
it was written from.
"""

import importlib.util
import json
import sys
import types

import pytest

# Keep this runnable where optional logging deps are absent. Probe the installed distribution,
# not sys.modules: structlog is a real dependency, and stubbing it merely because nothing has
# imported it yet would replace the package for every test collected afterwards.
if importlib.util.find_spec("structlog") is None:  # pragma: no cover - minimal environments

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

from utils.models.model_config import (  # noqa: E402
    get_base_model_from_checkpoint,
    get_base_model_from_lora,
)
from utils.training_runs import (  # noqa: E402
    base_model_from_run_dir_name,
    build_default_output_dir_name,
)


@pytest.mark.parametrize(
    "dir_name,expected",
    [
        # The shape the heuristic is written for.
        ("unsloth_Qwen3-8B_1771227800", "unsloth/Qwen3-8B"),
        # A hand-made or foreign folder may carry a date-time stamp instead of an epoch.
        ("unsloth_Qwen3-8B_20260101-120000", "unsloth/Qwen3-8B"),
        ("unsloth_Qwen3-8B_20260101", "unsloth/Qwen3-8B"),
        # Underscores inside the model name survive the round trip.
        ("unsloth_llama_3_8b_1771227800", "unsloth/llama_3_8b"),
        # No model segment between the prefix and the timestamp: the originally reported bug.
        ("unsloth_Qwen3-8B", None),
        ("unsloth_", None),
        ("unsloth", None),
        ("unsloth__1771227800", None),
        # A doubled separator means the name really does start with '_', which HF allows.
        # Stripping it would resolve a different, equally valid repo.
        ("unsloth__Qwen3-8B_1771227800", "unsloth/_Qwen3-8B"),
        # The project suffix is not part of the model name.
        ("unsloth_Qwen3-8B__project-demo_1771227800", "unsloth/Qwen3-8B"),
        # ...and a name containing the marker is escaped by the generator.
        ("unsloth_x__project--y_1771227800", "unsloth/x__project-y"),
        # A non-timestamp tail belongs to the model name, so the whole name is unparseable
        # rather than truncated. Truncating gave 'unsloth/llama_3' -- valid, but nonexistent.
        ("unsloth_llama_3_8b", None),
        ("unsloth_gpt_oss_20b", None),
        ("unsloth_Qwen3-8B_final", None),
        ("unsloth_Qwen3-8B_v2", None),
        ("unsloth_Qwen3-8B_checkpoint-500", None),
        # A model name that is itself all digits still round-trips.
        ("unsloth_20260101_1771227800", "unsloth/20260101"),
        # Not ours: leave it to the caller's "could not detect" path.
        ("my-finetune_1771227800", None),
        ("meta-llama_Llama-3.1-8B_1771227800", None),
        ("Unsloth_Qwen3-8B_1771227800", None),
        ("checkpoint-500", None),
        ("", None),
    ],
)
def test_base_model_from_run_dir_name(dir_name, expected):
    assert base_model_from_run_dir_name(dir_name) == expected


@pytest.mark.parametrize(
    "repo_id",
    [
        "unsloth/Qwen3-8B",
        "unsloth/llama_3_8b",
        "unsloth/gpt_oss_20b",
        "unsloth/_Qwen3-8B",
        "unsloth/x__project-y",
        "unsloth/20260101",
        "unsloth/Llama-3.2-3B-Instruct",
    ],
)
@pytest.mark.parametrize("project_name", [None, "Demo Project", "customer support"])
def test_a_generated_run_dir_name_parses_back_to_its_model(repo_id, project_name):
    """The parser is the generator read backwards; this is what stops the two drifting."""
    dir_name = build_default_output_dir_name(repo_id, project_name, timestamp = 1771227800)
    assert base_model_from_run_dir_name(dir_name) == repo_id


def test_the_bare_org_is_never_returned():
    """``unsloth/`` fails huggingface_hub's validate_repo_id, so it must never escape."""
    candidates = [
        "unsloth_",
        "unsloth__1771227800",
        "unsloth___1771227800",
        "unsloth_1771227800",
        "unsloth__",
        "unsloth____",
        "unsloth_ _1771227800",
    ]
    assert [n for n in candidates if base_model_from_run_dir_name(n) == "unsloth/"] == []


@pytest.mark.parametrize(
    "dir_name",
    [
        "unsloth_ _1771227800",  # space
        "unsloth_._1771227800",  # bare dot
        "unsloth_-_1771227800",  # a name may not start or end with '-'
        "unsloth_..._1771227800",  # '..' is rejected outright
        "unsloth_--_1771227800",  # so is '--'
        "unsloth_.git_1771227800",  # a repo id may not end with '.git'
        "unsloth_a\tb_1771227800",  # control characters
        "unsloth_\n_1771227800",
        "unsloth_🦥_1771227800",  # not a word character
        "unsloth_" + "a" * 97 + "_1771227800",  # the Hub caps a name at 96 characters
    ],
)
def test_a_folder_name_that_cannot_be_a_repo_id_is_refused(dir_name):
    """A folder name is user input. Only trust the parse if the Hub would accept the result.

    ``build_default_output_dir_name`` sanitises everything here, so these shapes only arise
    from a hand-made folder -- which is exactly the case the fallback exists to serve.
    """
    assert base_model_from_run_dir_name(dir_name) is None


def test_the_transcribed_repo_id_rule_is_never_looser_than_the_hubs():
    """The rule is transcribed to keep this module stdlib-only, so pin it to the real one."""
    hub_validate = pytest.importorskip("huggingface_hub.utils").validate_repo_id

    def hub_accepts(name):
        try:
            hub_validate(f"unsloth/{name}")
            return True
        except Exception:
            return False

    names = [
        "Qwen3-8B",
        "llama_3_8b",
        "_Qwen3-8B",
        "x__project-y",
        "20260101",
        "a",
        "a" * 96,
        "a" * 97,
        ".git",
        "a.git",
        "x--y",
        "x..y",
        "-x",
        "x-",
        ".x",
        "x.",
        "x_",
        "🦥",
        "",
        " ",
        "a b",
        "a\tb",
        "Café-8B",
        "a-b.c_d",
    ]
    looser = [
        n
        for n in names
        if base_model_from_run_dir_name(f"unsloth_{n}_1771227800") is not None
        and not hub_accepts(n)
    ]
    assert looser == []


def _write_adapter(directory):
    directory.mkdir(parents = True)
    # No base_model_name_or_path, so detection has to fall through to the directory name.
    (directory / "adapter_config.json").write_text(json.dumps({}), encoding = "utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"")


@pytest.mark.parametrize(
    "dir_name,expected",
    [
        ("unsloth_Qwen3-8B", None),
        ("unsloth_llama_3_8b", None),
        ("unsloth_Qwen3-8B_1771227800", "unsloth/Qwen3-8B"),
        ("unsloth_Qwen3-8B__project-demo_1771227800", "unsloth/Qwen3-8B"),
    ],
)
def test_lora_detection_end_to_end(tmp_path, dir_name, expected):
    adapter = tmp_path / dir_name
    _write_adapter(adapter)
    assert get_base_model_from_lora(str(adapter)) == expected


@pytest.mark.parametrize(
    "dir_name,expected",
    [
        ("unsloth_Qwen3-8B", None),
        ("unsloth_llama_3_8b", None),
        ("unsloth_Qwen3-8B_1771227800", "unsloth/Qwen3-8B"),
        ("unsloth_Qwen3-8B__project-demo_1771227800", "unsloth/Qwen3-8B"),
    ],
)
def test_checkpoint_detection_end_to_end(tmp_path, dir_name, expected):
    checkpoint = tmp_path / dir_name
    checkpoint.mkdir()
    assert get_base_model_from_checkpoint(str(checkpoint)) == expected


def test_a_named_base_model_still_wins_over_the_directory_name(tmp_path):
    """The fallback must stay a last resort: an explicit config is never second-guessed."""
    adapter = tmp_path / "unsloth_Qwen3-8B_1771227800"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "meta-llama/Llama-3.1-8B"}), encoding = "utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    assert get_base_model_from_lora(str(adapter)) == "meta-llama/Llama-3.1-8B"


def test_a_trailing_separator_does_not_change_the_answer(tmp_path):
    """Callers pass raw user input; ``Path.name`` must have already normalised it."""
    adapter = tmp_path / "unsloth_Qwen3-8B_1771227800"
    _write_adapter(adapter)
    assert get_base_model_from_lora(str(adapter) + "/") == "unsloth/Qwen3-8B"


# --- the two resolvers in utils.transformers_version ---------------------------------------
# _resolve_base_model is reached *through* get_base_model_from_lora, so while it kept its own
# copy of the parse it rebuilt the bogus id one branch after the fixed function returned None.


def test_the_transformers_resolvers_agree_with_the_model_config_one(tmp_path):
    from utils.transformers_version import _resolve_base_model, recorded_local_base
    for dir_name in ("unsloth_Qwen3-8B", "unsloth_llama_3_8b", "unsloth_Qwen3-8B_1771227800"):
        expected = base_model_from_run_dir_name(dir_name)

        weights_only = tmp_path / "weights" / dir_name
        weights_only.mkdir(parents = True)
        (weights_only / "adapter_model.safetensors").write_bytes(b"")
        assert recorded_local_base(str(weights_only)) == (expected, False)
        assert _resolve_base_model(str(weights_only)) == (expected or str(weights_only))

        with_config = tmp_path / "config" / dir_name
        _write_adapter(with_config)
        assert _resolve_base_model(str(with_config)) == (expected or str(with_config))
