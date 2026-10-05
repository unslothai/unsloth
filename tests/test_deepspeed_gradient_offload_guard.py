"""DeepSpeed ZeRO-1/2 must not run Unsloth's gradient-offload checkpointer.

ZeRO installs hooks that consume each parameter gradient during backward.
Unsloth's smart checkpoint path may offload or clear that gradient first, so
DeepSpeed receives ``None`` and crashes in ``grad_reduc.view(-1)`` (#4195).

These tests execute the small policy functions directly from source. They need
neither torch nor DeepSpeed and therefore cover the launch-time decision on
every CI platform.
"""

import ast
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "unsloth" / "models" / "_utils.py").read_text(encoding = "utf-8")


class _Logger:
    def __init__(self):
        self.messages = []

    def warning_once(self, message):
        self.messages.append(message)


def _load():
    tree = ast.parse(SRC)
    names = {
        "_accelerate_deepspeed_zero_stage",
        "apply_unsloth_gradient_checkpointing",
    }
    namespace = {"json": json, "os": os}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(ast.get_source_segment(SRC, node), namespace)
    assert names <= namespace.keys()
    return namespace


NS = _load()
resolve_stage = NS["_accelerate_deepspeed_zero_stage"]
apply_checkpointing = NS["apply_unsloth_gradient_checkpointing"]


@contextmanager
def _environment(**values):
    names = {
        "ACCELERATE_USE_DEEPSPEED",
        "ACCELERATE_DEEPSPEED_ZERO_STAGE",
        "ACCELERATE_DEEPSPEED_CONFIG_FILE",
    }
    old = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    for name, value in values.items():
        os.environ[name] = str(value)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _install_spies():
    calls = []
    logger = _Logger()
    NS["unpatch_unsloth_smart_gradient_checkpointing"] = lambda: calls.append("unpatch")
    NS["patch_unsloth_smart_gradient_checkpointing"] = lambda **kwargs: calls.append(
        ("patch", kwargs)
    )
    NS["logger"] = logger
    return calls, logger


def test_zero_1_and_2_fall_back_to_standard_checkpointing():
    for stage in (1, 2):
        calls, logger = _install_spies()
        with _environment(
            ACCELERATE_USE_DEEPSPEED = "true",
            ACCELERATE_DEEPSPEED_ZERO_STAGE = stage,
        ):
            assert apply_checkpointing("unsloth", 4096, "bf16") is True
        assert calls == ["unpatch"]
        assert len(logger.messages) == 1
        assert f"ZeRO-{stage}" in logger.messages[0]


def test_zero_3_keeps_smart_checkpointing():
    calls, logger = _install_spies()
    with _environment(
        ACCELERATE_USE_DEEPSPEED = "true",
        ACCELERATE_DEEPSPEED_ZERO_STAGE = "3",
    ):
        assert apply_checkpointing("unsloth", 4096, "bf16") == "unsloth"
    assert calls == [("patch", {"dtype": "bf16"})]
    assert logger.messages == []


def test_config_file_zero_1_and_2_fall_back_to_standard_checkpointing():
    for stage in (1, 2):
        calls, logger = _install_spies()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deepspeed.json"
            path.write_text(
                json.dumps({"zero_optimization": {"stage": stage}}),
                encoding = "utf-8",
            )
            with _environment(
                ACCELERATE_USE_DEEPSPEED = "true",
                ACCELERATE_DEEPSPEED_CONFIG_FILE = path,
            ):
                assert apply_checkpointing("unsloth", 4096, "bf16") is True
        assert calls == ["unpatch"]
        assert len(logger.messages) == 1
        assert f"ZeRO-{stage}" in logger.messages[0]


def test_direct_stage_takes_precedence_over_config_file():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "deepspeed.json"
        path.write_text(
            json.dumps({"zero_optimization": {"stage": 2}}),
            encoding = "utf-8",
        )
        with _environment(
            ACCELERATE_USE_DEEPSPEED = "true",
            ACCELERATE_DEEPSPEED_ZERO_STAGE = "3",
            ACCELERATE_DEEPSPEED_CONFIG_FILE = path,
        ):
            assert resolve_stage() == 3


def test_stage_is_ignored_without_the_deepspeed_launch_flag():
    calls, logger = _install_spies()
    with _environment(ACCELERATE_DEEPSPEED_ZERO_STAGE = "2"):
        assert resolve_stage() is None
        assert apply_checkpointing("unsloth", 4096, "bf16") == "unsloth"
    assert calls == [("patch", {"dtype": "bf16"})]
    assert logger.messages == []


def test_invalid_or_incomplete_config_files_do_not_disable_offloading():
    cases = (
        "{not-json",
        "{}",
        '{"zero_optimization": {}}',
        '{"zero_optimization": {"stage": "auto"}}',
        "[]",
    )
    for content in cases:
        calls, logger = _install_spies()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deepspeed.json"
            path.write_text(content, encoding = "utf-8")
            with _environment(
                ACCELERATE_USE_DEEPSPEED = "true",
                ACCELERATE_DEEPSPEED_CONFIG_FILE = path,
            ):
                assert resolve_stage() is None
                assert apply_checkpointing("unsloth", 4096, "bf16") == "unsloth"
        assert calls == [("patch", {"dtype": "bf16"})]
        assert logger.messages == []


def test_missing_config_file_does_not_disable_offloading():
    calls, logger = _install_spies()
    with _environment(
        ACCELERATE_USE_DEEPSPEED = "true",
        ACCELERATE_DEEPSPEED_CONFIG_FILE = "/definitely/missing/deepspeed.json",
    ):
        assert resolve_stage() is None
        assert apply_checkpointing("unsloth", 4096, "bf16") == "unsloth"
    assert calls == [("patch", {"dtype": "bf16"})]
    assert logger.messages == []


def test_invalid_stage_does_not_disable_offloading():
    calls, logger = _install_spies()
    with _environment(
        ACCELERATE_USE_DEEPSPEED = "yes",
        ACCELERATE_DEEPSPEED_ZERO_STAGE = "not-an-integer",
    ):
        assert apply_checkpointing("unsloth", 4096, "bf16") == "unsloth"
    assert calls == [("patch", {"dtype": "bf16"})]
    assert logger.messages == []


def test_short_context_and_explicit_modes_keep_existing_behavior():
    calls, logger = _install_spies()
    with _environment():
        assert apply_checkpointing("unsloth", 256, "bf16") is True
        assert apply_checkpointing(True, 4096, "bf16") is True
        assert apply_checkpointing(False, 4096, "bf16") is False
    assert calls == ["unpatch", "unpatch", "unpatch"]
    assert logger.messages == []
