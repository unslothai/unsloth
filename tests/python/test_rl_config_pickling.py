# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression: `torch.save(trainer.args, ...)` must keep working once Unsloth has
patched a TRL trainer, and the file it writes must stay readable without Unsloth.

`Trainer._save_checkpoint` ends in `torch.save(self.args, os.path.join(output_dir,
TRAINING_ARGS_NAME))`. Pickle stores a class as `__module__` + `__qualname__` and
then refuses unless the object living at that path *is* the class. Patching a
trainer rebinds `<X>Config` at the module the pristine class calls home, which
breaks that identity for every instance of the pristine class, and the generated
class used to answer to `Unsloth<X>Trainer.Unsloth<X>Config`, a module that only
exists beside a compiled cache.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys

import pytest


pytest.importorskip("trl")


@pytest.fixture(scope = "module")
def patched():
    import unsloth  # noqa: F401
    import trl.trainer.sft_config as config_module
    return config_module.SFTConfig


def _make(config_class, output_dir):
    return config_class(
        output_dir = str(output_dir),
        bf16 = False,
        fp16 = False,
        use_cpu = True,
    )


def test_patched_config_answers_to_the_trl_name(patched):
    # Without this the class pickles under a module that only ships inside a
    # compiled cache, so the checkpoint cannot be read anywhere else.
    assert patched.__module__ == "trl.trainer.sft_config"
    assert patched.__qualname__ == "SFTConfig"

    import trl.trainer.sft_config as config_module

    assert getattr(config_module, "SFTConfig") is patched


def test_pristine_config_instance_still_pickles(patched, tmp_path):
    # An instance built before Unsloth patched TRL, or handed back by TRL's own TrainingArguments -> SFTConfig
    # conversion, belongs to the pristine class.
    pristine = patched.__mro__[1]
    assert not pristine.__name__.startswith("Unsloth")

    args = _make(pristine, tmp_path)
    restored = pickle.loads(pickle.dumps(args))
    assert restored.output_dir == args.output_dir


def test_patched_config_instance_pickles(patched, tmp_path):
    args = _make(patched, tmp_path)
    restored = pickle.loads(pickle.dumps(args))
    assert restored.output_dir == args.output_dir


@pytest.mark.parametrize("use_pristine", [False, True])
def test_torch_save_load_round_trip(patched, tmp_path, use_pristine):
    torch = pytest.importorskip("torch")

    config_class = patched.__mro__[1] if use_pristine else patched
    args = _make(config_class, tmp_path)
    path = tmp_path / f"training_args_{int(use_pristine)}.bin"
    torch.save(args, path)

    restored = torch.load(path, weights_only = False)
    assert restored.output_dir == args.output_dir
    assert restored.max_steps == args.max_steps


def test_checkpoint_loads_without_unsloth(patched, tmp_path):
    """The whole point of pickling under the TRL name: a checkpoint written here
    has to open on a machine that only has transformers and TRL."""
    torch = pytest.importorskip("torch")

    path = tmp_path / "training_args.bin"
    torch.save(_make(patched, tmp_path), path)

    script = (
        # Baseline FIRST, so this measures what the load drags in rather than what the interpreter already had.
        # An editable install of unsloth puts its own import finder (__editable___unsloth_..._finder) into sys.modules
        # at startup, which answers to a name test but says nothing about the checkpoint.
        "import sys\n"
        "preloaded = set(sys.modules)\n"
        "import json, torch\n"
        "obj = torch.load(sys.argv[1], weights_only = False)\n"
        "leaked = sorted(\n"
        "    m for m in set(sys.modules) - preloaded\n"
        "    if 'unsloth' in m.lower() or m.startswith('Unsloth')\n"
        ")\n"
        "print(json.dumps({\n"
        "    'cls': type(obj).__module__ + '.' + type(obj).__name__,\n"
        "    'output_dir': obj.output_dir,\n"
        "    'leaked': leaked,\n"
        "}))\n"
    )
    env = dict(os.environ)
    # A stock environment: no worktree on the path, no compiled cache beside it.
    env.pop("PYTHONPATH", None)
    env.pop("UNSLOTH_COMPILE_LOCATION", None)
    process = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output = True,
        text = True,
        env = env,
        cwd = str(tmp_path),
    )
    assert process.returncode == 0, process.stderr[-3000:]

    result = json.loads(process.stdout.strip().splitlines()[-1])
    assert result["cls"] == "trl.trainer.sft_config.SFTConfig", result
    assert result["output_dir"] == str(tmp_path)
    assert result["leaked"] == [], result


def test_training_arguments_conversion_keeps_unsloth_fields(patched, tmp_path):
    """TRL converts a plain `TrainingArguments` with `args = SFTConfig(**dict_args)`.
    That has to land on the Unsloth subclass, or the config reaches the trainer
    without the fields the patched trainer reads (unslothai/unsloth#3931)."""
    pytest.importorskip("transformers")
    from transformers import TrainingArguments

    import inspect

    generated = sys.modules.get("UnslothSFTTrainer")
    if generated is None:
        pytest.skip("UnslothSFTTrainer compiled module not loaded")

    source = inspect.getsource(generated._UnslothSFTTrainer.__init__)
    if "dict_args" not in source:
        pytest.skip("this TRL release does not convert TrainingArguments inline")
    # The conversion must not name the bare TRL class: that global was imported before the patching and so still points
    # at the pristine class.
    assert "args = UnslothSFTConfig(**dict_args)" in source, source[:2000]

    training_arguments = TrainingArguments(
        output_dir = str(tmp_path),
        bf16 = False,
        fp16 = False,
        use_cpu = True,
    )
    dict_args = training_arguments.to_dict()
    dict_args["hub_token"] = training_arguments.hub_token
    dict_args.pop("push_to_hub_token", None)

    converted = generated.UnslothSFTConfig(**dict_args)
    assert isinstance(converted, patched)
    assert hasattr(converted, "unsloth_num_chunks")
    assert pickle.loads(pickle.dumps(converted)).output_dir == str(tmp_path)


def test_every_patched_config_pickles_portably(tmp_path):
    """Sweep: no patched config may pickle under a compiled-cache module."""
    import trl
    import unsloth  # noqa: F401

    checked = 0
    for name in sorted(x for x in dir(trl) if x.endswith("Config")):
        config_class = getattr(trl, name)
        if not isinstance(config_class, type):
            continue
        try:
            args = config_class(output_dir = str(tmp_path))
        except TypeError:
            continue
        pickle.dumps(args, protocol = 2)
        assert not config_class.__module__.startswith("Unsloth"), (
            f"{name} pickles under {config_class.__module__}, which does not exist "
            "without a compiled cache"
        )
        checked += 1
    assert checked > 0, "no TRL configs were exercised"


def test_patched_config_still_subclasses_the_pristine_one(patched):
    """The generated class is renamed onto TRL's name, which must not tempt a
    later patching pass into making it its own base."""
    base = patched.__mro__[1]
    assert base is not patched
    assert base.__name__ == "SFTConfig"
    assert base.__module__.startswith("trl.")
    assert "_unsloth_patched_rl_config" not in base.__dict__


def test_reducer_registration_is_idempotent(patched, tmp_path):
    """Patching can run more than once; it must not stack reducers or recurse."""
    import copyreg

    from unsloth.models.rl import _patch_trl_rl_trainers
    import trl.trainer.sft_config as config_module

    before = len(copyreg.dispatch_table)
    for _ in range(3):
        _patch_trl_rl_trainers("sft_trainer")

    assert len(copyreg.dispatch_table) == before, "reducers stacked on re-patch"
    assert config_module.SFTConfig is patched, "re-patching swapped the class"
    # A self-referential reducer would recurse here rather than return.
    assert pickle.loads(pickle.dumps(_make(patched, tmp_path))).output_dir == str(tmp_path)


def test_a_displaced_sibling_wrapper_is_covered(patched, tmp_path):
    """TRL's deprecation shims are siblings of the patched class, not its bases.

    `trl.trainer.<x>_config.<X>Config` subclasses the real class in
    `trl.experimental.<x>`, and the wrapper resolution generates the patched
    class from that same parent -- so a subclass-only guard skips the shim even
    though its module attribute has already been taken over, and any instance
    captured before patching stays unpicklable.
    """
    import copyreg

    from unsloth.models.rl import (
        _UNSLOTH_CONFIG_PICKLE_TARGET,
        _register_config_pickle_fallback,
    )

    experimental_parent = patched.__mro__[1]

    class _Shim(experimental_parent):
        pass

    assert not issubclass(patched, _Shim), "not the sibling shape this test is about"

    registered = copyreg.dispatch_table.get(_Shim)
    try:
        _register_config_pickle_fallback(_Shim, patched)
        assert copyreg.dispatch_table.get(_Shim) is not None, "sibling shim left unregistered"
        assert getattr(_Shim, _UNSLOTH_CONFIG_PICKLE_TARGET, None) is patched
        restored = pickle.loads(pickle.dumps(_make(_Shim, tmp_path)))
        assert restored.output_dir == str(tmp_path)
        # Reduced through the patched class, so the file loads without Unsloth.
        assert type(restored) is patched
    finally:
        copyreg.dispatch_table.pop(_Shim, None)
        if registered is not None:
            copyreg.dispatch_table[_Shim] = registered


def test_an_unrelated_class_is_not_reduced_through_the_patched_one(patched):
    """The widening stops at classes whose state the patched class can hold.

    Rebuilding an unrelated class as this one would drop fields silently, which
    is worse than the PicklingError it avoids.
    """
    import copyreg

    from unsloth.models.rl import _register_config_pickle_fallback

    class _Unrelated:
        pass

    _register_config_pickle_fallback(_Unrelated, patched)
    assert _Unrelated not in copyreg.dispatch_table


# ---------------------------------------------------------------------------
# The reported ordering: a fresh interpreter that imports trl, builds a config,
# and only then imports unsloth. The fixture above cannot reproduce it, because
# by the time this module runs the session has already imported unsloth, so the
# probe needs a subprocess of its own.
# ---------------------------------------------------------------------------

_PRISTINE_PROBE = r"""
import json, os, pickle, sys, tempfile

from trl import SFTConfig as pristine
import trl.trainer.sft_config as config_module

result = {"captured_pristine": config_module.SFTConfig is pristine}
config = pristine(output_dir = "unused", bf16 = False, fp16 = False, use_cpu = True)

import unsloth
from unsloth import FastLanguageModel
import torch

result["rebound"] = config_module.SFTConfig is not pristine

try:
    pickle.dumps(config)
    result["pickle"] = "ok"
except Exception as error:
    result["pickle"] = "%s: %s" % (type(error).__name__, error)

try:
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "training_args.bin")
        torch.save(config, path)
        restored = torch.load(path, weights_only = False)
        assert restored.output_dir == "unused"
        assert restored.per_device_train_batch_size == config.per_device_train_batch_size
    result["torch_round_trip"] = "ok"
except Exception as error:
    result["torch_round_trip"] = "%s: %s" % (type(error).__name__, error)

print("UNSLOTH_PROBE " + json.dumps(result))
"""


@pytest.fixture(scope = "module")
def pristine_probe(tmp_path_factory):
    environment = dict(os.environ)
    # Keep the generated module out of the shared cache directory.
    environment["UNSLOTH_COMPILE_LOCATION"] = str(tmp_path_factory.mktemp("compiled_cache"))
    finished = subprocess.run(
        [sys.executable, "-c", _PRISTINE_PROBE],
        capture_output = True,
        text = True,
        timeout = 1800,
        env = environment,
    )
    marker = [line for line in finished.stdout.splitlines() if line.startswith("UNSLOTH_PROBE ")]
    if not marker:
        pytest.skip(
            "could not run the trl-before-unsloth probe:\n"
            f"{finished.stdout[-2000:]}\n{finished.stderr[-2000:]}"
        )
    payload = json.loads(marker[-1][len("UNSLOTH_PROBE ") :])
    if not payload.get("captured_pristine"):
        pytest.skip("trl was already patched before the probe captured the class")
    if not payload.get("rebound"):
        pytest.skip("this TRL/Unsloth pair does not rebind the config module")
    return payload


def test_config_captured_before_patching_still_pickles(pristine_probe):
    """The reported regression, in the order it was reported."""
    assert pristine_probe["pickle"] == "ok", pristine_probe["pickle"]


def test_config_captured_before_patching_survives_torch_save(pristine_probe):
    """Mirrors `Trainer._save`, which is where the crash surfaced."""
    assert pristine_probe["torch_round_trip"] == "ok", pristine_probe["torch_round_trip"]
