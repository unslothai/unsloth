# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""On MLX, trl.__all__ is filtered on first read: resolving every name at import pulled torch in."""

import importlib.metadata
import importlib.util
import os
import pathlib
import platform
import subprocess
import sys
import types

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "_mlx_trl_all.py"


@pytest.fixture
def lazy_all():
    spec = importlib.util.spec_from_file_location("_unsloth_mlx_trl_all_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _filter(trl, names):
    """Same contract as unsloth's _safe_mlx_trl_star_exports."""
    kept = []
    for name in names:
        try:
            getattr(trl, name)
        except Exception:
            continue
        kept.append(name)
    for name in ("SFTConfig", "SFTTrainer"):
        if name not in kept:
            kept.append(name)
    return kept


class _LazyLike(types.ModuleType):
    """trl's _LazyModule shape: a class-level __getattr__ importing on demand."""

    def __init__(self, name, lazy):
        super().__init__(name)
        self._lazy = lazy
        self.resolved = []
        self.__all__ = list(lazy)

    def __getattr__(self, name):
        if name not in self._lazy:
            raise AttributeError(name)
        self.resolved.append(name)
        value = self._lazy[name]()
        setattr(self, name, value)
        return value


def _broken():
    raise RuntimeError("optional dependency missing")


def test_nothing_resolves_until_all_is_read(lazy_all, monkeypatch):
    trl = _LazyLike("trl", {"DataCollator": object, "Broken": _broken, "GRPOTrainer": object})
    trl.SFTTrainer = trl.SFTConfig = object
    lazy_all.install_lazy_trl_all(trl, _filter)
    assert trl.resolved == []
    assert lazy_all.raw_trl_all(trl) == ["DataCollator", "Broken", "GRPOTrainer"]
    assert trl.resolved == []
    assert trl.__all__ == ["DataCollator", "GRPOTrainer", "SFTConfig", "SFTTrainer"]
    assert sorted(trl.resolved) == ["Broken", "DataCollator", "GRPOTrainer"]
    # Cached: a second read resolves nothing more.
    trl.resolved.clear()
    assert trl.__all__[-1] == "SFTTrainer" and trl.resolved == []
    monkeypatch.setitem(sys.modules, "trl", trl)
    namespace = {}
    exec("from trl import *", namespace)
    assert set(namespace) - {"__builtins__"} == {
        "DataCollator",
        "GRPOTrainer",
        "SFTConfig",
        "SFTTrainer",
    }


def test_class_level_getattr_still_serves_other_names(lazy_all):
    trl = _LazyLike("trl", {"DataCollator": lambda: "collator"})
    lazy_all.install_lazy_trl_all(trl, _filter)
    assert trl.DataCollator == "collator"
    with pytest.raises(AttributeError):
        trl.missing_name
    assert "DataCollator" in dir(trl)


def test_module_dict_getattr_is_chained(lazy_all):
    trl = types.ModuleType("trl")
    trl.ExistingExport = object()
    trl.__all__ = ["ExistingExport", "BrokenExport"]

    def _module_getattr(name):
        if name == "BrokenExport":
            raise RuntimeError("optional dependency missing")
        raise AttributeError(name)

    trl.__getattr__ = _module_getattr
    lazy_all.install_lazy_trl_all(trl, _filter)
    with pytest.raises(RuntimeError):
        trl.BrokenExport
    assert trl.__all__ == ["ExistingExport", "SFTConfig", "SFTTrainer"]


def test_reinstall_keeps_the_original_names_and_getattr(lazy_all):
    trl = types.ModuleType("trl")
    trl.__all__ = ["A"]
    trl.A = 1
    original = lambda name: (_ for _ in ()).throw(AttributeError(name))  # noqa: E731
    trl.__getattr__ = original
    lazy_all.install_lazy_trl_all(trl, _filter)
    assert trl.__all__ == ["A", "SFTConfig", "SFTTrainer"]
    lazy_all.install_lazy_trl_all(trl, _filter)
    assert lazy_all.raw_trl_all(trl) == ["A"]
    assert trl.__getattr__._unsloth_previous is original
    assert trl.__all__ == ["A", "SFTConfig", "SFTTrainer"]


def test_module_without_all(lazy_all):
    trl = types.ModuleType("trl")
    lazy_all.install_lazy_trl_all(trl, _filter)
    assert trl.__all__ == ["SFTConfig", "SFTTrainer"]


def test_a_reentrant_all_read_during_filtering_sees_the_raw_names(lazy_all):
    install_lazy_trl_all = lazy_all.install_lazy_trl_all

    trl = types.ModuleType("fake_trl_reentrant")
    trl.__all__ = ["A", "B"]
    seen = []

    def filter_exports(module, names):
        # An export whose import reads trl.__all__ again mid-filter.
        seen.append(list(module.__all__))
        return [n for n in names if n != "B"]

    install_lazy_trl_all(trl, filter_exports)
    assert trl.__all__ == ["A"]
    assert seen == [["A", "B"]]
    assert trl.__all__ == ["A"]


class _LazyDirLike(_LazyLike):
    """trl's _LazyModule.__dir__: the namespace listing, then any __all__ name not in it."""

    def __dir__(self):
        result = super().__dir__()
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result


@pytest.mark.parametrize("dir_first", [True, False])
def test_dir_matches_the_eager_filter(lazy_all, dir_first):
    def make():
        trl = _LazyDirLike(
            "trl", {"DataCollator": object, "Broken": _broken, "GRPOTrainer": object}
        )
        trl.SFTTrainer = trl.SFTConfig = object
        return trl

    eager = make()
    eager.__all__ = _filter(eager, list(eager.__all__))
    lazy = make()
    lazy_all.install_lazy_trl_all(lazy, _filter)
    if dir_first:
        listed = dir(lazy)
        assert lazy.__all__ == eager.__all__
    else:
        assert lazy.__all__ == eager.__all__
        listed = dir(lazy)
    assert listed == dir(eager)
    # Built: no hook or bookkeeping name is left in trl's namespace.
    assert not {"__getattr__", "__dir__", "_unsloth_mlx_trl_all"} & set(vars(lazy))


_needs_real_mlx = pytest.mark.skipif(
    platform.system() != "Darwin"
    or platform.machine() != "arm64"
    or os.environ.get("UNSLOTH_FORCE_GPU_PATH", "0") == "1"
    or importlib.util.find_spec("mlx") is None
    or importlib.util.find_spec("trl") is None,
    reason = "the real MLX import path needs Apple Silicon with mlx and trl",
)


@_needs_real_mlx
def test_real_mlx_import_leaves_trl_all_unbuilt():
    probe = (
        "import sys, unsloth, trl\n"
        "assert unsloth.DEVICE_TYPE == 'mlx', unsloth.DEVICE_TYPE\n"
        "assert '__all__' not in vars(trl), 'import unsloth filtered trl.__all__ eagerly'\n"
        "names = trl.__all__\n"
        "assert {'SFTConfig', 'SFTTrainer'} <= set(names), names\n"
        "ns = {}\n"
        "exec('from trl import *', ns)\n"
        "assert set(names) <= set(ns), set(names) - set(ns)\n"
        "assert ns['SFTTrainer'] is unsloth.UnslothTrainer\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd = pathlib.Path(__file__).resolve().parents[1],
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]


_SHIM_PROBE_PREAMBLE = """
import pickle, unsloth, trl
import mlx.nn as nn
from unsloth_zoo.mlx import trainer as zoo
assert unsloth.DEVICE_TYPE == "mlx"
def refused(fn, *needles, kind=NotImplementedError):
    try:
        fn()
    except kind as exc:
        assert all(n in str(exc) for n in needles), str(exc)
    else:
        raise AssertionError(f"not refused: {needles}")
model = nn.Linear(1, 1)
model.max_seq_length = 96
tok = object()
"""


def _run_shim_probe(body, before_unsloth = ""):
    result = subprocess.run(
        [sys.executable, "-c", before_unsloth + _SHIM_PROBE_PREAMBLE + body],
        cwd = pathlib.Path(__file__).resolve().parents[1],
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]


@_needs_real_mlx
def test_trl_trainers_zoo_implements_are_mapped():
    _run_shim_probe("""
for stem in ("DPO", "ORPO", "KTO", "GRPO", "CPO"):
    trainer, config = getattr(trl, stem + "Trainer"), getattr(trl, stem + "Config")
    assert issubclass(trainer, getattr(zoo, f"MLX{stem}Trainer")), stem
    assert issubclass(config, getattr(zoo, f"MLX{stem}Config")), stem
    cfg = config(output_dir="o", optim="adamw_torch")
    assert cfg.learning_rate == 1e-6 and cfg.per_device_train_batch_size == 8, stem
    assert cfg.gradient_accumulation_steps == 1 and cfg.warmup_steps == 0, stem
    assert cfg.weight_decay == 0.0 and cfg.num_train_epochs == 3, stem
    assert cfg.optim == "adamw" and cfg.save_steps == 500, stem
    assert config(max_steps=1, save_strategy="no").save_steps == 0, stem
    assert config(learning_rate=0.5, max_steps=2).learning_rate == 0.5, stem
    assert type(pickle.loads(pickle.dumps(cfg))) is config, stem
for name in ("PatchDPOTrainer", "PatchKTOTrainer", "PatchFastRL"):
    assert getattr(unsloth, name)() is None

# TRL's own positional order, ref_model and a trainer-level config field.
cfg = trl.DPOConfig(max_steps=1, max_length=512, max_prompt_length=64)
t = trl.DPOTrainer(model, None, cfg, None, [], None, tok, beta=0.25)
assert t.args.beta == 0.25 and cfg.beta == 0.1
assert t.args.max_length == 512, t.args.max_length
assert cfg.max_seq_length == zoo.MLXTrainingConfig.max_seq_length, "max_length is not the SFT alias here"
assert t.args.max_seq_length == 96, "follows the loaded model"
c = trl.DPOTrainer(model=model, args=cfg, train_dataset=[], processing_class=tok, max_seq_length=2048)
assert (c.args.max_seq_length, c.args.max_length) == (2048, 512), "trainer context length keeps TRL's budget"
long_model = nn.Linear(1, 1)
long_model.max_seq_length = 4096
b = trl.DPOTrainer(model=long_model, args=trl.DPOConfig(max_steps=1), train_dataset=[], max_length=48)
assert (b.args.max_seq_length, b.args.max_length) == (4096, 48), "trainer budget leaves the context length implicit"
import dataclasses
swept = trl.DPOTrainer(model=long_model, args=dataclasses.replace(trl.DPOConfig(max_length=4096), beta=0.2), train_dataset=[]).args
assert (swept.max_seq_length, swept.dataset_order, swept.beta) == (4096, "torch_randperm", 0.2), "a replaced config stays implicit"
assigned = trl.DPOConfig(max_length=4096)
assigned.max_seq_length = zoo.MLXTrainingConfig.max_seq_length
assert trl.DPOTrainer(model=long_model, args=assigned, train_dataset=[]).args.max_seq_length == zoo.MLXTrainingConfig.max_seq_length
plain = trl.DPOTrainer(model=model, args=zoo.MLXTrainingConfig(max_steps=1), train_dataset=[]).args
assert (plain.max_grad_norm, plain.max_grad_leaf_norm) == (0.0, 1.0), "zoo's leaf clip default survives conversion"
assert trl.DPOTrainer(model=model, args=trl.SFTConfig(max_steps=1), train_dataset=[]).args.max_grad_norm == 1.0
assert t.args.dataset_order == "torch_randperm" and t.args.max_grad_norm == 1.0
assert t.args.append_eos is True, "TRL appends EOS to completions"
assert t.tokenizer is tok and t.ref_model is None
ref = object()
assert trl.DPOTrainer(model=model, ref_model=ref, args=cfg, train_dataset=[], processing_class=tok).ref_model is ref

k = trl.KTOTrainer(model, ref, trl.KTOConfig(max_steps=1), [], None, tok, model_adapter_name="a")
assert k.ref_model is ref and k.model_adapter_name == "a"
assert k.args.dataset_order == "default", "each row's KL pair is its batch neighbor"
o = trl.ORPOTrainer(model, trl.ORPOConfig(max_steps=1), None, [], None, tok)
assert o.args.dataset_order == "torch_randperm"
assert isinstance(trl.CPOTrainer(model, trl.CPOConfig(max_steps=1), None, [], None, tok).args, zoo.MLXCPOConfig)

reward = lambda completions, **kw: [0.0] * len(completions)
g_cfg = trl.GRPOConfig(max_steps=1, num_generations=2, per_device_train_batch_size=2, max_completion_length=8)
g = trl.GRPOTrainer(model, [reward], g_cfg, [], None, tok)
assert g.reward_funcs == [reward] and g.args.dataset_order == "default"
refused(lambda: trl.GRPOTrainer(model, [reward], g_cfg, [], None, tok, beta=0.5), "GRPOConfig")

# Trainer-level overrides are normalized like config arguments.
o_cfg = trl.DPOConfig(max_steps=100, warmup_ratio=0.1)
o = trl.DPOTrainer(model=model, args=o_cfg, train_dataset=[], loraplus_lr_ratio=16, warmup_steps=30, optim="adamw_torch")
assert o.args.lora_plus_ratio == 16 and o.args.optim == "adamw"
assert o.args.warmup_steps == 30 and o.args._unsloth_mlx_warmup_steps_explicit
for trainer, cfg_ in ((trl.DPOTrainer, o_cfg), (trl.SFTTrainer, trl.SFTConfig(max_steps=1))):
    for kw in ({"lora_plus_ratio": 8, "loraplus_lr_ratio": 16}, {"loraplus_lr_ratio": 16, "lora_plus_ratio": 8}):
        assert trainer(model=model, args=cfg_, train_dataset=[], **kw).args.lora_plus_ratio == 8, kw
r = trl.DPOTrainer(model=model, args=o_cfg, train_dataset=[], max_steps=200)
assert r.args.warmup_steps == 20, r.args.warmup_steps

# Another config type carries its fields over, zoo's own config passes untouched.
assert trl.DPOTrainer(model=model, args=trl.SFTConfig(max_steps=7, learning_rate=0.1), train_dataset=[], processing_class=tok).args.learning_rate == 0.1
raw = zoo.MLXDPOConfig(max_steps=1, beta=0.3)
assert trl.DPOTrainer(model=model, args=raw, train_dataset=[], processing_class=tok).args.beta == 0.3
assert trl.DPOTrainer(model=model, args=raw, train_dataset=[]).args.max_seq_length == 96, "default follows the model"
for length, expected in ((64, 64), (512, 96)):
    raw = zoo.MLXDPOConfig(max_steps=1, max_seq_length=length)
    assert trl.DPOTrainer(model=model, args=raw, train_dataset=[]).args.max_seq_length == expected
""")


@_needs_real_mlx
def test_trl_mappings_refuse_what_zoo_lacks():
    _run_shim_probe("""
# A TRL field at its inert value is accepted, any other value is refused by name.
trl.DPOConfig(rpo_alpha=None, label_pad_token_id=-100, use_weighting=False)
refused(lambda: trl.DPOConfig(rpo_alpha=0.1), "rpo_alpha")
refused(lambda: trl.DPOConfig(use_weighting=True), "use_weighting")
refused(lambda: trl.DPOConfig(no_such_field=1), "no_such_field")
refused(
    lambda: trl.DPOTrainer(model=model, args=trl.DPOConfig(max_steps=1), train_dataset=[], compute_metrics=print),
    "DPOTrainer", "compute_metrics",
)
refused(lambda: trl.ORPOTrainer(model=model, train_dataset=[], peft_config=object()), "ORPOTrainer", "peft_config")
refused(lambda: trl.DPOTrainer(model="org/repo", train_dataset=[]), "model")
refused(lambda: trl.DPOTrainer(model=model, ref_model="org/repo", train_dataset=[]), "ref_model")
for name in ("GKDTrainer", "PPOTrainer", "RewardTrainer"):
    trainer = getattr(trl, name, None)
    if trainer is None:
        continue
    assert getattr(trainer, "_unsloth_mlx_unsupported", False), name
    refused(trainer, name, "DPO", "GRPO", "SFT")
""")


@_needs_real_mlx
def test_trl_configs_match_trl_semantics_and_stay_the_callers():
    _run_shim_probe("""
cfg = trl.DPOConfig(output_dir="o", max_seq_length=4096)
first = trl.DPOTrainer(model=model, args=cfg, train_dataset=[])
assert first.args.max_seq_length == 96 and cfg.max_seq_length == 4096
longer = nn.Linear(1, 1)
longer.max_seq_length = 2048
assert trl.DPOTrainer(model=longer, args=cfg, train_dataset=[]).args.max_seq_length == 2048
assert trl.DPOTrainer(model=model, evaluation_strategy="epoch", train_dataset=[]).args.eval_strategy == "epoch"
assert trl.SFTConfig(output_dir="o", eval_strategy="steps", logging_steps=7).eval_steps == 7
assert trl.SFTConfig(output_dir="o", eval_strategy="steps", eval_steps=3, logging_steps=7).eval_steps == 3
assert trl.SFTConfig(output_dir="o", eval_strategy="no", logging_steps=7).eval_steps == 0
assert trl.DPOConfig(output_dir="o", f_alpha_divergence_coef=0.5).output_dir == "o"
assert trl.DPOConfig(output_dir="o", use_liger_loss=None).output_dir == "o"
refused(lambda: trl.DPOConfig(output_dir="o", use_liger_loss=True), "use_liger_loss")
refused(lambda: trl.DPOConfig(output_dir="o", f_divergence_type="alpha_divergence"), "f_divergence_type")
""")


@_needs_real_mlx
def test_trl_config_defaults_match_trl():
    _run_shim_probe("""
assert trl.SFTConfig(output_dir="o", eval_strategy="steps").eval_steps == 10
assert trl.DPOConfig(output_dir="o", eval_strategy="steps").eval_steps == 10
assert trl.DPOConfig(output_dir="o", eval_strategy="steps", logging_steps=7).eval_steps == 7
assert trl.DPOConfig(output_dir="o", eval_steps=1).eval_strategy == "no"
assert trl.DPOConfig(output_dir="o").logging_steps == 10
assert unsloth.UnslothTrainingArguments(output_dir="o", eval_strategy="steps").eval_steps == 10
assert trl.DPOTrainer(model=model, args=trl.DPOConfig(output_dir="o"), train_dataset=[], eval_strategy="steps").args.eval_steps == 10
sft = trl.SFTTrainer(model=model, args=trl.SFTConfig(output_dir="o"), train_dataset=[], eval_strategy="steps")
assert sft.args.eval_steps == 10 and sft.args.logging_steps == 1
assert trl.SFTTrainer(model=model, args=trl.SFTConfig(output_dir="o", logging_steps=1), train_dataset=[], eval_strategy="steps").args.eval_steps == 1

from packaging.version import Version
modern = Version(trl.__version__) >= Version("0.29.0")
assert trl.DPOConfig(output_dir="o").truncation_mode == ("keep_start" if modern else "keep_end")
assert trl.DPOConfig(output_dir="o", truncation_mode="keep_end").truncation_mode == "keep_end"
assert trl.ORPOConfig(output_dir="o").truncation_mode == "keep_end"
assert trl.DPOConfig(output_dir="o").max_prompt_length == (None if modern else 512)
assert trl.DPOConfig(output_dir="o", max_prompt_length=64).max_prompt_length == 64
grpo = trl.GRPOConfig(output_dir="o", num_generations=2, per_device_train_batch_size=2)
assert grpo.max_completion_length == (512 if Version(trl.__version__) >= Version("1.10.0") else 256)

over = trl.DPOTrainer(model=model, args=trl.DPOConfig(output_dir="o"), train_dataset=[], max_grad_norm=0, dataset_order="default").args
assert (over.max_grad_norm, over.dataset_order) == (0, "default")
warm = trl.SFTConfig(output_dir="o", num_train_epochs=1, warmup_steps=0, warmup_ratio=0.1)
assert trl.SFTTrainer(model=model, args=warm, train_dataset=[], warmup_ratio=0.0).args.warmup_steps == 0

edited = trl.SFTConfig(output_dir="o", max_steps=100, warmup_ratio=0.1)
edited.warmup_steps, edited.logging_steps = 30, 7
rebuilt = trl.SFTTrainer(model=model, args=edited, train_dataset=[], learning_rate=1e-4).args
assert (rebuilt.warmup_steps, rebuilt.logging_steps) == (30, 7)
untouched = trl.SFTConfig(output_dir="o", max_steps=100, warmup_ratio=0.1)
assert trl.SFTTrainer(model=model, args=untouched, train_dataset=[], max_steps=200).args.warmup_steps == 20
""")


@_needs_real_mlx
def test_trainers_fall_back_to_the_models_tokenizer():
    _run_shim_probe("""
model._tokenizer = tok
for trainer in (trl.DPOTrainer(model=model, train_dataset=[]), trl.SFTTrainer(model=model, train_dataset=[])):
    assert trainer.tokenizer is tok and trainer.processing_class is tok
other = object()
assert trl.DPOTrainer(model=model, train_dataset=[], processing_class=other).tokenizer is other
assert trl.SFTTrainer(model=model, train_dataset=[], data_collator=unsloth.UnslothVisionDataCollator(model, other)).processing_class is other
""")


@_needs_real_mlx
def test_hf_argument_parser_builds_the_same_trl_configs():
    pytest.importorskip("transformers")
    _run_shim_probe("""
import dataclasses
from transformers import HfArgumentParser

def parsed(config, *flags):
    return HfArgumentParser(config).parse_args_into_dataclasses(args=["--output_dir", "o", *flags])[0]

for stem, extra in (("DPO", {}), ("ORPO", {}), ("KTO", {}), ("CPO", {}), ("GRPO", {"num_generations": 2, "per_device_train_batch_size": 2})):
    config = getattr(trl, stem + "Config")
    flags = [text for key, value in extra.items() for text in ("--" + key, str(value))]
    built, direct = parsed(config, *flags), config(output_dir="o", **extra)
    for field in dataclasses.fields(config):
        assert getattr(built, field.name) == getattr(direct, field.name), (stem, field.name)
    for name, value in vars(direct).items():
        if name.startswith("_unsloth"):
            assert getattr(built, name) == value, (stem, name)

explicit = parsed(trl.DPOConfig, "--learning_rate", "0.0002", "--max_steps", "60", "--loss_type", "ipo")
assert (explicit.learning_rate, explicit.max_steps, explicit.loss_type) == (0.0002, 60, ["ipo"])
assert parsed(trl.DPOConfig, "--max_grad_norm", "0.5")._unsloth_mlx_grad_clip_explicit is True
assert parsed(trl.DPOConfig, "--num_train_epochs", "0.5").num_train_epochs == 0.5

long_model = nn.Linear(1, 1)
long_model.max_seq_length = 4096
args = parsed(trl.DPOConfig, "--max_seq_length", "2048", "--dataset_order", "default")
explicit = trl.DPOTrainer(model=long_model, args=args, train_dataset=[]).args
assert (explicit.max_seq_length, explicit.dataset_order) == (2048, "default")
unset = trl.DPOTrainer(model=long_model, args=parsed(trl.DPOConfig), train_dataset=[]).args
assert (unset.max_seq_length, unset.dataset_order) == (4096, "torch_randperm")
assert trl.DPOConfig(output_dir="o").max_seq_length == zoo.MLXTrainingConfig.max_seq_length

off = parsed(trl.DPOConfig, "--max_grad_norm", "0")
assert trl.DPOTrainer(model=model, args=off, train_dataset=[]).args.max_grad_norm == 0
kept = dataclasses.replace(trl.DPOConfig(output_dir="o", max_grad_norm=0.0), beta=0.3)
assert trl.DPOTrainer(model=model, args=kept, train_dataset=[]).args.max_grad_norm == 0
assert trl.DPOConfig(output_dir="o", max_grad_value=0.0).max_grad_norm == 0
assert trl.DPOConfig(output_dir="o", max_grad_value=None).max_grad_norm == 1.0
assert zoo._resolve_mlx_grad_clipping(trl.DPOConfig(output_dir="o", max_grad_value=None))[-1] == "global_norm"
assert zoo._resolve_mlx_grad_clipping(trl.DPOConfig(output_dir="o", max_grad_value=0.0))[-1] == "none"
for stem in ("DPO", "ORPO", "KTO", "CPO", "GRPO"):
    extra = {"num_generations": 2, "per_device_train_batch_size": 2} if stem == "GRPO" else {}
    extra = {**extra, "max_completion_length": 8} if stem == "GRPO" else extra
    trainer_extra = {"reward_funcs": [print], "processing_class": tok} if stem == "GRPO" else {}
    config, trainer = getattr(trl, stem + "Config"), getattr(trl, stem + "Trainer")
    for norm in (0, -1, None):
        built = trainer(model=model, args=config(output_dir="o", max_grad_norm=norm, **extra), train_dataset=[], **trainer_extra)
        assert zoo._resolve_mlx_grad_clipping(built.args)[-1] == "none", stem
for key in ("max_grad_value", "max_grad_leaf_norm"):
    t = trl.DPOTrainer(model=model, args=trl.DPOConfig(output_dir="o"), train_dataset=[], **{key: 0.0})
    assert zoo._resolve_mlx_grad_clipping(t.args)[-1] == "none", key
long_model = nn.Linear(1, 1)
long_model.max_seq_length = 4096
for name, value in (
    ("max_grad_norm", 0.0), ("max_grad_norm", -1), ("max_grad_norm", None), ("max_grad_norm", 0.5),
    ("max_grad_value", 0.0), ("max_grad_value", 0.5), ("max_grad_leaf_norm", 0.0), ("max_grad_leaf_norm", 0.5),
    ("max_seq_length", 128), ("warmup_steps", 30), ("logging_steps", 7),
):
    for overrides in ({}, {"seed": 9}):
        built = trl.DPOTrainer(model=long_model, args=trl.DPOConfig(output_dir="o", **{name: value}), train_dataset=[], **overrides).args
        late = trl.DPOConfig(output_dir="o")
        setattr(late, name, value)
        assigned = trl.DPOTrainer(model=long_model, args=late, train_dataset=[], **overrides).args
        assert getattr(built, name) == getattr(assigned, name) == value, (name, value, overrides)
        assert zoo._resolve_mlx_grad_clipping(built) == zoo._resolve_mlx_grad_clipping(assigned), (name, value, overrides)
        assert built.max_seq_length == assigned.max_seq_length, (name, value, overrides)

reused = trl.DPOTrainer(model=long_model, args=trl.DPOConfig(output_dir="o"), train_dataset=[]).args
assert trl.DPOTrainer(model=model, args=reused, train_dataset=[]).args.max_seq_length == 96
native = zoo.MLXDPOConfig(max_steps=1)
for overrides in ({}, {"seed": native.seed}):
    args = trl.DPOTrainer(model=model, args=native, train_dataset=[], **overrides).args
    assert zoo._resolve_mlx_grad_clipping(args)[-1] == "leaf_norm", overrides
edited_norm = trl.DPOConfig(output_dir="o")
edited_norm.max_grad_norm = 0.5
assert trl.DPOTrainer(model=model, args=edited_norm, train_dataset=[], seed=9).args.max_grad_norm == 0.5
assert trl.DPOTrainer(model=model, args=trl.DPOConfig(output_dir="o"), train_dataset=[], seed=9).args.max_grad_norm == 1.0
assert parsed(trl.GRPOConfig, "--num_generations", "2", "--per_device_train_batch_size", "2", "--scale_rewards", "batch").scale_rewards == "batch"
assert pickle.loads(pickle.dumps(parsed(trl.DPOConfig))).learning_rate == 1e-6
""")


@_needs_real_mlx
def test_sft_shim_forwards_what_zoo_honors():
    _run_shim_probe("""
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)
    args = trl.SFTConfig(
        max_steps=1,
        assistant_only_loss=True,
        logging_first_step=True,
        eval_delay=3,
        include_num_input_tokens_seen="all",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
    )
assert args.assistant_only_loss is True and args.logging_first_step is True
assert args.eval_delay == 3 and args.include_num_input_tokens_seen == "all"
assert (args.eval_strategy, args.save_strategy, args.logging_strategy) == ("epoch",) * 3
assert trl.SFTConfig(max_steps=1, evaluation_strategy="epoch").eval_strategy == "epoch"
assert trl.SFTConfig(max_steps=1, completion_only_loss=False).completion_only_loss is False

# Text models train on completions: zoo masks prompt/completion rows.
for key in ("completion_only_loss", "train_on_completions"):
    t = trl.SFTTrainer(model, trl.SFTConfig(max_steps=1, **{key: True}), None, [], None, tok)
    assert getattr(t.args, key) is True

# Rebuilding the config for a trainer-level override keeps them.
t = trl.SFTTrainer(model=model, args=args, train_dataset=[], processing_class=tok, packing=False, seed=9)
assert t.args.seed == 9 and t.args.assistant_only_loss is True and t.args.logging_first_step is True

t = trl.SFTTrainer(model=model, args=args, train_dataset=[], processing_class=tok, loraplus_lr_ratio=16)
assert t.args.lora_plus_ratio == 16

refused(lambda: trl.SFTConfig(dataloader_drop_last=True), "dataloader_drop_last")
for fast in (unsloth.FastLanguageModel, unsloth.FastModel, unsloth.FastVisionModel):
    assert fast.for_training(model) is model
""")


@_needs_real_mlx
def test_trl_configs_built_before_the_shim_keep_their_meaning():
    # importorskip would find the shim an earlier test left in sys.modules.
    try:
        importlib.metadata.version("trl")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(
            reason = "trl is not installed, so no real DPOConfig can be built before the shim"
        )
    _run_shim_probe(
        """
assert RealDPO is not trl.DPOConfig
t = trl.DPOTrainer(model=model, args=RealDPO(output_dir="o", beta=0.3, max_length=99), train_dataset=[], processing_class=tok)
assert isinstance(t.args, zoo.MLXDPOConfig) and (t.args.beta, t.args.max_length, t.args.learning_rate) == (0.3, 99, 1e-6)
if hasattr(RealDPO(output_dir="o"), "rpo_alpha"):
    refused(lambda: trl.DPOTrainer(model=model, args=RealDPO(output_dir="o", rpo_alpha=1.0), train_dataset=[]), "rpo_alpha")
refused(lambda: trl.DPOTrainer(model=model, args=RealDPO(output_dir="o", use_weighting=True), train_dataset=[]), "use_weighting")
assert trl.DPOTrainer(model=model, args=RealDPO(output_dir="o", eval_strategy="epoch"), train_dataset=[]).args.eval_strategy == "epoch"
refused(lambda: trl.DPOTrainer(model=model, args=RealDPO(output_dir="o", dataloader_drop_last=True), train_dataset=[]), "dataloader_drop_last")
assert trl.DPOTrainer(model=model, args=RealDPO(output_dir="o", bf16=False, learning_rate=0.1, save_strategy="no", optim="adamw_torch"), train_dataset=[]).args.learning_rate == 0.1
for stem, real in RealConfigs.items():
    config = getattr(trl, stem + "Config")
    assert isinstance(unsloth._coerce_mlx_training_args(real(output_dir="o"), None, config), config), stem
refused(lambda: trl.GRPOTrainer(model, [print], RealGRPO(output_dir="o"), [], None, tok), "GRPOConfig")

class RefusesBf16(RealDPO):
    def __post_init__(self):
        assert self.bf16 is False, "Your setup doesn't support bf16/gpu."
        super().__post_init__()
assert trl.DPOTrainer(model=model, args=RefusesBf16(output_dir="o", bf16=False), train_dataset=[]).args.beta == 0.1
""",
        before_unsloth = (
            "import dataclasses, trl\n"
            # TRL defaults bf16 on, which transformers refuses on a Mac VM without MPS (GitHub's runners).
            "def cpu(c): return dataclasses.make_dataclass(c.__name__, [('bf16', bool, False)], bases=(c,))\n"
            "RealDPO = cpu(trl.DPOConfig)\n"
            "RealGRPO = cpu(trl.GRPOConfig)\n"
            "RealConfigs = {s: cpu(getattr(trl, s + 'Config')) for s in ('ORPO', 'KTO', 'CPO') if hasattr(trl, s + 'Config')}\n"
        ),
    )


@_needs_real_mlx
def test_trl_star_import_exposes_the_mappings_without_trl_installed():
    _run_shim_probe(
        """
assert trl.__version__.endswith("unsloth-mlx"), trl.__version__
namespace = {}
exec("from trl import *", namespace)
for stem in ("SFT", "DPO", "ORPO", "KTO", "GRPO", "CPO"):
    assert namespace[stem + "Trainer"] is getattr(trl, stem + "Trainer"), stem
    assert namespace[stem + "Config"] is getattr(trl, stem + "Config"), stem
""",
        before_unsloth = "import sys\nsys.modules['trl'] = None\n",
    )
