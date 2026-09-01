# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Guard the grad-accumulation loss-normalisation contract across TRL versions.

Three components each decide "is this loss already token-count normalised?", and
they must agree:

  * unsloth_zoo `_unsloth_get_batch_samples` decides from the forward signature.
  * The loss divides by num_items_in_batch when it is not None. Both
    `unsloth_fused_ce_loss` and TRL's `_chunked_cross_entropy_loss` do this
    without consulting `model_accepts_loss_kwargs`.
  * transformers `training_step` divides by grad-accum when
    `not self.model_accepts_loss_kwargs or num_items_in_batch is None`.

When a model class sets `accepts_loss_kwargs = False` (gemma3, qwen-vl,
paligemma, glm4v) and the loss still divides by the token count, loss and grads
are silently scaled 1/GA. Nothing raises; the effective LR is just GA times too
small. Regressed when TRL 1.7.0 defaulted SFT to "chunked_nll" (trl#5846):
clean on 0.22.2-1.6.0, reproducible from 1.4.0 by opting in explicitly.

Source/AST checks only, no GPU and no downloads, so they run in the CPU job that
already exercises TRL latest and TRL git main.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import sys
import textwrap
from pathlib import Path


os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import pytest


# daily-fresh-fetch collects this directory with only pytest installed.
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level = True)

# Unsloth refuses to import without a torch accelerator, so the GPU-less runner needs the same spoof the sibling CPU
_SPOOF_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SPOOF_DIR))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()


def test_sft_loss_type_default_is_nll_after_unsloth_patch():
    """chunked_nll bypasses the forward (fused CE never runs) and double-divides."""
    import unsloth  # noqa: F401  must precede trl
    import trl

    if not hasattr(trl.SFTConfig, "loss_type"):
        pytest.skip("this TRL has no SFTConfig.loss_type")

    cfg = trl.SFTConfig(output_dir = "unused")
    assert cfg.loss_type == "nll", (
        f"SFTConfig.loss_type resolved to {cfg.loss_type!r}, expected 'nll'. "
        "If TRL changed its default again, update the sft_trainer replacement in "
        "unsloth/models/rl.py -- do NOT add loss_type to the global replacements "
        "dict, it is an unrelated field in DPO/KTO/GRPO."
    )


def test_loss_type_replacement_did_not_leak_to_other_trainers():
    """loss_type is an unrelated field in DPO/KTO/GRPO; the global dict hits all."""
    import unsloth  # noqa: F401
    import trl

    expected = {"DPOConfig": ["sigmoid"], "KTOConfig": "kto", "GRPOConfig": "bnpo"}
    for name, want in expected.items():
        cfg_cls = getattr(trl, name, None)
        if cfg_cls is None or not hasattr(cfg_cls, "loss_type"):
            continue
        got = cfg_cls(output_dir = "unused").loss_type
        assert got == want, (
            f"{name}.loss_type is {got!r}, expected {want!r}. A loss_type "
            "replacement leaked out of the sft_trainer branch in rl.py."
        )


def test_explicit_loss_type_still_wins():
    """Pinning a default must not take the choice away from the user."""
    import unsloth  # noqa: F401
    import trl

    if not hasattr(trl.SFTConfig, "loss_type"):
        pytest.skip("this TRL has no SFTConfig.loss_type")
    cfg = trl.SFTConfig(output_dir = "unused", loss_type = "chunked_nll")
    assert cfg.loss_type == "chunked_nll", "explicit loss_type was clobbered"


def _pristine_sft_config_cls():
    """TRL's own SFTConfig, not the generated subclass patching rebinds over it."""
    import trl

    # Go by the marker rather than the name:
    cls = trl.SFTConfig
    while "_unsloth_patched_rl_config" in cls.__dict__ or cls.__name__.startswith("Unsloth"):
        cls = cls.__mro__[1]
    return cls


def test_pristine_trl_sft_config_default_is_nll_too():
    """`from trl import SFTConfig` before `import unsloth` keeps TRL's own class.

    Patching only rebinds the module aliases, so that caller never sees the
    generated subclass and would still build a chunked_nll config and hand it to
    the patched trainer. The same ordering is covered by the padding-free tests.
    """
    import unsloth  # noqa: F401  must precede trl

    pristine = _pristine_sft_config_cls()
    if not hasattr(pristine, "loss_type"):
        pytest.skip("this TRL has no SFTConfig.loss_type")

    got = pristine(output_dir = "unused").loss_type
    assert got == "nll", (
        f"pristine {pristine.__name__}.loss_type resolved to {got!r}, expected "
        "'nll'. _pin_pristine_sft_loss_type in unsloth/models/rl.py stopped "
        "reaching TRL's own class, so a pre-unsloth `from trl import SFTConfig` "
        "still double-normalises the loss by 1/GA."
    )


def test_pristine_trl_sft_config_keeps_an_explicit_loss_type():
    """Pinning the pristine default must not take the choice away either."""
    import unsloth  # noqa: F401

    pristine = _pristine_sft_config_cls()
    if not hasattr(pristine, "loss_type"):
        pytest.skip("this TRL has no SFTConfig.loss_type")

    for wanted in ("chunked_nll", "dft"):
        got = pristine(output_dir = "unused", loss_type = wanted).loss_type
        assert got == wanted, f"explicit loss_type {wanted!r} was clobbered to {got!r}"


def test_dataclass_field_default_is_nll_for_hfargumentparser():
    """`HfArgumentParser` reads the field, not the `__init__` default.

    It builds one argparse argument per `dataclasses.fields()` entry and always
    passes the value through, so a field left at TRL's unresolved `None` sends
    `loss_type = None` into `__post_init__` and comes back out as chunked_nll
    however the `__init__` default reads.
    """
    import dataclasses

    import unsloth  # noqa: F401

    pristine = _pristine_sft_config_cls()
    if not hasattr(pristine, "loss_type"):
        pytest.skip("this TRL has no SFTConfig.loss_type")

    import trl

    for cls in (pristine, trl.SFTConfig):
        field = {f.name: f for f in dataclasses.fields(cls)}["loss_type"]
        assert field.default == "nll", (
            f"{cls.__name__}.loss_type field default is {field.default!r}, "
            "expected 'nll'. HfArgumentParser and any other dataclass-driven "
            "entry point would pass that default through and land on chunked_nll."
        )


# The normalisation predicates themselves --------------------------------------------------------------------------
def _divides_by_num_items(fn) -> bool:
    """True when the source contains a division by num_items_in_batch/n_items."""
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    names = {"num_items_in_batch", "n_items"}
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            for sub in ast.walk(node.right):
                if isinstance(sub, ast.Name) and sub.id in names:
                    return True
    return False


def test_transformers_training_step_still_keys_off_model_accepts_loss_kwargs():
    """If upstream changes this predicate, our whole reconciliation is stale."""
    from transformers import Trainer

    try:
        source = inspect.getsource(Trainer.training_step)
    except (OSError, TypeError):
        pytest.skip("Trainer.training_step source unavailable")

    assert "model_accepts_loss_kwargs" in source, (
        "transformers' training_step no longer references model_accepts_loss_kwargs. "
        "The grad-accum normalisation contract changed upstream; re-check "
        "unsloth_zoo's _unsloth_get_batch_samples against the new predicate."
    )
    assert (
        "num_items_in_batch" in source
    ), "transformers' training_step no longer references num_items_in_batch."


def test_trl_chunked_ce_divides_by_num_items_without_consulting_the_flag():
    """Fires if TRL starts gating this, at which point the pin can be dropped."""
    trl_sft = pytest.importorskip("trl.trainer.sft_trainer")

    fn = getattr(trl_sft, "_chunked_cross_entropy_loss", None)
    if fn is None:
        pytest.skip("this TRL has no _chunked_cross_entropy_loss")

    assert _divides_by_num_items(fn), (
        "TRL's _chunked_cross_entropy_loss no longer divides by num_items_in_batch. "
        "If TRL now gates this on model_accepts_loss_kwargs, the loss_type pin in "
        "unsloth/models/rl.py may no longer be needed -- re-measure before removing."
    )

    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return
    assert "model_accepts_loss_kwargs" not in source, (
        "TRL's chunked CE now consults model_accepts_loss_kwargs. Good news: "
        "re-evaluate whether unsloth still needs to pin loss_type='nll'."
    )


def test_unsloth_fused_ce_has_the_same_num_items_contract():
    """Ours divides by n_items too, which is why the compiled-path flag matters."""
    ce = pytest.importorskip("unsloth_zoo.fused_losses.cross_entropy_loss")

    fn = getattr(ce, "unsloth_fused_ce_loss", None)
    if fn is None:
        pytest.skip("unsloth_fused_ce_loss not present")
    source = inspect.getsource(fn)
    assert "n_items" in source, (
        "unsloth_fused_ce_loss no longer takes n_items; the normalisation "
        "contract changed on our side and rl.py's loss_type pin should be re-checked."
    )


def test_unsloth_get_batch_samples_is_installed_and_shaped_as_expected():
    """_utils.py raises NotImplementedError on this shape; catch it before train()."""
    from transformers import Trainer

    fn = getattr(Trainer, "get_batch_samples", None)
    if fn is None:
        pytest.skip("Trainer has no get_batch_samples")

    try:
        source = inspect.getsource(fn).strip()
    except (OSError, TypeError):
        pytest.skip("get_batch_samples source unavailable")

    assert source.endswith("return batch_samples, num_items_in_batch"), (
        "get_batch_samples no longer ends in the expected 2-tuple return. "
        "unsloth/models/_utils.py raises NotImplementedError on this exact check."
    )


def test_rl_py_scopes_loss_type_to_sft_trainer():
    """AST guard: no loss_type replacement outside an `if trainer_file ==` branch."""
    from unsloth.models import rl

    source = inspect.getsource(rl)
    tree = ast.parse(source)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "replacements" for t in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
        if "loss_type" not in keys:
            continue
        # A loss_type entry is only legitimate inside an `if trainer_file == ...` branch.
        guarded = False
        for parent in ast.walk(tree):
            if not isinstance(parent, ast.If):
                continue
            if node not in [
                n
                for b in (parent.body, parent.orelse)
                for n in ast.walk(ast.Module(body = b, type_ignores = []))
            ]:
                continue
            if "trainer_file" in ast.dump(parent.test):
                guarded = True
                break
        if not guarded:
            offenders.append(keys)

    assert not offenders, (
        f"found an unguarded `loss_type` replacement: {offenders}. It must live "
        'inside an `if trainer_file == "...":` branch -- the global replacements '
        "dict is applied by regex to every generated config, and loss_type is a "
        "real field in DPOConfig, KTOConfig and GRPOConfig."
    )
