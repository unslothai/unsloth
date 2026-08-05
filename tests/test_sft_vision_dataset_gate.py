"""Tests _backport_vision_dataset_gate in rl.py against REAL TRL sources.

TRL 0.22.x decides "skip dataset preparation" and "use the vision collator"
from `_is_vlm` (the model) alone, so a VLM fine-tuned on a text-only dataset
reaches the trainer with a raw `text` column and no tokenized ones, and
transformers strips every column:

    ValueError: No columns in the dataset match the model's forward method
    signature ... The following columns have been ignored: [text]

Magistral_(24B)-Reasoning-Conversational hits this; it pins trl==0.22.2.
TRL 0.24.0+ keys the same decisions off `_is_vision_dataset`, back-ported here.

The patch is textual, so the tests run it over the installed TRL's
sft_trainer.py plus a checked-in 0.22.2 excerpt and require the result to
still parse. No GPU, no network.
"""

import ast
import importlib.util
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RL_PY = REPO_ROOT / "unsloth" / "models" / "rl.py"


def _load_backport():
    """Grab the helper without importing rl.py (which needs trl at import)."""
    src = RL_PY.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_backport_vision_dataset_gate":
            ns = {}
            exec(ast.get_source_segment(src, node), ns)
            return ns["_backport_vision_dataset_gate"]
    raise AssertionError("_backport_vision_dataset_gate not found in rl.py")


backport = _load_backport()

# The three decision points, verbatim from trl 0.22.2 sft_trainer.py.
TRL_022_EXCERPT = textwrap.dedent("""\
    class SFTTrainer:
        def __init__(self, train_dataset, args, data_collator, model):
            dataset_sample = next(iter(train_dataset))
            if args.completion_only_loss is None:
                self.completion_only_loss = "prompt" in dataset_sample
            if data_collator is None and not self._is_vlm:
                data_collator = DataCollatorForLanguageModeling()
            elif data_collator is None and self._is_vlm:
                data_collator = DataCollatorForVisionLanguageModeling()
            skip_prepare_dataset = (
                args.dataset_kwargs is not None and args.dataset_kwargs.get("skip_prepare_dataset", False) or self._is_vlm
            )
            if not skip_prepare_dataset:
                train_dataset = self._prepare_dataset(train_dataset)
""")

# TRL 0.24.0+ already computes the flag itself.
TRL_MODERN_EXCERPT = textwrap.dedent("""\
    class SFTTrainer:
        def __init__(self, train_dataset, args, data_collator, model):
            dataset_sample = next(iter(train_dataset))
            self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample
            if data_collator is None and not self._is_vision_dataset:
                data_collator = DataCollatorForLanguageModeling()
""")


def test_patches_all_three_decision_points_on_022():
    out = backport(TRL_022_EXCERPT)
    assert (
        'self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample' in out
    )
    assert "if data_collator is None and not (self._is_vlm and self._is_vision_dataset):" in out
    assert "elif data_collator is None and self._is_vlm and self._is_vision_dataset:" in out
    assert "or (self._is_vlm and self._is_vision_dataset)" in out
    # Every bare `or self._is_vlm` gate must be gone.
    assert 'skip_prepare_dataset", False) or self._is_vlm\n' not in out


def test_patched_source_still_parses():
    ast.parse(backport(TRL_022_EXCERPT))


def test_modern_trl_is_untouched():
    assert backport(TRL_MODERN_EXCERPT) == TRL_MODERN_EXCERPT


def test_idempotent():
    once = backport(TRL_022_EXCERPT)
    assert backport(once) == once


def test_unrecognised_source_is_returned_unchanged():
    other = "class SFTTrainer:\n    def __init__(self):\n        pass\n"
    assert backport(other) == other


def _installed_trl_sft_source():
    try:
        # find_spec imports the parents, so a missing trl raises rather than
        # returning None, and the marker would fail collection instead of skip.
        spec = importlib.util.find_spec("trl.trainer.sft_trainer")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).read_text(encoding = "utf-8")


@pytest.mark.skipif(_installed_trl_sft_source() is None, reason = "trl not installed")
def test_installed_trl_source_survives_the_patch():
    src = _installed_trl_sft_source()
    out = backport(src)
    ast.parse(out)  # must stay valid whether or not it was patched
    if 'self._is_vision_dataset = "image" in dataset_sample' in src:
        assert out == src, "modern TRL must not be rewritten"
    else:
        assert "self._is_vision_dataset" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
