# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The vision run, and the one way it goes green while testing nothing.

A "vision run" that never puts an image on the GPU is a text run in a costume.
It trains, its loss falls, its adapter updates, and every assertion a text leg
makes passes. TRL will produce exactly that state if
`remove_unused_columns=False` is dropped, because the image column is removed
before the collator ever sees it.

So the rules are read off a REAL collated batch, and the guards below are
calibrated to catch the costume rather than the crash.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "t4_smoke"
sys.path.insert(0, str(PAYLOAD))

from run_vision_t4 import build_conversations, vision_failures  # noqa: E402

SRC = (PAYLOAD / "run_vision_t4.py").read_text(encoding = "utf-8")


def _args(**over):
    base = dict(
        max_steps = 2,
        require_vision_lora = True,
        export = False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _good(**over):
    result = {
        "pixels": {
            "columns": ["input_ids", "labels", "pixel_values"],
            "pixel_keys": ["pixel_values"],
            "pixel_sizes": {"pixel_values": {"numel": 602112, "shape": [1, 3, 448, 448]}},
        },
        "vision_lora": {"vision_module_count": 48, "language_modules": 112},
        "metrics": [{"step": 1, "loss": 3.0}, {"step": 2, "loss": 2.0}],
        "adapter_update": {"before": 0.0, "after": 91.2, "tensors": 864, "changed": True},
        "generated": "\\frac{1}{2}",
    }
    result.update(over)
    return result


def test_the_measured_shape_passes():
    assert vision_failures(_good(), _args()) == []


def test_a_batch_with_no_pixels_is_the_headline_failure():
    """The costume. Everything else about this run is healthy."""
    broken = vision_failures(
        _good(pixels = {"columns": ["input_ids", "labels"], "pixel_keys": [], "pixel_sizes": {}}),
        _args(),
    )
    assert broken and "no pixel tensor at all" in broken[0]


def test_an_empty_pixel_tensor_is_a_failure():
    broken = vision_failures(
        _good(pixels = {"pixel_sizes": {"pixel_values": {"numel": 0, "shape": [0]}}}),
        _args(),
    )
    assert broken and "pixel tensors are empty" in broken[0]


def test_a_lora_that_never_reached_the_vision_tower_is_a_failure():
    """`finetune_vision_layers=True` is a request, not a result."""
    broken = vision_failures(
        _good(vision_lora = {"vision_module_count": 0, "language_modules": 112}), _args()
    )
    assert broken and "no LoRA module landed" in broken[0]


def test_the_vision_lora_rule_can_be_turned_off_for_a_language_only_run():
    assert (
        vision_failures(
            _good(vision_lora = {"vision_module_count": 0}), _args(require_vision_lora = False)
        )
        == []
    )


def test_an_adapter_that_did_not_move_is_a_failure():
    """Starts at exactly zero by construction, so any movement is a real
    optimizer step rather than a tolerance question."""
    broken = vision_failures(
        _good(adapter_update = {"before": 0.0, "after": 0.0, "tensors": 864, "changed": False}),
        _args(),
    )
    assert broken and "did not move" in broken[0]


def test_empty_or_missing_generation_is_a_failure():
    assert vision_failures(_good(generated = ""), _args())
    assert vision_failures(_good(generated = None), _args())


def test_a_short_step_count_is_a_failure():
    broken = vision_failures(_good(metrics = [{"step": 1, "loss": 3.0}]), _args())
    assert broken and "expected 2 logged steps" in broken[0]


def test_a_non_finite_loss_is_a_failure():
    broken = vision_failures(
        _good(metrics = [{"step": 1, "loss": float("nan")}, {"step": 2, "loss": 2.0}]), _args()
    )
    assert broken and "non-finite" in broken[0]


def test_the_export_rule_only_fires_when_the_export_was_requested():
    assert vision_failures(_good(), _args(export = False)) == []
    broken = vision_failures(_good(), _args(export = True))
    assert broken and "failed" in broken[0]
    assert (
        vision_failures(
            _good(export = {"ok": True, "files": [{"name": "model.safetensors", "mb": 1200.0}]}),
            _args(export = True),
        )
        == []
    )


def test_an_export_that_reported_ok_and_wrote_nothing_is_a_failure():
    """The gpt-oss lesson: an export can succeed and leave no file anywhere."""
    broken = vision_failures(
        _good(export = {"ok": True, "files": [], "dir": "/tmp/x"}), _args(export = True)
    )
    assert broken and "wrote nothing" in broken[0]


def test_the_conversation_shape_matches_the_notebook():
    """Built from the notebook's own convert_to_conversation. A different shape
    trains something no notebook produces, which tests the leg."""
    rows = build_conversations([{"image": "IMG", "text": "x^2"}])
    assert len(rows) == 1
    messages = rows[0]["messages"]
    assert messages[0]["role"] == "user"
    kinds = [part["type"] for part in messages[0]["content"]]
    assert "image" in kinds and "text" in kinds
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["text"] == "x^2"


def test_the_four_settings_vision_training_needs_are_all_present():
    """Dropping `remove_unused_columns = False` is the exact edit that turns
    this into a text run: TRL removes the image column before the collator sees
    it, and nothing raises."""
    assert "remove_unused_columns = False" in SRC
    assert 'dataset_text_field = ""' in SRC
    assert 'dataset_kwargs = {"skip_prepare_dataset": True}' in SRC
    assert "UnslothVisionDataCollator(model, tokenizer)" in SRC


def test_the_pixel_evidence_is_read_before_training():
    """After `trainer.train()` the dataloader has been consumed, and a
    re-created one is not necessarily the object the trainer used."""
    pixels_at = SRC.index('result["pixels"] = pixel_evidence(trainer)')
    train_at = SRC.index("stats = trainer.train()")
    assert pixels_at < train_at


def test_the_export_does_not_land_in_the_artifact_volume():
    """/kaggle/working is 21GB and a merged 2B is a meaningful fraction of it."""
    assert 'tempfile.mkdtemp(prefix = "vision_export_")' in SRC
    # Scoped to the export BLOCK. A whole-file search matches args.outdir in
    # main(), where it is correct, and the assertion would fail for a reason
    # that has nothing to do with the export.
    # Anchored on the mkdtemp rather than on `if args.export:`, because that
    # string appears FIRST in vision_failures() and the naive split lands in
    # the wrong function -- which is how this assertion failed the first time.
    block = SRC.split("export_dir = tempfile.mkdtemp", 1)[1].split('result["export"] = record', 1)[
        0
    ]
    assert "args.outdir" not in block, "the merged model is written into the artifact volume"
    assert "model.save_pretrained_merged(export_dir, tokenizer)" in block


def test_the_train_dataset_is_a_dataset_and_its_images_stay_pil():
    """Two failures in one, both measured rather than guessed.

    TRL 1.x rejects a plain list, which is what the notebook passes:

        TypeError: `train_dataset` must be a `Dataset` or `IterableDataset`,
        got `list`

    And the obvious fix corrupts the data. `Dataset.from_list` Arrow-encodes a
    nested PIL object into a `{bytes, path}` DICT on the way back out, so the
    collator receives something that is not an image and nothing says so.

    `with_transform` applies at access time, keeps the column's Image feature,
    and still satisfies TRL's type check.
    """
    from datasets import Dataset as HFDataset
    from PIL import Image

    from run_vision_t4 import conversation_dataset

    base = HFDataset.from_dict({"image": [Image.new("RGB", (8, 8))], "text": ["x^2"]})
    built = conversation_dataset(base)
    assert isinstance(built, HFDataset), "TRL 1.x rejects anything else"
    image = built[0]["messages"][0]["content"][1]["image"]
    assert isinstance(image, Image.Image), (
        f"the image came back as {type(image).__name__}, which is the silent "
        f"Arrow corruption Dataset.from_list produces"
    )


def test_from_list_would_have_corrupted_the_images():
    """The negative control. Without it, the test above passes for a
    `with_transform` that happens to work and says nothing about why the
    obvious alternative was rejected."""
    from datasets import Dataset as HFDataset
    from PIL import Image

    from run_vision_t4 import build_conversations

    rows = build_conversations([{"image": Image.new("RGB", (8, 8)), "text": "x^2"}])
    naive = HFDataset.from_list(rows)
    image = naive[0]["messages"][0]["content"][1]["image"]
    assert not isinstance(image, Image.Image), (
        "from_list now preserves PIL, so the with_transform indirection may no "
        "longer be needed; re-check before simplifying"
    )


def test_a_marker_that_matched_nothing_is_refused_rather_than_answered_no():
    """Measured on `unsloth-probe-vision-train-r2-8ed253`, and it named the
    wrong defect. PEFT calls these parameters `lora_B`, with a capital B; the
    marker was matched against the raw name, so it matched none of the 864 of
    them and summed to zero both before AND after. The run had trained
    perfectly well -- loss 1.13 -> 0.56, a merged 4.3 GB export -- and the
    report said the optimizer applied nothing.

    Zero over zero tensors and zero over 864 tensors are opposite findings and
    read identically, which is why the count is carried.
    """
    broken = vision_failures(
        _good(adapter_update = {"before": 0.0, "after": 0.0, "tensors": 0, "changed": False}),
        _args(),
    )
    assert broken, "a question that was never asked must not pass"
    assert "never asked" in broken[0], broken
    assert "did not move" not in broken[0], (
        "reporting an unmatched marker as an untrained adapter sends the "
        "reader after the wrong bug, which is what happened on hardware"
    )


def test_adapter_sum_finds_the_capital_b_peft_names():
    """Drives the REAL function, because every rule above is fed a dict written
    by hand and none of them execute the code that produces it. That is exactly
    how the capital-B bug reached hardware."""
    import torch

    from run_vision_t4 import adapter_sum

    class _Stub:
        def named_parameters(self):
            # The names PEFT actually emits, capitals and all.
            yield (
                "base_model.model.visual.blocks.0.attn.qkv.lora_A.default.weight",
                torch.ones(2, 2),
            )
            yield (
                "base_model.model.visual.blocks.0.attn.qkv.lora_B.default.weight",
                torch.full((2, 2), 3.0),
            )
            yield (
                "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
                torch.full((2, 2), 1.0),
            )

    got = adapter_sum(_Stub())
    assert got["tensors"] == 2, got
    assert got["sum"] == 16.0, got


def test_the_leg_actually_DRIVES_the_vision_run():
    """The gap this closes was live for two rounds: `run_vision_t4.py` was in
    the leg's `files` and nothing ever executed it, so Vision_FLA_compile
    trained TEXT, asserted kernels, and shipped a payload it never ran.

    A file that is copied and not run is the quietest kind of coverage there
    is: every guard in this module passed, on a leg where the image path was
    dead.
    """
    import sys as _sys

    _sys.path.insert(0, str(ROOT / ".github" / "scripts" / "kaggle_t4_ci"))
    import legs  # noqa: E402

    leg = legs.LEGS["vision_fla_compile"]
    assert "--vision-run" in leg.args, "the leg ships the payload but never runs it"
    assert "run_vision_t4.py" in leg.files
    assert (
        "--export-gguf" in leg.args
    ), "the merged vision export is the half the text path cannot exercise"


def test_the_parent_spawns_the_vision_run_after_the_cycles():
    """Two 4bit models resident at once on a 14.56GB card is how a leg becomes
    an OOM blamed on the thing it was testing. It is also what keeps a vision
    failure from reading as a text-training one."""
    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert '"run_vision_t4.py"' in src
    cycles_at = src.index("runs.append(json.loads(report_file.read_text")
    spawn_at = src.index('"run_vision_t4.py"')
    assert cycles_at < spawn_at


def test_a_vision_run_that_wrote_no_report_is_a_failure_not_a_silence():
    """ "the vision run did not happen" and "the vision run passed" are opposite
    outcomes, and an absent report must not read as the second."""
    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    assert '"the vision process wrote no report"' in src
    assert 'failures += report["vision_failures"]' in src


def test_the_vision_step_count_is_pinned_low_rather_than_inherited():
    """A vision step on a T4 is ~100s (317.9s for three, measured on
    unsloth-probe-vision-train-r3). Inheriting the text side's --max-steps
    would quietly add half an hour to the leg."""
    src = (PAYLOAD / "run_t4_smoke.py").read_text(encoding = "utf-8")
    spawn = src[src.index('"run_vision_t4.py"') :]
    spawn = spawn[: spawn.index("subprocess.run(vision_cmd)")]
    # Whitespace-insensitive: the repo's formatter reflows this list to one
    # argument per line, and a guard matching the unformatted spelling goes red
    # on a reformat rather than on a regression. That has now happened twice in
    # this payload, so it is worth doing by default.
    flat = "".join(spawn.split())
    assert '"--max-steps","3",' in flat
    assert "args.max_steps" not in flat, "the text step count must not reach the vision run"
