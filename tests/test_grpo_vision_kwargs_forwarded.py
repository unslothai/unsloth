# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Both GRPO logprob paths forward every multimodal key TRL produced: two hand written
four-key lists dropped spatial_shapes, num_tiles, image_position_ids, and pixel_values for
models without image_grid_thw. unslothai/unsloth#6960."""

import inspect
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _zoo_vision_helpers(*names):
    """The multi image helpers ship with unsloth_zoo. An unsloth_zoo installed from before
    they landed has none of them, and the static gates in this file already prove this repo
    asks for them and fails loudly without them, so behaviour that can only be driven
    through the zoo is skipped rather than reported as this repo being broken."""
    zoo = pytest.importorskip("unsloth_zoo.rl_replacements")
    missing = [name for name in names if not hasattr(zoo, name)]
    if missing:
        pytest.skip(f"the installed unsloth_zoo has no {', '.join(missing)}")
    return tuple(getattr(zoo, name) for name in names)


def _read_source() -> str:
    with open(SOURCE_PATH, "r", encoding = "utf-8") as fh:
        return fh.read()


def test_the_key_tuple_lives_in_one_place():
    src = _read_source()
    assert "grpo_get_vision_inputs" in src
    assert src.count("vision_inputs.get(") >= 10
    assert src.count("def _unsloth_grpo_vision_inputs(") == 1
    assert src.count("_unsloth_grpo_vision_inputs(kwargs)") == 1
    assert src.count("_unsloth_grpo_vision_inputs(inputs)") == 1
    for gone in (
        'kwargs.get("image_grid_thw", None)',
        'kwargs.get("num_images", None)',
        'kwargs.get("pixel_values", None)',
    ):
        assert gone not in src, gone


def test_no_grad_pass_uses_the_shared_chunker():
    from unsloth.models.rl_replacements import grpo_trainer__get_per_token_logps_and_entropies

    patched = grpo_trainer__get_per_token_logps_and_entropies(
        "_get_per_token_logps_and_entropies", ""
    )
    assert "grpo_vision_chunks" in patched
    assert "**vision_chunk" in patched
    for gone in (
        "pixel_values_chunks",
        "image_grid_thw_chunks",
        "pixel_attention_mask_chunks",
        "image_sizes_chunks",
        "_extra_vision_kwargs",
    ):
        assert gone not in patched, gone


def test_compute_loss_hands_the_whole_set_to_the_gradient_pass():
    from unsloth.models.rl_replacements import grpo_trainer_compute_loss

    patched = grpo_trainer_compute_loss("compute_loss", "")
    assert "**_vision_inputs" in patched
    assert patched.count("**_vision_inputs") == 2, "both the modern and the legacy call"
    # A hand named kwarg here would reach the logprob pass but not the loss.
    for call in patched.split("grpo_accumulated_loss(")[1:]:
        body = call.split("\n                )")[0]
        for gone in (
            "pixel_values =",
            "image_grid_thw =",
            "pixel_attention_mask =",
            "image_sizes =",
            "num_images =",
            "token_type_ids =",
        ):
            assert gone not in body, gone


def test_both_paths_call_the_same_helper():
    grpo_accumulated_loss, grpo_vision_chunks = _zoo_vision_helpers(
        "grpo_accumulated_loss", "grpo_vision_chunks"
    )

    grad_source = inspect.getsource(grpo_accumulated_loss)
    assert "grpo_vision_chunks" in grad_source
    from unsloth_zoo import rl_replacements as zoo_rl

    assert zoo_rl.RL_REPLACEMENTS["grpo_vision_chunks"] is grpo_vision_chunks


def test_old_zoo_without_the_helper_fails_loudly_for_vision_runs():
    from unsloth.models.rl_replacements import grpo_trainer__get_per_token_logps_and_entropies

    patched = grpo_trainer__get_per_token_logps_and_entropies(
        "_get_per_token_logps_and_entropies", ""
    )
    assert "Please upgrade unsloth_zoo" in patched
    assert "grpo_vision_chunks" in patched


def test_lfm2vl_keys_reach_the_forward_kwargs():
    import torch

    grpo_get_vision_inputs, grpo_vision_chunks = _zoo_vision_helpers(
        "grpo_get_vision_inputs", "grpo_vision_chunks"
    )

    num_tiles = [2, 3]
    total_tiles = sum(num_tiles)
    inputs = {
        "advantages": torch.zeros(2),
        "pixel_values": torch.randn(total_tiles, 3, 16, 16),
        "pixel_attention_mask": torch.ones(total_tiles, 16, 16),
        "spatial_shapes": torch.tensor([[2, 2]] * total_tiles),
        "num_tiles": num_tiles,
        "num_images": [1, 1],
    }
    chunks = grpo_vision_chunks(grpo_get_vision_inputs(inputs), 2, 1)
    assert [c["pixel_values"].shape[0] for c in chunks] == [2, 3]
    assert [c["spatial_shapes"].shape[0] for c in chunks] == [2, 3]
    assert [c["pixel_attention_mask"].shape[0] for c in chunks] == [2, 3]


def test_legacy_fallback_list_matches_the_zoo_tuple():
    import ast
    import re

    (GRPO_VISION_KEYS,) = _zoo_vision_helpers("GRPO_VISION_KEYS")

    src = _read_source()
    match = re.search(
        r"key: get\(key, None\)\s*for key in (\([^)]*\))",
        src,
        re.DOTALL,
    )
    assert match, "the old-zoo fallback key list moved; update this gate"
    fallback = ast.literal_eval(match.group(1))
    assert set(fallback) == set(GRPO_VISION_KEYS), (
        "the fallback list and unsloth_zoo.GRPO_VISION_KEYS have drifted: "
        f"{sorted(set(GRPO_VISION_KEYS) - set(fallback))} missing"
    )


def test_generate_forward_wrapper_keeps_the_real_signature():
    """generate() validates kwargs against inspect.signature(self.forward)."""
    import inspect as _inspect

    import torch

    from unsloth.models.rl import _install_grpo_hidden_states_forward_wrapper

    class _Toy(torch.nn.Module):
        config = type("cfg", (), {"is_encoder_decoder": False})()

        def forward(
            self,
            input_ids = None,
            pixel_values = None,
            spatial_shapes = None,
            **kwargs,
        ):
            return input_ids

    model = _Toy()
    before = list(_inspect.signature(model.forward).parameters)
    installed = _install_grpo_hidden_states_forward_wrapper(model)
    assert installed, "the wrapper did not install on a model without hidden-state support"
    after = list(_inspect.signature(model.forward).parameters)
    assert after == before, f"signature lost: {before} became {after}"
    assert "pixel_values" in after and "spatial_shapes" in after


def test_the_wrapper_survives_accelerates_fp32_unwrap():
    """The signature must not be bought by making the wrapper removable: functools.wraps
    sets __wrapped__, which accelerate's extract_model_from_parallel(keep_fp32_wrapper =
    False) walks straight past this wrapper on every GRPO step."""
    import inspect as _inspect

    import torch
    from accelerate.utils.operations import convert_outputs_to_fp32
    from accelerate.utils.other import extract_model_from_parallel

    from unsloth.models.rl import _install_grpo_hidden_states_forward_wrapper

    class _Toy(torch.nn.Module):
        config = type("cfg", (), {"is_encoder_decoder": False})()

        def forward(
            self,
            input_ids = None,
            pixel_values = None,
            spatial_shapes = None,
            **kwargs,
        ):
            return input_ids

    model = _Toy()
    real_forward = model.forward
    model._original_forward = real_forward
    model.forward = convert_outputs_to_fp32(
        torch.autocast(device_type = "cuda", dtype = torch.bfloat16)(real_forward)
    )

    assert _install_grpo_hidden_states_forward_wrapper(model)
    assert getattr(model.forward, "_unsloth_grpo_hidden_states_forward_wrapped", False)
    assert not hasattr(
        model.forward, "__wrapped__"
    ), "the wrapper carries __wrapped__, so accelerate will unwrap straight past it"

    unwrapped = extract_model_from_parallel(model, keep_fp32_wrapper = False)

    assert getattr(
        unwrapped.forward, "_unsloth_grpo_hidden_states_forward_wrapped", False
    ) or getattr(
        getattr(unwrapped.forward, "__func__", None),
        "_unsloth_grpo_hidden_states_forward_wrapped",
        False,
    ), "accelerate removed the hidden-state wrapper"
    after = list(_inspect.signature(unwrapped.forward).parameters)
    assert "pixel_values" in after and "spatial_shapes" in after, after


def test_num_tiles_survives_the_output_dict_rewrite():
    """TRL 1.7.0 nested num_tiles inside the num_images block; the insert must go after it."""
    from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions

    source = (
        '        if "image_sizes" in forward_kwargs:\n'
        '            output["image_sizes"] = forward_kwargs["image_sizes"]\n'
        "        if images is not None:\n"
        '            output["num_images"] = num_images\n'
        "            if num_tiles is not None:\n"
        '                output["num_tiles"] = num_tiles\n'
        "        return output\n"
    )
    patched = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", source
    )
    lines = patched.splitlines()
    num_images_at = next(i for i, l in enumerate(lines) if 'output["num_images"]' in l)
    num_tiles_at = next(i for i, l in enumerate(lines) if 'output["num_tiles"]' in l)
    except_at = next((i for i, l in enumerate(lines) if l.strip() == "except NameError:"), None)
    assert except_at is not None, "the sampling logprob block was not inserted"
    assert num_images_at < num_tiles_at < except_at
    # still nested inside `if images is not None:`
    assert lines[num_tiles_at].startswith(" " * 16)


def test_the_trl_1_0_spelling_of_the_gemma_4_position_ids_is_forwarded():
    """TRL 1.0.x emits `pixel_position_ids`, 1.1.0 `image_position_ids`; both are live."""
    import torch

    GRPO_VISION_KEYS, grpo_get_vision_inputs, grpo_vision_chunks = _zoo_vision_helpers(
        "GRPO_VISION_KEYS", "grpo_get_vision_inputs", "grpo_vision_chunks"
    )

    assert "pixel_position_ids" in GRPO_VISION_KEYS
    assert "image_position_ids" in GRPO_VISION_KEYS

    for key in ("pixel_position_ids", "image_position_ids"):
        inputs = {
            "pixel_values": torch.randn(4, 3, 16, 16),
            key: torch.arange(4).unsqueeze(-1),
            "num_images": [1, 1, 1, 1],
        }
        chunks = grpo_vision_chunks(grpo_get_vision_inputs(inputs), 4, 2)
        assert [tuple(c) for c in chunks] == [("pixel_values", key)] * 2, chunks
        assert [c["pixel_values"].shape[0] for c in chunks] == [2, 2]
        assert [c[key].shape[0] for c in chunks] == [2, 2]
        other = "image_position_ids" if key == "pixel_position_ids" else "pixel_position_ids"
        assert all(other not in c for c in chunks)


def test_a_multi_image_row_indexes_the_position_ids_by_image():
    import torch

    grpo_get_vision_inputs, grpo_vision_chunks = _zoo_vision_helpers(
        "grpo_get_vision_inputs", "grpo_vision_chunks"
    )

    inputs = {
        "pixel_values": torch.randn(3, 3, 16, 16),
        "image_position_ids": torch.arange(3).unsqueeze(-1),
        "num_images": [2, 1],
    }
    chunks = grpo_vision_chunks(grpo_get_vision_inputs(inputs), 2, 1)
    assert [c["pixel_values"].shape[0] for c in chunks] == [2, 1]
    assert [c["image_position_ids"].tolist() for c in chunks] == [[[0], [1]], [[2]]]


def test_an_old_zoo_still_forwards_the_sample_indexed_keys():
    """An older unsloth_zoo must cost only image indexed slicing, not token_type_ids."""
    from unsloth.models.rl_replacements import grpo_trainer__get_per_token_logps_and_entropies

    patched = grpo_trainer__get_per_token_logps_and_entropies(
        "_get_per_token_logps_and_entropies", ""
    )
    assert "_unsloth_grpo_vision_inputs(kwargs)" in patched
    fallback = patched.split("if _grpo_vision_chunks is None:")[-1]
    assert '"token_type_ids", "mm_token_type_ids"' in fallback
    assert "vision_chunks = [{} for _ in input_ids_chunks]" not in patched


def test_the_wrapper_keeps_trls_logits_to_keep_probe_answerable():
    """GRPOTrainer.__init__ probes the forward signature for `logits_to_keep` once."""
    import inspect as _inspect

    import torch

    from unsloth.models.rl import _install_grpo_hidden_states_forward_wrapper

    class _Takes(torch.nn.Module):
        config = type("cfg", (), {"is_encoder_decoder": False})()

        def forward(
            self,
            input_ids = None,
            logits_to_keep = 0,
            **kwargs,
        ):
            return input_ids

    class _Refuses(torch.nn.Module):
        config = type("cfg", (), {"is_encoder_decoder": False})()

        def forward(
            self,
            input_ids = None,
            pixel_values = None,
            **kwargs,
        ):
            return input_ids

    takes, refuses = _Takes(), _Refuses()
    assert _install_grpo_hidden_states_forward_wrapper(takes)
    assert _install_grpo_hidden_states_forward_wrapper(refuses)
    assert "logits_to_keep" in _inspect.signature(takes.forward).parameters
    assert "logits_to_keep" not in _inspect.signature(refuses.forward).parameters


def test_the_multi_image_zoo_probe_does_not_rest_on_a_local_variable():
    """Grepping grpo_accumulated_loss for "num_images", now an unread local, faked it."""
    from unsloth.models.rl_replacements import grpo_trainer_compute_loss

    patched = grpo_trainer_compute_loss("compute_loss", "")
    gate = patched.split("_unsloth_requires_multi_image_zoo(num_images)")[1]
    gate = gate.split("Please upgrade ")[0]
    assert "from unsloth_zoo.rl_replacements import grpo_vision_chunks" in gate
    # probe before grep, or the grep still decides on a current zoo
    assert gate.index("grpo_vision_chunks") < gate.index("inspect.getsource")


def _gradient_zoo_gate():
    """The gradient path's zoo gate, as a runnable block taken out of the patched source.

    Extracted by AST rather than by string slicing so the case below RUNS the real check
    instead of restating it: the block is self contained (it reads `self` and `pixel_values`
    and imports the zoo itself), so executing it is the behaviour and not a description.
    """
    import ast
    import textwrap

    from unsloth.models.rl_replacements import grpo_trainer_compute_loss

    source = textwrap.dedent(grpo_trainer_compute_loss("compute_loss", ""))
    tree = ast.parse(source)
    blocks = [
        segment
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        for segment in [ast.get_source_segment(source, node) or ""]
        if "grpo_vision_chunks" in segment and "_unsloth_grpo_vision_zoo_checked" in segment
    ]
    assert blocks, "the gradient path has no zoo gate at all"
    # The innermost match: every enclosing `if` contains the gate's text too.
    return textwrap.dedent(min(blocks, key = len))


class _Trainer:
    pass


def _run_gate(block, *, pixel_values, zoo_module):
    import sys
    import types

    name = "unsloth_zoo.rl_replacements"
    previous = sys.modules.get(name)
    sys.modules[name] = zoo_module
    trainer = _Trainer()
    try:
        exec(compile(block, "<gate>", "exec"), {"self": trainer, "pixel_values": pixel_values})
    finally:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous
    return trainer


def test_the_gradient_path_refuses_an_old_zoo_without_the_chunker():
    """The no-grad gate does not cover the gradient call, and the default run never reaches it.

    With beta = 0 and num_iterations = 1 there are no reference or old logprobs to compute, so
    `_get_per_token_logps_and_entropies` -- where the other gate lives -- is never called. An
    older `grpo_accumulated_loss` takes arbitrary kwargs, ignores the ones it does not know and
    replaces `pixel_values` with None for a model that carries no `image_grid_thw`, so a vision
    run would have trained on the text alone and said nothing.
    """
    import types

    block = _gradient_zoo_gate()
    old_zoo = types.ModuleType("unsloth_zoo.rl_replacements")
    with pytest.raises(RuntimeError) as raised:
        _run_gate(block, pixel_values = object(), zoo_module = old_zoo)
    assert "upgrade unsloth_zoo" in str(raised.value)

    # A text-only run is untouched: there are no pixels to drop.
    _run_gate(block, pixel_values = None, zoo_module = old_zoo)

    # And a zoo that exports the chunker passes, once, and remembers it.
    new_zoo = types.ModuleType("unsloth_zoo.rl_replacements")
    new_zoo.grpo_vision_chunks = lambda *_a, **_k: None
    trainer = _run_gate(block, pixel_values = object(), zoo_module = new_zoo)
    assert trainer._unsloth_grpo_vision_zoo_checked is True
    # Checked once per trainer: the flag short-circuits a later step even on a broken zoo,
    # so the probe is not paid on every accumulation step.
    already = _Trainer()
    already._unsloth_grpo_vision_zoo_checked = True
    import sys

    sys.modules["unsloth_zoo.rl_replacements"] = old_zoo
    try:
        exec(compile(block, "<gate>", "exec"), {"self": already, "pixel_values": object()})
    finally:
        del sys.modules["unsloth_zoo.rl_replacements"]


def test_the_gradient_gate_runs_before_every_accumulated_loss_call():
    """Placement, since the block above proves only what it does once reached."""
    from unsloth.models.rl_replacements import grpo_trainer_compute_loss

    patched = grpo_trainer_compute_loss("compute_loss", "")
    gate = patched.index("_unsloth_grpo_vision_zoo_checked")
    calls = [
        index
        for index in range(len(patched))
        if patched.startswith("grpo_accumulated_loss(", index)
    ]
    assert calls, "no accumulated-loss call to gate"
    assert all(gate < index for index in calls), (gate, calls)


def _persisted_vision_output(forward_kwargs, output):
    """Run the block the output rewrite injects, on the batch dict it is given."""
    import textwrap

    from unsloth.models.rl_replacements import (
        _unsloth_grpo_vision_inputs,
        grpo_trainer__generate_and_score_completions,
    )

    source = (
        '        if "image_sizes" in forward_kwargs:\n'
        '            output["image_sizes"] = forward_kwargs["image_sizes"]\n'
        "        if images is not None:\n"
        '            output["num_images"] = num_images\n'
        "        return output\n"
    )
    patched = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", source
    )
    lines = patched.splitlines()
    start = next(i for i, line in enumerate(lines) if "_unsloth_vision_output = _unsloth" in line)
    stop = next(i for i, line in enumerate(lines) if "output[_vision_key] = _vision_value" in line)
    block = textwrap.dedent("\n".join(lines[start - 1 : stop + 1]))
    namespace = {
        "_unsloth_grpo_vision_inputs": _unsloth_grpo_vision_inputs,
        "forward_kwargs": forward_kwargs,
        "output": output,
    }
    exec(compile(block, "<output-extras>", "exec"), namespace)
    return output


def test_every_vision_kwarg_reaches_the_training_batch():
    """TRL 0.24.0's output block copies five vision keys, so spatial_shapes, num_tiles and
    the position ids never reach compute_loss and the gradient forward runs without them."""
    already = object()
    forward_kwargs = {
        "pixel_values": already,
        "spatial_shapes": "shapes",
        "num_tiles": [2, 3],
        "image_position_ids": "positions",
        "token_type_ids": "types",
    }
    output = {"pixel_values": already, "num_images": [1, 1]}
    persisted = _persisted_vision_output(forward_kwargs, output)
    for key, value in forward_kwargs.items():
        assert persisted[key] == value, key
    assert persisted["pixel_values"] is already, "an already saved key was overwritten"
    assert persisted["num_images"] == [1, 1]


def test_a_key_the_processor_did_not_produce_is_not_invented():
    persisted = _persisted_vision_output({"pixel_values": "pixels"}, {})
    assert persisted == {"pixel_values": "pixels"}, persisted


def test_the_tiles_of_a_sample_survive_the_shuffle_and_the_slice():
    """_prepare_inputs shuffles and slices the batch by sample index, so a tile indexed
    tensor has to be a list per sample while that happens."""
    import torch

    trl_utils = pytest.importorskip("trl.trainer.utils")
    from unsloth.models.rl_replacements import (
        _unsloth_grpo_split_vision_by_sample,
        _unsloth_grpo_unsplit_vision,
    )

    num_tiles = [2, 3, 1]
    owners = torch.tensor([0, 0, 1, 1, 1, 2])
    batch = {
        "prompt_ids": torch.arange(3).unsqueeze(-1),
        "advantages": torch.zeros(3),
        "num_images": [1, 1, 1],
        "num_tiles": num_tiles,
        "pixel_values": owners.reshape(-1, 1).float(),
        "spatial_shapes": owners.reshape(-1, 1).clone(),
        "pixel_attention_mask": owners.reshape(-1, 1).clone(),
    }
    torch.manual_seed(0)
    split = _unsloth_grpo_split_vision_by_sample(batch)
    assert isinstance(split["pixel_values"], list), "the tile axis was left flat"
    shuffled = trl_utils.shuffle_sequence_dict(split)
    chunks = trl_utils.split_tensor_dict(shuffled, 1)
    restored = _unsloth_grpo_unsplit_vision(trl_utils.unsplit_pixel_values_by_grid(chunks[0]))

    assert restored["pixel_values"].shape[0] == len(owners)
    at = 0
    seen = []
    for prompt, tiles in zip(restored["prompt_ids"].tolist(), restored["num_tiles"]):
        sample = prompt[0]
        seen.append(sample)
        assert tiles == num_tiles[sample], (sample, tiles)
        for key in ("pixel_values", "spatial_shapes", "pixel_attention_mask"):
            rows = restored[key][at : at + tiles].flatten().tolist()
            assert rows == [sample] * tiles, (key, sample, rows)
        at += tiles
    assert sorted(seen) == [0, 1, 2], seen


def test_the_gemma_position_ids_are_split_by_image_not_by_tile():
    import torch

    from unsloth.models.rl_replacements import _unsloth_grpo_split_vision_by_sample

    batch = {
        "num_images": [2, 1],
        "pixel_values": torch.arange(3).reshape(3, 1).float(),
        "image_position_ids": torch.arange(3).reshape(3, 1),
    }
    split = _unsloth_grpo_split_vision_by_sample(batch)
    assert [tensor.shape[0] for tensor in split["pixel_values"]] == [2, 1]
    assert [tensor.flatten().tolist() for tensor in split["image_position_ids"]] == [[0, 1], [2]]


def test_the_grid_layout_is_left_to_trl():
    """TRL splits image_grid_thw itself in every version that persists it; splitting it
    twice would hand the forward a list of lists."""
    import torch

    from unsloth.models.rl_replacements import _unsloth_grpo_split_vision_by_sample

    batch = {
        "num_images": [1, 1],
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
        "pixel_values": torch.zeros(8, 1),
    }
    assert _unsloth_grpo_split_vision_by_sample(batch) is batch


def test_a_padded_row_per_sample_is_left_alone():
    """Idefics and SmolVLM pad to one row per sample, which TRL slices correctly already."""
    import torch

    from unsloth.models.rl_replacements import _unsloth_grpo_split_vision_by_sample

    batch = {
        "num_images": [2, 2],
        "pixel_values": torch.zeros(2, 2, 3, 4),
    }
    assert _unsloth_grpo_split_vision_by_sample(batch) is batch


def test_a_batch_trl_already_split_is_not_split_again():
    import torch

    from unsloth.models.rl_replacements import _unsloth_grpo_split_vision_by_sample

    batch = {"num_images": [1, 1], "pixel_values": [torch.zeros(2, 1), torch.zeros(3, 1)]}
    assert _unsloth_grpo_split_vision_by_sample(batch) is batch


def test_the_counts_are_not_merged_as_if_they_were_tensors():
    import torch

    from unsloth.models.rl_replacements import _unsloth_grpo_unsplit_vision

    batch = {"num_images": [1, 1], "num_tiles": [2, 3], "pixel_values": [torch.zeros(2, 1)]}
    restored = _unsloth_grpo_unsplit_vision(batch)
    assert restored["num_images"] == [1, 1]
    assert restored["num_tiles"] == [2, 3]
    assert restored["pixel_values"].shape == (2, 1)


def test_prepare_inputs_splits_before_the_shuffle_and_merges_after_the_slice():
    from unsloth.models.rl_replacements import grpo_trainer__prepare_inputs

    source = (
        "    def _prepare_inputs(self, generation_batch):\n"
        "        if self._step % generate_every == 0:\n"
        "            generation_batch = self._generate_and_score_completions(generation_batch)\n"
        "            generation_batch = split_pixel_values_by_grid(generation_batch)\n"
        "            generation_batch = shuffle_sequence_dict(generation_batch)\n"
        "            generation_batches = split_tensor_dict(generation_batch, 2)\n"
        "            self._buffered_inputs = ["
        "unsplit_pixel_values_by_grid(batch) for batch in generation_batches]\n"
    )
    patched = grpo_trainer__prepare_inputs("_prepare_inputs", source)
    lines = patched.splitlines()
    split_at = next(i for i, l in enumerate(lines) if "split_pixel_values_by_grid(" in l)
    ours_at = next(i for i, l in enumerate(lines) if "_unsloth_grpo_split_vision_by_sample" in l)
    shuffle_at = next(i for i, l in enumerate(lines) if "shuffle_sequence_dict(" in l)
    assert split_at < ours_at < shuffle_at, (split_at, ours_at, shuffle_at)
    assert lines[ours_at].startswith(" " * 12), lines[ours_at]
    assert (
        "_unsloth_grpo_unsplit_vision(unsplit_pixel_values_by_grid(batch))" in patched
    ), "the slice is handed back still split"


def test_a_prepare_inputs_that_changed_shape_says_so():
    from unsloth.models import rl_replacements

    said = []
    original = rl_replacements._warn_once
    rl_replacements._warn_once = lambda where, message: said.append(where)
    try:
        patched = rl_replacements.grpo_trainer__prepare_inputs(
            "_prepare_inputs",
            "        generation_batch = split_pixel_values_by_grid(generation_batch)\n",
        )
    finally:
        rl_replacements._warn_once = original
    assert "_unsloth_grpo_split_vision_by_sample" not in patched
    assert said == ["grpo_split_vision_by_sample"], said


def test_both_halves_travel_with_the_generated_module():
    from unsloth.models.rl_replacements import RL_PRE_ITEMS

    pre = "\n".join(RL_PRE_ITEMS["grpo_trainer"])
    assert "def _unsloth_grpo_split_vision_by_sample(" in pre
    assert "def _unsloth_grpo_unsplit_vision(" in pre
