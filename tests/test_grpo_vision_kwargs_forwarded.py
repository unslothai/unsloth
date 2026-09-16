"""Both GRPO logprob paths forward every multimodal key TRL produced.

Regression cover for unslothai/unsloth#6960. The no-grad old/reference pass lives here,
in `grpo_trainer__get_per_token_logps_and_entropies`, and the gradient pass lives in
unsloth_zoo's `grpo_accumulated_loss`. They used to read two hand written lists of four
keys, so `spatial_shapes` (LFM2-VL), `num_tiles` (LFM2-VL, InternVL) and
`image_position_ids` (Gemma 4) reached neither, and `pixel_values` reached neither for any
model that does not emit `image_grid_thw`.
"""

import inspect
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _read_source() -> str:
    with open(SOURCE_PATH, "r", encoding = "utf-8") as fh:
        return fh.read()


def test_the_key_tuple_lives_in_one_place():
    """unsloth must not keep a second copy of the key list."""
    src = _read_source()
    assert "grpo_get_vision_inputs" in src
    # Every multimodal key is read off the shared mapping, in both the logprob pass and
    # compute_loss, so the tuple in unsloth_zoo is the only list of names.
    assert src.count("vision_inputs.get(") >= 10
    # And one collector for both paths, so the fallback names cannot be written twice.
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
    # The per-key chunk lists the old implementation zipped are gone.
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
    # Neither grpo_accumulated_loss call may name the multimodal kwargs by hand any more,
    # or a key added to the tuple would reach the logprob pass and not the loss.
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
    from unsloth_zoo.rl_replacements import grpo_accumulated_loss, grpo_vision_chunks

    grad_source = inspect.getsource(grpo_accumulated_loss)
    assert "grpo_vision_chunks" in grad_source
    # One object, so a slicing fix cannot land on one path only.
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
    """End to end on the shared chunker with LFM2-VL shaped inputs."""
    import torch
    from unsloth_zoo.rl_replacements import grpo_get_vision_inputs, grpo_vision_chunks

    num_tiles = [2, 3]
    total_tiles = sum(num_tiles)
    # What TRL's _generate_and_score_completions puts in the inputs dict for LFM2-VL.
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
    """The old-zoo fallback in compute_loss must name every key the tuple names."""
    import ast
    import re

    from unsloth_zoo.rl_replacements import GRPO_VISION_KEYS

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
    """transformers' generate validates kwargs against inspect.signature(self.forward).

    A bare (*args, **kwargs) wrapper makes every kwarg a VLM takes only on forward look
    unused, which is how LFM2-VL GRPO died on "The following `model_kwargs` are not used
    by the model: ['pixel_values', 'pixel_attention_mask', 'spatial_shapes']".
    """
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
    """The signature must not be bought by making the wrapper removable.

    accelerate's `extract_model_from_parallel(keep_fp32_wrapper = False)` runs on every
    GRPO step. Once `prepare_model` has recorded `_original_forward`, which it does for a
    full finetune and for a prepared reference model, it walks the `__wrapped__` chain
    looking for that original and rebinds whatever it lands on:

        while hasattr(forward, "__wrapped__"):
            forward = forward.__wrapped__
            if forward == original_forward: break
        model.forward = MethodType(forward, model)

    So `functools.wraps` here, which sets `__wrapped__`, is a route straight through this
    wrapper to the bare forward: the wrapper is silently gone, and GRPO goes back to
    materialising vocabulary-wide logits and to the OOM it exists to avoid. Setting
    `__signature__` alone gives generate the same answer with no link to follow.
    """
    import inspect as _inspect

    import torch
    from accelerate.utils.operations import convert_outputs_to_fp32
    from accelerate.utils.other import extract_model_from_parallel

    from unsloth.models.rl import _install_grpo_hidden_states_forward_wrapper

    class _Toy(torch.nn.Module):
        config = type("cfg", (), {"is_encoder_decoder": False})()

        def forward(self, input_ids = None, pixel_values = None, spatial_shapes = None, **kwargs):
            return input_ids

    model = _Toy()
    # What accelerate.prepare_model leaves behind under mixed precision.
    real_forward = model.forward
    model._original_forward = real_forward
    model.forward = convert_outputs_to_fp32(
        torch.autocast(device_type = "cuda", dtype = torch.bfloat16)(real_forward)
    )

    assert _install_grpo_hidden_states_forward_wrapper(model)
    assert getattr(model.forward, "_unsloth_grpo_hidden_states_forward_wrapped", False)
    # No __wrapped__: that attribute is the entire mechanism accelerate unwraps through.
    assert not hasattr(model.forward, "__wrapped__"), (
        "the wrapper carries __wrapped__, so accelerate will unwrap straight past it"
    )

    unwrapped = extract_model_from_parallel(model, keep_fp32_wrapper = False)

    assert getattr(unwrapped.forward, "_unsloth_grpo_hidden_states_forward_wrapped", False) or \
        getattr(
            getattr(unwrapped.forward, "__func__", None),
            "_unsloth_grpo_hidden_states_forward_wrapped",
            False,
        ), "accelerate removed the hidden-state wrapper"
    # And generate still sees the vision kwargs after that rebind.
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
    # num_tiles must still sit directly under num_images, above the inserted block.
    assert num_images_at < num_tiles_at < except_at
    # And it must still be nested inside `if images is not None:`, at 12 spaces.
    assert lines[num_tiles_at].startswith(" " * 16)


def test_the_trl_1_0_spelling_of_the_gemma_4_position_ids_is_forwarded():
    """TRL 1.0.x emits `pixel_position_ids`; 1.1.0 renamed it `image_position_ids`.

    Both are real keys in released TRLs an installed Unsloth can sit next to, and the
    model kwarg carries the same name as the inputs key, so each has to leave under the
    name it arrived with. Reading only one of the two drops the metadata for the other
    half of the version range, which is the very fault this change exists to close.
    """
    import torch
    from unsloth_zoo.rl_replacements import (
        GRPO_VISION_KEYS,
        grpo_get_vision_inputs,
        grpo_vision_chunks,
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
        # And the other spelling never appears: a model taking one rejects the other.
        other = "image_position_ids" if key == "pixel_position_ids" else "pixel_position_ids"
        assert all(other not in c for c in chunks)


def test_a_multi_image_row_indexes_the_position_ids_by_image():
    """Two images on one row: both tensors are image indexed, not sample indexed."""
    import torch
    from unsloth_zoo.rl_replacements import grpo_get_vision_inputs, grpo_vision_chunks

    inputs = {
        "pixel_values": torch.randn(3, 3, 16, 16),
        "image_position_ids": torch.arange(3).unsqueeze(-1),
        "num_images": [2, 1],
    }
    chunks = grpo_vision_chunks(grpo_get_vision_inputs(inputs), 2, 1)
    assert [c["pixel_values"].shape[0] for c in chunks] == [2, 1]
    assert [c["image_position_ids"].tolist() for c in chunks] == [[[0], [1]], [[2]]]


def test_an_old_zoo_still_forwards_the_sample_indexed_keys():
    """Without the shared chunker only the image indexed slicing is unavailable.

    A text run carrying token_type_ids used to be sliced by this path itself, so an
    unsloth_zoo predating the chunker must not cost it those keys; only the vision case
    is refused, and loudly.
    """
    from unsloth.models.rl_replacements import grpo_trainer__get_per_token_logps_and_entropies

    patched = grpo_trainer__get_per_token_logps_and_entropies(
        "_get_per_token_logps_and_entropies", ""
    )
    assert "_unsloth_grpo_vision_inputs(kwargs)" in patched
    fallback = patched.split("if _grpo_vision_chunks is None:")[-1]
    assert '"token_type_ids", "mm_token_type_ids"' in fallback
    assert "vision_chunks = [{} for _ in input_ids_chunks]" not in patched


def test_the_wrapper_keeps_trls_logits_to_keep_probe_answerable():
    """TRL reads `inspect.signature(model.forward)` once, in GRPOTrainer.__init__, to decide
    whether the model accepts `logits_to_keep` (SmolVLM and Idefics3 do not).

    A bare (*args, **kwargs) wrapper answered "no" for every model, so no model was ever
    given the limiter. Restoring the signature restores the per model answer, which also
    means a text only run starts passing `logits_to_keep` again: same loss to five
    decimals, not bit for bit.
    """
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
    """The "upgrade unsloth_zoo" gate for multi image GRPO.

    It used to answer by grepping `inspect.getsource(grpo_accumulated_loss)` for the
    string "num_images". Once that function delegates its slicing to the shared chunker
    the name survives there only as a local nobody reads, so removing it, which is what
    any linter asks for, silently turned a working multi image run into a false
    "Please upgrade unsloth_zoo". The gate now probes for the helper that actually does
    the work.
    """
    from unsloth.models.rl_replacements import grpo_trainer_compute_loss

    patched = grpo_trainer_compute_loss("compute_loss", "")
    gate = patched.split("_unsloth_requires_multi_image_zoo(num_images)")[1]
    gate = gate.split("Please upgrade ")[0]
    assert "from unsloth_zoo.rl_replacements import grpo_vision_chunks" in gate
    # The import probe has to be consulted before the source grep, or the grep still
    # decides the answer on a current zoo.
    assert gate.index("grpo_vision_chunks") < gate.index("inspect.getsource")
