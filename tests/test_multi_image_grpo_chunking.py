"""Static + behavioral checks for multi-image GRPO chunking and the zoo
compatibility guard in unsloth/models/rl_replacements.py."""

from __future__ import annotations

import math
import os
import re
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _read_source() -> str:
    with open(SOURCE_PATH, "r") as fh:
        return fh.read()


# Per-chunk slicing fixes (cum_rows, cum_imgs, axes)


def test_cum_rows_materialized_on_cpu():
    src = _read_source()
    idx = src.find("cum_rows = torch.cat")
    assert idx != -1, "cum_rows assignment must exist"
    window = src[idx : idx + 400]
    assert "rows_per_sample.cumsum(0)" in window
    assert ").cpu()" in window, "cum_rows must be moved to CPU once via .cpu() after construction"


def test_cum_imgs_slice_indices_use_item():
    src = _read_source()
    assert "cum_imgs[start].item()" in src
    assert "cum_imgs[end].item()" in src


def test_image_sizes_image_axis_branch_present():
    src = _read_source()
    assert "image_sizes[img_start:img_end]" in src
    assert "_image_sizes_n" in src and "total_images" in src


def test_pixel_attention_mask_three_way_check_present():
    src = _read_source()
    assert "pixel_attention_mask[img_start:img_end]" in src
    assert "pixel_attention_mask[start_pixel_idx:end_pixel_idx]" in src
    assert "pixel_attention_mask[start:end]" in src
    assert "image_grid_thw.shape[0]" in src


def test_image_sizes_chunked_after_branch_decision():
    src = _read_source()
    pattern = re.compile(
        r"attention_mask_chunks\.append\(attention_mask\[start:end\]\)\s*\n\s*"
        r"image_sizes_chunks\.append\(slice_sample_axis\(image_sizes,\s*start,\s*end\)\)",
    )
    assert pattern.search(src) is None, (
        "image_sizes_chunks must not be appended unconditionally on the "
        "sample axis above the if/else; the axis is chosen per branch"
    )


# Behavioral simulation of chunk math


def _simulate_chunk_indices(num_images, B):
    total_samples = len(num_images)
    batch_size = max(1, math.ceil(total_samples / B))
    cum_imgs = [0]
    for n in num_images:
        cum_imgs.append(cum_imgs[-1] + n)
    chunks = []
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        chunks.append((start, end, cum_imgs[start], cum_imgs[end]))
    return chunks


def test_simulate_multi_image_chunk_image_axis_correct():
    chunks = _simulate_chunk_indices([2, 1, 3, 1], B = 2)
    assert chunks == [(0, 2, 0, 3), (2, 4, 3, 7)]


def test_simulate_uniform_image_chunking_unchanged():
    chunks = _simulate_chunk_indices([1, 1, 1, 1], B = 2)
    assert chunks == [(0, 2, 0, 2), (2, 4, 2, 4)]


def test_simulate_pixel_attention_mask_axis_decision():
    def select_axis(
        pam_shape0,
        pixel_values_shape0,
        image_grid_thw_shape0,
        input_ids_shape0,
        num_images_provided,
    ):
        if num_images_provided and pam_shape0 == image_grid_thw_shape0:
            return "image"
        if pam_shape0 == pixel_values_shape0 and pam_shape0 != input_ids_shape0:
            return "pixel"
        return "sample"

    assert select_axis(3, 9, 3, 2, True) == "image"
    assert select_axis(9, 9, 3, 2, True) == "pixel"
    assert select_axis(4, 4, 4, 4, False) == "sample"
    assert select_axis(2, 2, 2, 2, False) == "sample"


# Zoo compatibility guard


def test_zoo_guard_branch_present():
    src = _read_source()
    assert "_unsloth_grpo_zoo_checked" in src
    assert "raise RuntimeError" in src
    assert "https://github.com/unslothai/unsloth-zoo/pull/613" in src
    assert "Multi-image GRPO" in src


def test_guard_helper_skips_all_ones_num_images():
    src = _read_source()
    helper_match = re.search(
        r"def _unsloth_requires_multi_image_zoo\(value\):.*?return any\(int\(n\) != 1 for n in counts\)",
        src,
        re.DOTALL,
    )
    assert helper_match, "guard helper must compute any(int(n) != 1)"
    namespace: dict = {}

    class _FakeTensor:
        def __init__(self, values):
            self._values = list(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def reshape(self, *_args, **_kwargs):
            return self

        def tolist(self):
            return list(self._values)

    namespace["torch"] = type("torch_stub", (), {"Tensor": _FakeTensor})()
    exec(helper_match.group(0), namespace)
    helper = namespace["_unsloth_requires_multi_image_zoo"]

    assert helper(None) is False
    assert helper([1, 1, 1, 1]) is False
    assert helper([2, 1]) is True
    assert helper([0, 1, 1]) is True
    assert helper(_FakeTensor([1, 1, 1])) is False
    assert helper(_FakeTensor([2, 1])) is True


def test_guard_prefers_inspect_signature_over_getsource():
    src = _read_source()
    helper_idx = src.find("_unsloth_requires_multi_image_zoo")
    body = src[helper_idx:]
    sig_call = body.find("inspect.signature(grpo_accumulated_loss).parameters")
    src_call = body.find("inspect.getsource(grpo_accumulated_loss)")
    assert sig_call != -1
    assert src_call != -1
    assert sig_call < src_call, "signature.parameters must run before the getsource fallback"


def test_guard_only_raises_when_both_checks_fail():
    src = _read_source()
    pattern = re.compile(
        r"_supports_num_images\s*=\s*\(\s*\"num_images\"\s*\n?\s*in\s+inspect\.signature.*?"
        r"if not _supports_num_images:.*?_supports_num_images\s*=\s*\"num_images\" in _zoo_src.*?"
        r"if not _supports_num_images:\s*\n\s*raise RuntimeError",
        re.DOTALL,
    )
    assert pattern.search(src), "guard flow must be: signature check, source fallback, then raise"


def test_guard_introspection_failure_does_not_silent_no_op():
    src = _read_source()
    assert "(TypeError, OSError)" in src, "guard must catch inspect.getsource failures explicitly"
    assert re.search(
        r"_zoo_src\s*=\s*['\"]{2}", src
    ), "introspection failure path must default _zoo_src to empty string"


# Tiled VLM kwargs forwarding (spatial_shapes/num_tiles/image_position_ids, T6960)


def _extract_extra_vision_kwargs_block():
    src = _read_source()
    mid = src.find("_extra_vision_kwargs = {}")
    assert mid != -1
    # rewind to the start of the line so the block's own leading whitespace
    # is included; otherwise textwrap.dedent can't compute a common margin
    start = src.rfind(chr(10), 0, mid) + 1
    end = src.find('with torch.amp.autocast(device_type = "cuda"', mid)
    assert end != -1 and end > start
    return textwrap.dedent(src[start:end])


def _build_extra_vision_kwargs(
    token_type_ids_chunk = None,
    mm_token_type_ids_chunk = None,
    spatial_shapes_chunk = None,
    num_tiles_chunk = None,
    image_position_ids_chunk = None,
):
    namespace = {
        "token_type_ids_chunk": token_type_ids_chunk,
        "mm_token_type_ids_chunk": mm_token_type_ids_chunk,
        "spatial_shapes_chunk": spatial_shapes_chunk,
        "num_tiles_chunk": num_tiles_chunk,
        "image_position_ids_chunk": image_position_ids_chunk,
    }
    exec(_extract_extra_vision_kwargs_block(), namespace)
    return namespace["_extra_vision_kwargs"]


def test_spatial_shapes_num_tiles_image_position_ids_extracted():
    """Reproduction: on base, these three kwargs are dropped before the chunk
    forward() call is built; on head, extraction (kwargs.get) and forwarding
    (_extra_vision_kwargs[...] = ...) are both present and wired together."""
    src = _read_source()
    assert 'kwargs.get("spatial_shapes", None)' in src
    assert 'kwargs.get("num_tiles", None)' in src
    assert 'kwargs.get("image_position_ids", None)' in src

    # Behavioral: execute the real _extra_vision_kwargs construction block
    # (extracted verbatim from source) and confirm the three kwargs actually
    # reach the dict that is **-unpacked into the model forward() call.
    extra = _build_extra_vision_kwargs(
        spatial_shapes_chunk = "SHAPES",
        num_tiles_chunk = "TILES",
        image_position_ids_chunk = "POS_IDS",
    )
    assert extra.get("spatial_shapes") == "SHAPES"
    assert extra.get("num_tiles") == "TILES"
    assert extra.get("image_position_ids") == "POS_IDS"


def test_cum_tiles_cumsum_present_and_cpu_materialized():
    src = _read_source()
    idx = src.find("cum_tiles = [0]")
    assert idx != -1, "cum_tiles cumulative-sum construction must exist"
    window = src[idx : idx + 150]
    assert "for _n_tiles in num_tiles:" in window
    assert "cum_tiles.append(cum_tiles[-1] + int(_n_tiles))" in window
    assert "else:\n                cum_tiles = None" in src
    # gated the same way as cum_rows/cum_imgs
    gate_idx = src.rfind("if (", 0, idx)
    gate_window = src[gate_idx:idx]
    assert "num_tiles is not None" in gate_window
    assert "image_grid_thw is not None" in gate_window
    assert "pixel_values is not None" in gate_window
    assert "num_images is not None" in gate_window
    # plain python list -> no GPU tensor / device kwarg, unlike cum_rows
    assert "torch.tensor" not in window and "device" not in window


def test_three_new_chunk_lists_initialized():
    src = _read_source()
    idx = src.find("mm_token_type_ids_chunks = []")
    assert idx != -1
    window = src[idx : idx + 200]
    assert "spatial_shapes_chunks = []" in window
    assert "num_tiles_chunks = []" in window
    assert "image_position_ids_chunks = []" in window


def _simulate_tile_boundaries(num_images, num_tiles_per_image, B):
    """Mirror of the real cum_imgs/cum_tiles per-chunk slicing math using
    plain Python, independent of torch."""
    total_samples = len(num_images)
    batch_size = max(1, math.ceil(total_samples / B))
    cum_imgs = [0]
    for n in num_images:
        cum_imgs.append(cum_imgs[-1] + n)
    cum_tiles = [0]
    for n in num_tiles_per_image:
        cum_tiles.append(cum_tiles[-1] + n)
    chunks = []
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        img_start, img_end = cum_imgs[start], cum_imgs[end]
        tile_start, tile_end = cum_tiles[img_start], cum_tiles[img_end]
        chunks.append((img_start, img_end, tile_start, tile_end))
    return chunks


def test_simulate_tile_boundary_slicing_matches_num_tiles_cumsum():
    # 4 samples, images per sample = [2, 1, 1, 2] (image axis, sum = 6 images)
    num_images = [2, 1, 1, 2]
    # per-image tile counts, aligned to the image axis (6 entries)
    num_tiles_per_image = [3, 1, 2, 4, 1, 2]
    chunks = _simulate_tile_boundaries(num_images, num_tiles_per_image, B = 2)
    # sample chunk 0 covers samples [0,2) -> images [0,3) -> tiles [0,6)
    # sample chunk 1 covers samples [2,4) -> images [3,6) -> tiles [6,13)
    assert chunks == [(0, 3, 0, 6), (3, 6, 6, 13)]
    total_tiles = sum(num_tiles_per_image)
    assert chunks[-1][-1] == total_tiles


def test_all_three_new_chunk_lists_appended_in_every_branch():
    src = _read_source()
    for name in ("num_tiles_chunks", "image_position_ids_chunks", "spatial_shapes_chunks"):
        appends = len(re.findall(re.escape(name) + r"\.append\(", src))
        assert appends >= 3, (
            f"{name} must be appended in the tile-capable branch (slice + None fallback) "
            f"and in the no-vision else branch to stay aligned with the other chunk lists "
            f"for zip(); found {appends} append call(s)"
        )
    # explicit None fallback inside the image_grid_thw/pixel_values branch
    assert "if num_tiles is None or img_start is None:" in src
    assert "if image_position_ids is None or img_start is None:" in src
    assert "if spatial_shapes is None or cum_tiles is None or img_start is None:" in src
    # explicit None fallback in the outer else (no vision at all) keeps all
    # chunk lists the same length so zip() cannot silently misalign
    else_marker = (
        "image_sizes_chunks.append(slice_sample_axis(image_sizes, start, end))\n"
        "                    num_tiles_chunks.append(None)"
    )
    assert else_marker in src, "outer else branch must append None for all three new lists"


def test_extra_vision_kwargs_forwards_tile_kwargs():
    src = _read_source()
    assert '_extra_vision_kwargs["spatial_shapes"] = spatial_shapes_chunk' in src
    assert '_extra_vision_kwargs["num_tiles"] = num_tiles_chunk' in src
    assert '_extra_vision_kwargs["image_position_ids"] = image_position_ids_chunk' in src
    # the for-loop unpack tuple must include the 3 new chunk vars
    assert (
        "mm_token_type_ids_chunk,\n"
        "                    spatial_shapes_chunk,\n"
        "                    num_tiles_chunk,\n"
        "                    image_position_ids_chunk,\n"
        "                ) in zipped_inputs:"
    ) in src
    # zipped_inputs must zip() the 3 new chunk lists
    assert (
        "mm_token_type_ids_chunks,\n"
        "                spatial_shapes_chunks,\n"
        "                num_tiles_chunks,\n"
        "                image_position_ids_chunks,\n"
        "            )"
    ) in src

    # Behavioral: verify actual forwarding through the extracted block
    extra = _build_extra_vision_kwargs(spatial_shapes_chunk = [1, 2])
    assert extra == {"spatial_shapes": [1, 2]}


def test_text_only_grpo_unaffected():
    """Negative space: a text-only GRPO batch has no vision kwargs at all, so
    kwargs.get(...) returns None for spatial_shapes/num_tiles/image_position_ids,
    and the None-gated forwarding must append None (not forward anything)."""
    extra = _build_extra_vision_kwargs()
    assert extra == {}, "text-only batches must not forward any tile kwargs"

    src = _read_source()
    # the outer (no image_grid_thw/pixel_values) branch is unaffected: it
    # still appends None for the 3 new lists exactly like the pre-existing ones
    idx = src.find("else:\n                    pixel_values_chunks.append(None)")
    assert idx != -1
    window = src[idx : idx + 500]
    assert "num_tiles_chunks.append(None)" in window
    assert "image_position_ids_chunks.append(None)" in window
    assert "spatial_shapes_chunks.append(None)" in window
