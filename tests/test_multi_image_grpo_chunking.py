"""Static + behavioral checks for the multi-image GRPO chunking and
zoo compatibility guard in unsloth/models/rl_replacements.py."""

from __future__ import annotations

import math
import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _read_source() -> str:
    with open(SOURCE_PATH, "r") as fh:
        return fh.read()


# ---------- Per-chunk slicing fixes (cum_rows, cum_imgs, axes) ----------


def test_cum_rows_materialized_on_cpu():
    src = _read_source()
    idx = src.find("cum_rows = torch.cat")
    assert idx != -1, "cum_rows assignment must exist"
    window = src[idx : idx + 400]
    assert "rows_per_sample.cumsum(0)" in window
    assert (
        ").cpu()" in window
    ), "cum_rows must be moved to CPU once via .cpu() after construction"


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


# ---------- Behavioral simulation of chunk math ----------


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


# ---------- Zoo compatibility guard ----------


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
    assert (
        sig_call < src_call
    ), "signature.parameters must run before the getsource fallback"


def test_guard_only_raises_when_both_checks_fail():
    src = _read_source()
    pattern = re.compile(
        r"_supports_num_images\s*=\s*\(\s*\"num_images\"\s*\n?\s*in\s+inspect\.signature.*?"
        r"if not _supports_num_images:.*?_supports_num_images\s*=\s*\"num_images\" in _zoo_src.*?"
        r"if not _supports_num_images:\s*\n\s*raise RuntimeError",
        re.DOTALL,
    )
    assert pattern.search(
        src
    ), "guard flow must be: signature check, source fallback, then raise"


def test_guard_introspection_failure_does_not_silent_no_op():
    src = _read_source()
    assert (
        "(TypeError, OSError)" in src
    ), "guard must catch inspect.getsource failures explicitly"
    assert re.search(
        r"_zoo_src\s*=\s*['\"]{2}", src
    ), "introspection failure path must default _zoo_src to empty string"
