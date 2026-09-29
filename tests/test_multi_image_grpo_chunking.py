"""Static + behavioral checks for multi-image GRPO chunking and the zoo
compatibility guard in unsloth/models/rl_replacements.py."""

from __future__ import annotations

import math
import os
import re
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


def test_source_reads_the_shared_key_tuple():
    src = _read_source()
    assert "grpo_get_vision_inputs" in src
    assert "grpo_vision_chunks" in src
    assert "pixel_values_chunks" not in src
    assert "image_grid_thw_chunks" not in src


def test_grid_model_slices_rows_by_patch_and_grid_by_image():
    import torch

    (grpo_vision_chunks,) = _zoo_vision_helpers("grpo_vision_chunks")

    num_images = [2, 1, 3, 1]
    grid = torch.tensor([[1, 2, 2]] * sum(num_images))  # 4 patch rows per image
    rows = int(grid.prod(dim = -1).sum())
    vision = {
        "pixel_values": torch.arange(rows).reshape(rows, 1).float(),
        "image_grid_thw": grid,
        "num_images": num_images,
    }
    chunks = grpo_vision_chunks(vision, total_samples = 4, batch_size = 2)
    assert len(chunks) == 2
    # samples 0 and 1 hold images 0 to 2, so patch rows 0 to 11
    assert chunks[0]["pixel_values"].shape[0] == 12
    assert chunks[0]["image_grid_thw"].shape[0] == 3
    assert chunks[1]["pixel_values"].shape[0] == 16
    assert chunks[1]["image_grid_thw"].shape[0] == 4
    assert torch.equal(chunks[1]["pixel_values"], vision["pixel_values"][12:])


def test_image_sizes_follows_the_image_axis_when_it_is_per_image():
    import torch

    (grpo_vision_chunks,) = _zoo_vision_helpers("grpo_vision_chunks")

    vision = {
        "pixel_values": torch.zeros(12, 1),
        "image_grid_thw": torch.tensor([[1, 2, 2]] * 3),
        "image_sizes": torch.tensor([[10, 10], [20, 20], [30, 30]]),
        "num_images": [2, 1],
    }
    chunks = grpo_vision_chunks(vision, total_samples = 2, batch_size = 1)
    assert chunks[0]["image_sizes"].tolist() == [[10, 10], [20, 20]]
    assert chunks[1]["image_sizes"].tolist() == [[30, 30]]


def test_pixel_attention_mask_axis_is_chosen_per_shape():
    import torch

    (grpo_vision_chunks,) = _zoo_vision_helpers("grpo_vision_chunks")

    base = {
        "pixel_values": torch.zeros(12, 1),
        "image_grid_thw": torch.tensor([[1, 2, 2]] * 3),
        "num_images": [2, 1],
    }
    # one mask row per image: image axis
    per_image = grpo_vision_chunks({**base, "pixel_attention_mask": torch.zeros(3, 4)}, 2, 1)
    assert per_image[0]["pixel_attention_mask"].shape[0] == 2
    assert per_image[1]["pixel_attention_mask"].shape[0] == 1
    # one mask row per patch row: patch axis
    per_row = grpo_vision_chunks({**base, "pixel_attention_mask": torch.zeros(12, 4)}, 2, 1)
    assert per_row[0]["pixel_attention_mask"].shape[0] == 8
    assert per_row[1]["pixel_attention_mask"].shape[0] == 4


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
