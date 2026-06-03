"""Unit tests for GGUF intermediate-file cleanup / reordering logic.

Regression coverage for issue #4932 (PR #4937): ``bf16`` (and any first-conversion
format) must be preserved when it is explicitly requested alongside quantized types,
and removed when it is only a conversion intermediate.

The logic under test lives in ``unsloth.save._finalize_gguf_file_list``. Importing
``unsloth.save`` pulls in torch / bitsandbytes / peft and the full package bootstrap,
so — following the pattern in ``test_save_shell_injection.py`` — we extract just that
function from source via AST and ``exec`` it in isolation. The function is pure and
uses only builtins, so it runs with no third-party dependencies and no GPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


SAVE_PY = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"


def _load_finalizer():
    source = SAVE_PY.read_text(encoding = "utf-8")
    tree = ast.parse(source, filename = str(SAVE_PY))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_finalize_gguf_file_list":
            func_src = ast.get_source_segment(source, node)
            namespace: dict = {}
            exec(compile(func_src, str(SAVE_PY), "exec"), namespace)
            return namespace["_finalize_gguf_file_list"]
    raise AssertionError("_finalize_gguf_file_list not found in save.py")


finalize = _load_finalizer()


def _run(initial_files, quants, quantization_method, first_conversion, is_vlm, quants_created = None):
    """Build inputs the way ``save_to_gguf`` does and invoke the finalizer.

    ``all_saved_locations`` starts as a copy of ``initial_files`` with the quantized
    outputs appended after it. ``quants_created`` defaults to ``bool(quants)``.
    """
    all_saved_locations = list(initial_files) + list(quants)
    if quants_created is None:
        quants_created = bool(quants)
    return finalize(
        all_saved_locations = all_saved_locations,
        initial_files = list(initial_files),
        quantization_method = quantization_method,
        first_conversion = first_conversion,
        is_vlm = is_vlm,
        quants_created = quants_created,
    )


# --------------------------------------------------------------------------- #
# Text models                                                                 #
# --------------------------------------------------------------------------- #

def test_text_intermediate_only_single_shard_deleted():
    """bf16 used only as intermediate (not requested) -> deleted from list and disk."""
    bf16 = "m.BF16.gguf"
    final, to_delete = _run(
        initial_files = [bf16],
        quants = ["m.Q4_K_M.gguf", "m.Q8_0.gguf"],
        quantization_method = ["q4_k_m", "q8_0"],
        first_conversion = "bf16",
        is_vlm = False,
    )
    assert to_delete == [bf16]
    assert bf16 not in final
    assert set(final) == {"m.Q4_K_M.gguf", "m.Q8_0.gguf"}


def test_text_intermediate_only_multi_shard_all_deleted():
    """Sharded intermediate -> EVERY text shard deleted, not just the first (PR 6bcb02f5)."""
    shards = ["m-00001-of-00002.BF16.gguf", "m-00002-of-00002.BF16.gguf"]
    final, to_delete = _run(
        initial_files = shards,
        quants = ["m.Q4_K_M.gguf"],
        quantization_method = ["q4_k_m"],
        first_conversion = "bf16",
        is_vlm = False,
    )
    assert to_delete == shards
    assert all(s not in final for s in shards)
    assert final == ["m.Q4_K_M.gguf"]


def test_text_bf16_requested_is_retained_single_shard():
    """Core regression for #4932: bf16 requested alongside quants must survive."""
    bf16 = "m.BF16.gguf"
    final, to_delete = _run(
        initial_files = [bf16],
        quants = ["m.Q4_K_M.gguf", "m.Q8_0.gguf"],
        quantization_method = ["q4_k_m", "q8_0", "bf16"],
        first_conversion = "bf16",
        is_vlm = False,
    )
    assert to_delete == []
    assert bf16 in final
    # A runnable quant is first; the full-precision file lands last.
    assert final[0] != bf16
    assert final[-1] == bf16


def test_text_bf16_requested_is_retained_multi_shard():
    shards = ["m-00001-of-00002.BF16.gguf", "m-00002-of-00002.BF16.gguf"]
    final, to_delete = _run(
        initial_files = shards,
        quants = ["m.Q4_K_M.gguf"],
        quantization_method = ["q4_k_m", "bf16"],
        first_conversion = "bf16",
        is_vlm = False,
    )
    assert to_delete == []
    assert all(s in final for s in shards)
    assert final[0] == "m.Q4_K_M.gguf"


# --------------------------------------------------------------------------- #
# Vision-language models (mmproj is the last initial file)                    #
# --------------------------------------------------------------------------- #

def test_vlm_intermediate_only_keeps_mmproj_deletes_text():
    """VLM, bf16 not requested: text deleted, mmproj kept and ends at index -1."""
    text = "m.BF16.gguf"
    mmproj = "mmproj.BF16.gguf"
    final, to_delete = _run(
        initial_files = [text, mmproj],
        quants = ["m.Q4_K_M.gguf"],
        quantization_method = ["q4_k_m"],
        first_conversion = "bf16",
        is_vlm = True,
    )
    assert to_delete == [text]
    assert mmproj not in to_delete
    assert final[-1] == mmproj          # consumed as --mmproj downstream
    assert final[0] == "m.Q4_K_M.gguf"  # consumed as -m (runnable) downstream
    assert text not in final


def test_vlm_intermediate_only_multi_shard_text_deleted_mmproj_kept():
    shards = ["m-00001-of-00002.BF16.gguf", "m-00002-of-00002.BF16.gguf"]
    mmproj = "mmproj.BF16.gguf"
    final, to_delete = _run(
        initial_files = shards + [mmproj],
        quants = ["m.Q4_K_M.gguf"],
        quantization_method = ["q4_k_m"],
        first_conversion = "bf16",
        is_vlm = True,
    )
    assert to_delete == shards
    assert mmproj not in to_delete
    assert final[-1] == mmproj
    assert final[0] == "m.Q4_K_M.gguf"


def test_vlm_bf16_requested_retains_text_and_mmproj_single_shard():
    text = "m.BF16.gguf"
    mmproj = "mmproj.BF16.gguf"
    final, to_delete = _run(
        initial_files = [text, mmproj],
        quants = ["m.Q4_K_M.gguf"],
        quantization_method = ["q4_k_m", "bf16"],
        first_conversion = "bf16",
        is_vlm = True,
    )
    assert to_delete == []
    assert text in final
    assert final[-1] == mmproj
    assert final[0] == "m.Q4_K_M.gguf"


def test_vlm_bf16_requested_multi_shard_preserves_order_and_mmproj_last():
    """VLM reorder (PR c9666917): multi-shard text kept in original order, mmproj last."""
    shards = ["m-00001-of-00002.BF16.gguf", "m-00002-of-00002.BF16.gguf"]
    mmproj = "mmproj.BF16.gguf"
    final, to_delete = _run(
        initial_files = shards + [mmproj],
        quants = ["m.Q4_K_M.gguf", "m.Q8_0.gguf"],
        quantization_method = ["q4_k_m", "q8_0", "bf16"],
        first_conversion = "bf16",
        is_vlm = True,
    )
    assert to_delete == []
    assert final[-1] == mmproj
    assert final[0] != mmproj
    # Text shards retained in their original (non-reversed) order, between quants and mmproj.
    text_positions = [final.index(s) for s in shards]
    assert text_positions == sorted(text_positions)
    assert all(s in final for s in shards)


# --------------------------------------------------------------------------- #
# No-op / short-circuit paths                                                 #
# --------------------------------------------------------------------------- #

def test_bf16_only_no_quants_is_noop():
    """quantization_method=['bf16'] -> no quants created, list returned untouched."""
    bf16 = "m.BF16.gguf"
    final, to_delete = _run(
        initial_files = [bf16],
        quants = [],
        quantization_method = ["bf16"],
        first_conversion = "bf16",
        is_vlm = False,
        quants_created = False,
    )
    assert to_delete == []
    assert final == [bf16]


def test_quants_not_created_short_circuits_regardless_of_flags():
    """When quants_created is False the input list is returned verbatim (no reverse)."""
    initial = ["a.BF16.gguf", "mmproj.BF16.gguf"]
    final, to_delete = _run(
        initial_files = initial,
        quants = [],
        quantization_method = ["q4_k_m"],
        first_conversion = "bf16",
        is_vlm = True,
        quants_created = False,
    )
    assert to_delete == []
    assert final == initial


# --------------------------------------------------------------------------- #
# Cross-cutting invariant the downstream call site relies on                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "initial_files, quants, quantization_method, first_conversion, is_vlm",
    [
        (["m.BF16.gguf"], ["m.Q4_K_M.gguf", "m.Q8_0.gguf"], ["q4_k_m", "q8_0"], "bf16", False),
        (["m.BF16.gguf"], ["m.Q4_K_M.gguf"], ["q4_k_m", "bf16"], "bf16", False),
        (["m.BF16.gguf", "mmproj.BF16.gguf"], ["m.Q4_K_M.gguf"], ["q4_k_m"], "bf16", True),
        (["s1.BF16.gguf", "s2.BF16.gguf", "mmproj.BF16.gguf"], ["m.Q4_K_M.gguf", "m.Q8_0.gguf"], ["q4_k_m", "q8_0", "bf16"], "bf16", True),
    ],
)
def test_index_invariants(initial_files, quants, quantization_method, first_conversion, is_vlm):
    """index 0 is never the mmproj; for VLMs index -1 is always the mmproj."""
    final, _ = _run(
        initial_files = initial_files,
        quants = quants,
        quantization_method = quantization_method,
        first_conversion = first_conversion,
        is_vlm = is_vlm,
    )
    assert final, "final list must not be empty when quants were created"
    if is_vlm:
        mmproj = initial_files[-1]
        assert final[-1] == mmproj
        assert final[0] != mmproj
