# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""patch_flash_attn_varlen_compat.py and its workflow steps.

The pinned flash-attn revision appends `int num_splits = 0` to mha_varlen_fwd. pybind11 drops the
default when it binds a function pointer, so `varlen_fwd` needs all 22 arguments and the 21-argument
call that xFormers' FA2 operator makes is refused. The patch adds the 21-argument form back as an
overload that forwards num_splits = 0.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "prebuilt-cuda-wheels.yml"
SCRIPT = REPO / ".github" / "scripts" / "patch_flash_attn_varlen_compat.py"

_spec = importlib.util.spec_from_file_location("patch_flash_attn_varlen_compat", SCRIPT)
patch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch)

# The parts of csrc/flash_attn/flash_api.cpp the patch reads, verbatim from the pinned revision
# (comments shortened).
_PINNED = """\
std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size
               const at::Tensor &k,  // total_k x num_heads_k x head_size
               const at::Tensor &v,  // total_k x num_heads_k x head_size
               std::optional<at::Tensor> &out_, // total_q x num_heads x head_size
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<const at::Tensor> &leftpad_k_, // batch_size
               std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               std::optional<at::Tensor> unused_generator_compat,
               int num_splits = 0) {
    return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass");
    m.def("varlen_fwd", &FLASH_NAMESPACE::mha_varlen_fwd, "Forward pass (variable length)");
    m.def("bwd", &FLASH_NAMESPACE::mha_bwd, "Backward pass");
}
"""

# 2.8.3 has no num_splits: its binding already takes 21 arguments.
_V283 = _PINNED.replace(",\n               int num_splits = 0)", ")")

_NAMES = [
    "q", "k", "v", "out_", "cu_seqlens_q", "cu_seqlens_k", "seqused_k", "leftpad_k_",
    "block_table_", "alibi_slopes_", "max_seqlen_q", "max_seqlen_k", "p_dropout",
    "softmax_scale", "zero_tensors", "is_causal", "window_size_left", "window_size_right",
    "softcap", "return_softmax", "unused_generator_compat",
]  # fmt: skip


def _tree(tmp_path, text = _PINNED):
    path = tmp_path / "csrc" / "flash_attn" / "flash_api.cpp"
    path.parent.mkdir(parents = True)
    path.write_text(text, encoding = "utf-8")
    return path


def _run(root):
    return subprocess.run([sys.executable, str(SCRIPT), str(root)], capture_output = True, text = True)


def test_adds_a_21_argument_overload_that_forwards_num_splits_zero(tmp_path):
    path = _tree(tmp_path)
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "21-argument" in result.stdout
    text = path.read_text(encoding = "utf-8")
    # The 22-argument binding stays, and comes first.
    original = text.index('m.def("varlen_fwd", &FLASH_NAMESPACE::mha_varlen_fwd,')
    overload = text.index('m.def("varlen_fwd", [](')
    assert original < overload < text.index('m.def("bwd"')
    call = re.search(r"return FLASH_NAMESPACE::mha_varlen_fwd\((.*?)\);", text).group(1)
    assert [a.strip() for a in call.split(",")] == _NAMES + ["0"]
    lambda_params = re.search(r"\[\]\((.*?)\) \{", text).group(1)
    assert "num_splits" not in lambda_params and "//" not in lambda_params
    assert len(lambda_params.split(",")) == 21


def test_is_idempotent(tmp_path):
    path = _tree(tmp_path)
    assert _run(tmp_path).returncode == 0
    first = path.read_text(encoding = "utf-8")
    again = _run(tmp_path)
    assert again.returncode == 0 and "already patched" in again.stdout
    assert path.read_text(encoding = "utf-8") == first


def test_a_revision_without_num_splits_is_left_alone(tmp_path):
    path = _tree(tmp_path, _V283)
    result = _run(tmp_path)
    assert result.returncode == 0 and "nothing to do" in result.stdout
    assert path.read_text(encoding = "utf-8") == _V283


@pytest.mark.parametrize(
    "text, message",
    [
        (_PINNED.replace('"Forward pass (variable length)"', '"Varlen forward"'), "found 0"),
        (
            _PINNED.replace("mha_varlen_fwd(at::Tensor &q", "mha_varlen_fwd2(at::Tensor &q"),
            "found 0",
        ),
        (_PINNED + _PINNED, "found 2"),
    ],
)
def test_drift_fails_loudly(tmp_path, text, message):
    path = _tree(tmp_path, text)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "::error::" in result.stderr and message in result.stderr
    assert path.read_text(encoding = "utf-8") == text


def test_a_missing_file_fails(tmp_path):
    result = _run(tmp_path)
    assert result.returncode == 1 and "does not exist" in result.stderr


@pytest.mark.skipif(shutil.which("g++") is None, reason = "needs a C++ compiler")
def test_the_overload_compiles(tmp_path):
    patched, _ = patch.patch_source(_PINNED)
    stubs = (
        "#include <vector>\n#include <optional>\n"
        "namespace at { struct Tensor {}; }\n"
        "struct Module { template <class F> void def(const char*, F&&, const char*) {} void doc() {} };\n"
        "#define PYBIND11_MODULE(name, m) void bind(Module &m)\n"
        "#define FLASH_NAMESPACE flash\n"
        "namespace flash { int mha_fwd, mha_bwd;\n"
    )
    source = stubs + patched.replace("PYBIND11_MODULE", "}\nPYBIND11_MODULE", 1)
    source = source.replace('m.doc() = "FlashAttention";', "")
    source = source.replace("&FLASH_NAMESPACE::mha_fwd", "0").replace(
        "&FLASH_NAMESPACE::mha_bwd", "0"
    )
    cpp = tmp_path / "stub.cpp"
    cpp.write_text(source, encoding = "utf-8")
    result = subprocess.run(
        ["g++", "-std=c++17", "-fsyntax-only", str(cpp)], capture_output = True, text = True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("job", ["warm", "build"])
def test_runs_on_every_flash_attn_cell_before_the_compile(job):
    steps = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"][job]["steps"]
    names = [s.get("name", "") for s in steps]
    index = next(
        i
        for i, s in enumerate(steps)
        if "patch_flash_attn_varlen_compat.py src" in s.get("run", "")
    )
    assert steps[index]["if"] == "matrix.package == 'flash-attn'"
    compile_step = next(
        i for i, n in enumerate(names) if n.startswith(("Compile slice", "Build the wheel"))
    )
    assert index < compile_step
