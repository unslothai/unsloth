# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""torchao 0.18 must not be able to kill `import unsloth`.

torchao 0.17 guarded `from torch.nn.functional import ScalingType, SwizzleType`
behind `torch_version_at_least("2.10.0")`; 0.18.0 left it unguarded at module
level, so torch below 2.10 raises ImportError. It surfaces while importing
transformers, where unsloth_zoo's import guard re-raises a bare Exception
naming neither torchao nor torch. Seen on Colab in Gemma3_(4B)-Vision-GRPO,
Qwen3_5_(4B)_Vision and Qwen3_8B_FP8_GRPO.

The placeholder refuses to be used: 0.17 left these names undefined on old
torch anyway, and a stub impersonating a real enum could hand a float8 path a
meaningless value, which is worse than the crash. Checked against the 0.18.0
source: neither symbol is evaluated at import time, so being strict is safe.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unsloth.import_fixes import (  # noqa: E402
    _TORCHAO_TORCH_SYMBOLS,
    _make_torch_symbol_placeholder,
    fix_torchao_torch_symbol_skew,
)

GPU_INIT = ROOT / "unsloth" / "_gpu_init.py"


# ---- the placeholder ------------------------------------------------------


def test_it_can_be_imported():
    """The whole point: satisfy `from torch.nn.functional import X`."""
    ph = _make_torch_symbol_placeholder("ScalingType", "detail here")
    assert ph is not None
    assert ph.__name__ == "ScalingType"


def test_using_it_raises_with_an_actionable_message():
    ph = _make_torch_symbol_placeholder("ScalingType", "torchao 0.18 vs torch 2.9")
    with pytest.raises(RuntimeError) as e:
        ph.DYNAMIC
    msg = str(e.value)
    assert "torchao<0.18" in msg, "the user needs something to actually do"
    assert "ScalingType" in msg


def test_instantiating_it_raises_too():
    ph = _make_torch_symbol_placeholder("SwizzleType", "d")
    with pytest.raises(RuntimeError):
        ph()


def test_it_never_pretends_to_be_a_real_value():
    """A stub returning None or 0 would flow into a float8 config and produce
    silently wrong behaviour instead of an error."""
    ph = _make_torch_symbol_placeholder("ScalingType", "d")
    for attr in ("DYNAMIC", "STATIC", "value", "name"):
        with pytest.raises(RuntimeError):
            getattr(ph, attr)


def test_repr_is_honest():
    ph = _make_torch_symbol_placeholder("ScalingType", "d")
    assert "placeholder" in repr(ph)


def test_it_is_marked_as_ours():
    ph = _make_torch_symbol_placeholder("ScalingType", "d")
    assert getattr(ph, "__unsloth_placeholder__", False) is True


# ---- the gating -----------------------------------------------------------


def test_it_is_a_no_op_when_torch_already_has_the_symbols():
    """On torch >= 2.10 overwriting the real enum with a raising placeholder
    would BREAK float8 rather than fix anything."""
    import torch.nn.functional as F

    added = [n for n in _TORCHAO_TORCH_SYMBOLS if not hasattr(F, n)]
    if not added:
        assert fix_torchao_torch_symbol_skew() is False


def test_a_healthy_torchao_is_left_alone():
    """0.17 and earlier guard their own import, so patching there would put a
    placeholder into torch for no reason."""
    if importlib.util.find_spec("torchao") is None:
        pytest.skip("torchao not installed")
    from importlib.metadata import version
    from packaging.version import Version

    if Version(version("torchao")) < Version("0.18.0"):
        import torch.nn.functional as F

        before = {n: hasattr(F, n) for n in _TORCHAO_TORCH_SYMBOLS}
        assert fix_torchao_torch_symbol_skew() is False
        after = {n: hasattr(F, n) for n in _TORCHAO_TORCH_SYMBOLS}
        assert before == after, "nothing may be added for a healthy torchao"


def test_no_torchao_means_nothing_to_do():
    if importlib.util.find_spec("torchao") is not None:
        pytest.skip("torchao is installed here")
    assert fix_torchao_torch_symbol_skew() is False


def test_it_never_raises():
    """It runs during `import unsloth`, so anything it raises replaces the
    problem it exists to prevent."""
    assert fix_torchao_torch_symbol_skew() in (True, False)


def test_calling_it_twice_is_stable():
    first = fix_torchao_torch_symbol_skew()
    second = fix_torchao_torch_symbol_skew()
    assert second is False or first == second


# ---- the wiring -----------------------------------------------------------


def test_it_runs_before_unsloth_zoo_is_imported():
    """unsloth_zoo pulls in transformers and therefore torchao, so calling the
    fix after that import would be pointless."""
    lines = GPU_INIT.read_text(encoding = "utf-8").splitlines()
    call = next(i for i, l in enumerate(lines) if l.strip() == "fix_torchao_torch_symbol_skew()")
    zoo = next(i for i, l in enumerate(lines) if l.strip() == "import unsloth_zoo")
    assert call < zoo, f"called at line {call + 1}, too late for line {zoo + 1}"


def test_it_is_imported_and_cleaned_up():
    src = GPU_INIT.read_text(encoding = "utf-8")
    assert "fix_torchao_torch_symbol_skew," in src, "not imported"
    assert (
        "del fix_torchao_torch_symbol_skew" in src
    ), "every other fix is deleted after use; this one must be too"


def test_the_symbol_list_matches_what_torchao_imports():
    """Taken from the whole installed package, not one file.

    Reading only mx_formats/mx_tensor.py misses scaled_grouped_mm, which
    float8_tensor.py imports on the path of a plain `import torchao`, so a list
    without it leaves the import exactly as dead.
    """
    assert set(_TORCHAO_TORCH_SYMBOLS) == {
        "ScalingType",
        "SwizzleType",
        "scaled_grouped_mm",
        "scaled_dot_product_attention",
    }


def test_symbols_torch_already_provides_are_never_replaced():
    """scaled_dot_product_attention exists on every supported torch, so a
    raising placeholder would break attention itself."""
    import torch.nn.functional as F
    import unsloth.import_fixes as IF

    real = F.scaled_dot_product_attention
    IF.fix_torchao_torch_symbol_skew()
    assert F.scaled_dot_product_attention is real


# ---- the fix actually unblocks the import --------------------------------


def test_the_real_torchao_018_import_line_is_unblocked(monkeypatch):
    """The decisive test: run torchao 0.18's own import line on this torch and
    show it goes from raising to succeeding. Everything above is gating."""
    import torch.nn.functional as F
    import unsloth.import_fixes as IF

    # conftest.py imports unsloth, so on an affected environment the placeholders are already on F. Drop them, or
    # the "before" half cannot raise, this test skips itself, and the guard test after it fails.
    for n in _TORCHAO_TORCH_SYMBOLS:
        if getattr(getattr(F, n, None), "__unsloth_placeholder__", False):
            delattr(F, n)

    # Gate on the two symbols the line below imports: `any` over the whole tuple is always true
    # (scaled_dot_product_attention always exists), which made this test skip on every machine.
    if all(hasattr(F, n) for n in ("ScalingType", "SwizzleType")):
        pytest.skip("this torch already provides ScalingType/SwizzleType")

    # the line as torchao 0.18 ships it
    line = "from torch.nn.functional import ScalingType, SwizzleType"

    with pytest.raises(ImportError):
        exec(line, {})

    monkeypatch.setattr(
        IF, "importlib_version", lambda name: "0.18.0" if name == "torchao" else "0"
    )
    try:
        assert IF.fix_torchao_torch_symbol_skew() is True
        exec(line, {})  # must not raise now
        from torch.nn.functional import ScalingType

        with pytest.raises(RuntimeError):
            ScalingType.DYNAMIC  # still refuses to be used
    finally:
        for n in _TORCHAO_TORCH_SYMBOLS:
            if getattr(getattr(F, n, None), "__unsloth_placeholder__", False):
                delattr(F, n)


def test_the_cleanup_in_the_test_above_is_real():
    """Guards the fixture, not the product: a failed delattr above would let
    every later test see a patched torch and pass vacuously."""
    import torch.nn.functional as F
    for n in _TORCHAO_TORCH_SYMBOLS:
        obj = getattr(F, n, None)
        assert not getattr(
            obj, "__unsloth_placeholder__", False
        ), f"a placeholder for {n} leaked out of a test"


# ---- the Mac / MLX path ---------------------------------------------------

INIT = ROOT / "unsloth" / "__init__.py"


def test_the_mlx_path_applies_the_fix_too():
    """The MLX branch imports unsloth_zoo directly and never reaches
    _gpu_init, so without the fix there Apple Silicon hits the same dead
    import."""
    lines = INIT.read_text(encoding = "utf-8").splitlines()
    fix = next(i for i, l in enumerate(lines) if "_fix_torchao()" in l)
    zoo = next(i for i, l in enumerate(lines) if l.strip() == "import unsloth_zoo")
    assert fix < zoo, "the fix must precede the MLX unsloth_zoo import"


def test_the_mlx_call_is_inside_the_mlx_branch():
    """_gpu_init already owns the GPU path, and calling it outside the branch
    would touch torch before the Apple Silicon detection wants it. Checked
    structurally: string offsets cannot tell "inside the if" from "after it".
    """
    import ast as _ast

    tree = _ast.parse(INIT.read_text(encoding = "utf-8"))

    def _mlx_if(nodes):
        for n in nodes:
            if isinstance(n, _ast.If) and _ast.dump(n.test).count("_IS_MLX"):
                return n
        return None

    node = _mlx_if(tree.body)
    assert node is not None, "no `if _IS_MLX:` branch found"
    inside = any(
        isinstance(c, _ast.Call) and getattr(c.func, "id", "") == "_fix_torchao"
        for stmt in node.body
        for c in _ast.walk(stmt)
    )
    assert inside, "the call must be INSIDE the `if _IS_MLX:` body"

    outside = any(
        isinstance(c, _ast.Call) and getattr(c.func, "id", "") == "_fix_torchao"
        for stmt in tree.body
        if stmt is not node
        for c in _ast.walk(stmt)
    )
    assert not outside, "it must not also run on the non-MLX path"


def test_the_mlx_call_cannot_break_the_import():
    """On Mac this runs first, so an exception here would replace a torchao
    problem with an unsloth problem."""
    src = INIT.read_text(encoding = "utf-8")
    i = src.index("_fix_torchao()")
    window = src[max(0, i - 400) : i + 200]
    assert "except Exception" in window and "pass" in window


# ---- version gating across the strings that actually ship ----------------


@pytest.mark.parametrize(
    "version,affected",
    [
        ("0.18.0", True),
        ("0.18.0+cu130", True),  # wheels carry a local version
        ("0.19.0.dev20260801", True),  # a dev build of a later release still has it
        ("1.0.0", True),  # future majors, until upstream fixes it
        ("0.17.0", False),  # guards its own import
        ("0.17.0+cu130", False),
        ("0.13.0", False),  # the floor in pyproject
    ],
)
def test_the_version_gate(version, affected):
    """The gate decides whether we touch torch at all, so it must cope with
    local versions and dev builds, not just clean releases."""
    from unsloth.import_fixes import Version
    assert (Version(version) >= Version("0.18.0")) is affected


def test_an_unparseable_version_is_not_patched(monkeypatch):
    """Leave torch alone rather than guess from a version we cannot parse."""
    import unsloth.import_fixes as IF

    monkeypatch.setattr(IF, "importlib_version", lambda name: "not-a-version-at-all")
    assert IF.fix_torchao_torch_symbol_skew() is False


def test_a_missing_version_is_not_patched(monkeypatch):
    import unsloth.import_fixes as IF

    def _boom(name):
        raise Exception("no metadata")

    monkeypatch.setattr(IF, "importlib_version", _boom)
    assert IF.fix_torchao_torch_symbol_skew() is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
