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

"""`import unsloth` must survive a torchao that wants an aten op this torch lacks.

torchao 0.18 registers a float8 handler at module scope::

    @implements([aten._grouped_mm.default])
    def float8_grouped_mm(func, types, args, kwargs):

`aten::_grouped_mm` arrived in torch 2.8, so on anything older the attribute
lookup raises before the decorator is applied::

    AttributeError: '_OpNamespace' 'aten' object has no attribute '_grouped_mm'

transformers imports torchao from `modeling_utils` under
`is_torchao_available()`, so this kills `import transformers` and therefore
`import unsloth`. Seen on Colab in Granite4.0, which pins torch 2.7.1 via uv
while resolving a current torchao.

This is the sibling of the `ScalingType` skew already covered by
test_torchao_subprocess_fix.py, but it is NOT the same bug: the lookup goes
through `torch.ops`, so adding names to `torch.nn.functional` does nothing for
it. The two are fixed independently and tested independently.

The tests below run on a torch that HAS the operator, so they mostly assert the
safety properties -- that the fix stays out of the way, refuses to guess, and
never registers twice. That is the behaviour most likely to cause damage, since
registering into the `aten` namespace on a healthy torch would be far worse
than the crash being fixed.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from unsloth import import_fixes as IF  # noqa: E402


# ---- the guard ----------------------------------------------------------


def test_a_present_op_is_reported_as_present():
    assert IF._torch_op_is_missing("aten", "mm") is False


def test_an_absent_op_is_reported_as_missing():
    assert IF._torch_op_is_missing("aten", "_unsloth_definitely_not_an_op") is True


def test_an_op_in_an_unknown_namespace_is_missing():
    """`getattr(torch.ops, "nope")` happily hands back a namespace object, but
    looking an op up inside it still raises AttributeError -- and an op in a
    namespace that does not exist is, correctly, missing.

    This only stays harmless because the sole caller passes "aten", which
    always exists, so the answer is never used to decide whether to register
    into a namespace that is not there.
    """
    assert IF._torch_op_is_missing("unsloth_no_such_namespace", "_grouped_mm") is True


def test_a_non_attribute_error_means_do_not_touch_it(monkeypatch):
    """Only a plain AttributeError is evidence of absence. Anything else means
    we could not tell, and the safe answer is to leave torch alone."""

    class Exploding:
        def __getattr__(self, item):
            raise RuntimeError("something else entirely")

    monkeypatch.setattr(torch.ops, "aten", Exploding(), raising = False)
    assert IF._torch_op_is_missing("aten", "_grouped_mm") is False


# ---- the fix, on a torch that does not need it --------------------------


@pytest.mark.skipif(
    not hasattr(torch.ops.aten, "_grouped_mm"),
    reason = "this torch is missing the op; see the live test",
)
def test_it_does_nothing_when_torch_already_has_the_op():
    """The property that matters most here. Registering a placeholder over a
    real aten operator would replace a working grouped matmul with one that
    raises."""
    before = torch.ops.aten._grouped_mm
    assert IF._ensure_aten_grouped_mm("detail") is False
    assert torch.ops.aten._grouped_mm is before


@pytest.mark.skipif(
    not hasattr(torch.ops.aten, "_grouped_mm"),
    reason = "this torch is missing the op; see the live test",
)
def test_the_real_operator_still_works_afterwards():
    IF._ensure_aten_grouped_mm("detail")
    assert callable(torch.ops.aten._grouped_mm)
    # The schema is torch's own, not a placeholder we substituted.
    assert "offs" in str(torch.ops.aten._grouped_mm.default._schema)


def test_it_never_registers_twice(monkeypatch):
    """A second call must be a no-op even if the op still looks missing --
    re-defining a schema raises, and the fix should not depend on that."""
    monkeypatch.setattr(IF, "_aten_grouped_mm_library", object())
    assert IF._ensure_aten_grouped_mm("detail") is False


# ---- the schema ---------------------------------------------------------


def test_the_placeholder_schema_matches_upstream():
    """Read off `torch.ops.aten._grouped_mm.default._schema` on torch 2.9.

    If it drifts, torchao's decorator still resolves but anything that
    introspects the signature sees a lie, so pin it.
    """
    s = IF._ATEN_GROUPED_MM_SCHEMA
    assert s.startswith("_grouped_mm(Tensor self, Tensor mat2")
    for kwarg in ("Tensor? offs=None", "Tensor? bias=None", "ScalarType? out_dtype=None"):
        assert kwarg in s, kwarg
    assert s.endswith("-> Tensor")


def test_the_schema_parses_as_a_real_torch_schema():
    """Registering it under a private namespace proves torch accepts the
    string, without touching `aten` on a machine that does not need help."""
    lib = torch.library.Library("unsloth_schema_probe", "FRAGMENT")
    try:
        lib.define(IF._ATEN_GROUPED_MM_SCHEMA)
        assert hasattr(torch.ops.unsloth_schema_probe, "_grouped_mm")
    finally:
        del lib


def test_the_placeholder_refuses_to_compute_rather_than_guessing():
    """A placeholder that returned a plausible tensor would be the worst
    outcome: a silently wrong grouped matmul is not debuggable."""
    lib = torch.library.Library("unsloth_refuse_probe", "FRAGMENT")
    try:
        lib.define(IF._ATEN_GROUPED_MM_SCHEMA)

        def _refuse(
            self,
            mat2,
            offs = None,
            bias = None,
            out_dtype = None,
        ):
            raise RuntimeError("Unsloth: placeholder, cannot be used")

        lib.impl("_grouped_mm", _refuse, "CompositeExplicitAutograd")
        a = torch.ones(4, 4)
        b = torch.ones(1, 4, 4)
        with pytest.raises(RuntimeError, match = "placeholder"):
            torch.ops.unsloth_refuse_probe._grouped_mm(a, b)
    finally:
        del lib


# ---- wiring -------------------------------------------------------------


def test_the_subprocess_fix_covers_the_op_too():
    """vLLM inspects architectures in a child process that never sees an
    in-process patch, so the generated sitecustomize has to carry this fix as
    well as the functional-symbol one."""
    src = IF._subprocess_sitecustomize_source()
    assert "torch.ops.aten._grouped_mm" in src
    assert "_grouped_mm(Tensor self, Tensor mat2" in src
    assert "CompositeExplicitAutograd" in src


def test_the_subprocess_fix_keeps_the_library_alive():
    """A torch.library.Library deregisters its schema when collected, so a
    local variable would silently undo the fix."""
    src = IF._subprocess_sitecustomize_source()
    assert "global _ATEN_LIBRARY" in src
    assert "_ATEN_LIBRARY = None" in src


def test_the_generated_sitecustomize_is_valid_python():
    import ast
    ast.parse(IF._subprocess_sitecustomize_source())


def test_the_subprocess_fix_never_aborts_interpreter_startup():
    """It runs in every child process on the machine. Raising there would be
    far worse than the import error it fixes."""
    src = IF._subprocess_sitecustomize_source()
    tail = src[src.index("_chain_to_the_real_sitecustomize()") :]
    assert "try:" in tail and "except Exception:" in tail


def test_the_symbol_fix_reports_the_op_as_one_of_its_patches():
    """`fix_torchao_torch_symbol_skew` returns whether it did anything; the
    op half has to count, or a torch needing only that fix reports False."""
    import inspect

    src = inspect.getsource(IF.fix_torchao_torch_symbol_skew)
    assert "_ensure_aten_grouped_mm" in src
    assert 'patched.append("aten::_grouped_mm")' in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
