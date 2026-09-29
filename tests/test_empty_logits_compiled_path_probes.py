# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth#409, compiled path: unsloth_zoo.compiler generates its OWN empty-logits sentinel.

`unsloth_compile_transformers()` hands the trainers source text built from
`compiler._cross_entropy_code`, which carries a second `EmptyLogits` that never goes through
`unsloth/models/_utils.py`. Fixing the singleton there leaves that copy claiming
`__dataclass_fields__`, so torch's `_apply_to_tensors` (the FSDP2 mixed-precision output cast)
still calls `dataclasses.replace` on it and dies. The generated text is read off the installed
zoo and exec'd, because the module is never importable without an accelerator.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
import torch
from packaging.version import Version
from torch.distributed.utils import _apply_to_tensors


# Not tensor dunders: the loop after the class binds those as real instance attributes, so
# only names outside `dir(torch.Tensor)` ever reach `__getattr__`.
PROTOCOL_DUNDERS = (
    "__dataclass_fields__",
    "__fields__",
    "__attrs_attrs__",
    "__get_validators__",
    "__pydantic_fields__",
    "__dataclass_params__",
)

# The sentinel exactly as unsloth_zoo 2026.9.4 generated it, kept verbatim so the probes below
# are shown to have teeth on every machine, whatever zoo happens to be installed.
PRE_FIX_SENTINEL_SOURCE = """
LOGITS_ERROR_STRING = "Unsloth: Logits are empty, set UNSLOTH_RETURN_LOGITS"
def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr): return return_none if attr == "to" else raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
    def __reduce__(self): return (type(self), ())
    def __eq__(self, other): return type(other).__name__ == "EmptyLogits"
    __hash__ = object.__hash__
"""


# The first unsloth_zoo release that generates a fixed sentinel, i.e. the one that carries
# unslothai/unsloth-zoo#1259. Published, and pyproject.toml's floor now names it too.
#
# This deliberately does NOT read the `unsloth_zoo>=` floor out of pyproject.toml, which is
# what it used to do. The two agree again today, but the coupling is only ever correct while
# the pin and the fix name the same release, and they came apart once already: the pin sat at
# 2026.9.4 for as long as #1259 was unpublished, and 2026.9.4 is exactly the zoo that still
# generates the BROKEN sentinel, so a gate reading the pin would have run these probes against
# a zoo carrying the bug and failed on a correct tree. What the probes assert is a property of
# the INSTALLED zoo, so the gate names the release that fixes it, directly.
#
# Where an older zoo is what is actually installed, every probe below skips and the always-on
# canary at the bottom of this file is what keeps the assertions honest.
ZOO_RELEASE_WITH_GENERATED_SENTINEL_FIX = Version("2026.9.5")


def _installed_zoo() -> tuple[Path, Version]:
    spec = importlib.util.find_spec("unsloth_zoo")  # locates without importing: no GPU needed
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("unsloth_zoo is not installed")
    compiler = Path(spec.submodule_search_locations[0]) / "compiler.py"
    if not compiler.is_file():
        pytest.skip(f"unsloth_zoo carries no compiler.py at {compiler}")
    from importlib.metadata import PackageNotFoundError, version

    try:
        return compiler, Version(version("unsloth_zoo"))
    except PackageNotFoundError:
        pytest.skip("unsloth_zoo has no installed distribution metadata to version-gate on")


def _generated_sentinel_source(compiler: Path) -> str:
    """The sentinel slice of `_cross_entropy_code`, pulled out of the file as text."""
    module = ast.parse(compiler.read_text())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "_cross_entropy_code" for target in node.targets
        ):
            code = ast.literal_eval(node.value)
            break
    else:
        pytest.fail(
            f"{compiler} defines no module-level `_cross_entropy_code`; the compiled trainers "
            f"are built from somewhere else now and this probe is aimed at nothing"
        )
    start = code.index("LOGITS_ERROR_STRING")
    end = code.index("EMPTY_LOGITS = EmptyLogits()")
    return code[start:end]


def _build(source: str):
    namespace: dict = {"torch": torch}
    exec(compile(source, "<unsloth_zoo _cross_entropy_code>", "exec"), namespace)
    return namespace["EmptyLogits"]()


@pytest.fixture(scope = "module")
def generated_sentinel():
    compiler, installed = _installed_zoo()
    if installed < ZOO_RELEASE_WITH_GENERATED_SENTINEL_FIX:
        pytest.skip(
            f"installed unsloth_zoo {installed} still generates the pre-fix sentinel; it is "
            f"fixed from {ZOO_RELEASE_WITH_GENERATED_SENTINEL_FIX} "
            f"(unslothai/unsloth-zoo#1259) onwards, which pyproject.toml's floor now "
            f"requires, so upgrade unsloth_zoo here to run these. The canary test in this "
            f"file runs unconditionally and shows these probes have teeth."
        )
    return _build(_generated_sentinel_source(compiler))


@pytest.mark.parametrize("name", PROTOCOL_DUNDERS)
def test_the_generated_sentinel_does_not_claim_a_protocol_dunder(generated_sentinel, name):
    assert not hasattr(generated_sentinel, name), (
        f"the sentinel unsloth_zoo.compiler generates answers hasattr({name!r}); a library that "
        f"duck-types on it takes a branch the sentinel cannot honour"
    )


def test_the_fsdp2_output_cast_leaves_the_generated_sentinel_alone(generated_sentinel):
    """`_apply_to_tensors` is what FSDP2 mixed precision runs over every forward's outputs."""
    outputs = {"loss": torch.tensor(1.0), "logits": generated_sentinel}
    cast = _apply_to_tensors(lambda tensor: tensor.to(torch.bfloat16), outputs)
    assert cast["loss"].dtype is torch.bfloat16
    assert cast["logits"] is generated_sentinel


def test_the_generated_sentinel_still_absorbs_the_to_call(generated_sentinel):
    """accelerate calls `.to(device)` on outputs, so `to` stays special-cased ahead of the guard."""
    assert generated_sentinel.to("cpu") is None


def test_an_ordinary_attribute_on_the_generated_sentinel_still_explains_itself(generated_sentinel):
    with pytest.raises(NotImplementedError) as excinfo:
        generated_sentinel.shape()
    assert "UNSLOTH_RETURN_LOGITS" in str(excinfo.value)


def test_these_probes_fail_on_the_sentinel_zoo_used_to_generate():
    """Mutation control: the pre-fix copy must trip both probes, so a skip above is the only
    way they stay quiet. Without this, a zoo that stopped shipping the fix would look green."""
    stale = _build(PRE_FIX_SENTINEL_SOURCE)
    assert hasattr(stale, "__dataclass_fields__")
    with pytest.raises(TypeError, match = "replace"):
        _apply_to_tensors(lambda tensor: tensor, {"logits": stale})
