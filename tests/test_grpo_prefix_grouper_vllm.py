# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO PrefixGrouper also runs under vLLM (fast_inference=True), and one self-verify outlier on an
already verified signature no longer turns it off for the run.

Both PrefixGrouper sites used to skip vLLM: the no-grad old / ref log-prob forward in
rl_replacements.py and the gradient forward in zoo's grpo_accumulated_loss (rewritten here at
generation time). UNSLOTH_GRPO_PREFIX_GROUPER_VLLM=0 restores the skip.
"""

import ast
import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RL_REPLACEMENTS = ROOT / "unsloth" / "models" / "rl_replacements.py"
TEXT = RL_REPLACEMENTS.read_text(encoding = "utf-8")


def _top_level(names):
    tree = ast.parse(TEXT, filename = str(RL_REPLACEMENTS))
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    namespace = {"re": re}
    exec(compile(ast.Module(body = body, type_ignores = []), str(RL_REPLACEMENTS), "exec"), namespace)
    return namespace


_PATCH = _top_level(
    {
        "_GRPO_PREFIX_GROUPER_VLLM_GATE",
        "_patch_grpo_accumulated_loss_prefix_grouper_vllm_gate",
        "_GRPO_PREFIX_GROUPER_GRAD_KILL",
        "_patch_grpo_accumulated_loss_prefix_grouper_kill",
    }
)
patch_vllm_gate = _PATCH["_patch_grpo_accumulated_loss_prefix_grouper_vllm_gate"]
patch_kill = _PATCH["_patch_grpo_accumulated_loss_prefix_grouper_kill"]

# The shape of zoo's gate line (unsloth_zoo/rl_replacements.py, grpo_accumulated_loss).
_ZOO_LIKE = """
def grpo_accumulated_loss(trainer, unwrapped_model, _pg_funcs, _need_entropy = False):
    _pg_engage = bool(_pg_funcs) and not getattr(trainer, "use_vllm", False) and not _need_entropy
    if _pg_engage:
        return unwrapped_model(prefix_seg_info = 1)
    return None
"""


class _Trainer:
    def __init__(self, use_vllm):
        self.use_vllm = use_vllm


def _run(
    source,
    flag,
    use_vllm,
    need_entropy = False,
):
    namespace = {"UNSLOTH_GRPO_PREFIX_GROUPER_VLLM_ON": flag}
    exec(source, namespace)
    model = lambda **kw: "pg"  # noqa: E731
    return namespace["grpo_accumulated_loss"](_Trainer(use_vllm), model, ("f",), need_entropy)


def test_zoo_vllm_gate_patch_engages_under_vllm():
    patched = patch_vllm_gate(_ZOO_LIKE)
    assert patched != _ZOO_LIKE
    assert patch_vllm_gate(patched) == patched  # idempotent
    assert _run(patched, True, use_vllm = True) == "pg"
    assert _run(patched, True, use_vllm = False) == "pg"
    # Kill switch restores zoo's skip; other exclusions are untouched.
    assert _run(patched, False, use_vllm = True) is None
    assert _run(patched, False, use_vllm = False) == "pg"
    assert _run(patched, True, use_vllm = True, need_entropy = True) is None


def test_unpatched_zoo_skips_vllm():
    assert _run(_ZOO_LIKE, True, use_vllm = True) is None


def test_zoo_vllm_gate_patch_leaves_other_sources():
    no_pg = "def grpo_accumulated_loss(trainer):\n    return getattr(trainer, 'use_vllm', False)\n"
    assert patch_vllm_gate(no_pg) == no_pg
    # A zoo that already dropped the vLLM skip is left as is.
    no_skip = _ZOO_LIKE.replace(' and not getattr(trainer, "use_vllm", False)', "")
    assert patch_vllm_gate(no_skip) == no_skip


def test_installed_zoo_vllm_gate_is_patched():
    zoo = pytest.importorskip("unsloth_zoo.rl_replacements")
    function = getattr(zoo, "grpo_accumulated_loss", None)
    if function is None:
        function = zoo.RL_REPLACEMENTS.get("grpo_accumulated_loss")
    source = inspect.getsource(function)
    if "prefix_seg_info" not in source:
        pytest.skip("this unsloth_zoo has no gradient PrefixGrouper")
    patched = patch_vllm_gate(source)
    gate = [
        line
        for line in patched.splitlines()
        if re.search(r"_pg_engage\s*=\s*bool\(_pg_funcs\)", line)
    ]
    assert len(gate) == 1
    if "use_vllm" in gate[0]:
        assert "UNSLOTH_GRPO_PREFIX_GROUPER_VLLM_ON or not getattr(trainer" in gate[0]
    compile(patched, "<patched grpo_accumulated_loss>", "exec")


def test_patch_is_chained_into_the_generated_trainer():
    assert "_patch_grpo_accumulated_loss_prefix_grouper_vllm_gate(\n" in TEXT
    # The generated cache inlines the function without this module's globals, so it defines the flag.
    assert (
        "UNSLOTH_GRPO_PREFIX_GROUPER_VLLM_ON = _unsloth_os.environ.get('UNSLOTH_GRPO_PREFIX_GROUPER_VLLM', '1')"
        in TEXT
    )


def test_nograd_gate_engages_under_vllm():
    node = next(
        n
        for n in ast.parse(TEXT).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "grpo_trainer__get_per_token_logps_and_entropies"
    )
    source = ast.get_source_segment(TEXT, node)
    start = source.index("_pg_engage = (")
    gate = source[start : source.index("if _pg_engage:", start)]
    assert 'UNSLOTH_GRPO_PREFIX_GROUPER_VLLM_ON or not getattr(self, "use_vllm", False)' in gate
    assert re.search(r"and not getattr\(self, \"use_vllm\", False\)", gate) is None


# The shape of zoo's gradient self-verify mismatch branch.
_ZOO_VERIFY = """
def grpo_accumulated_loss(unwrapped_model, _pg_layout, _pg_diff, _PG_TOL_KILL = 1.5):
    # prefix_seg_info
    _pg_u = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_unsafe", None)
    if _pg_u is None:
        _pg_u = set()
    if _pg_diff >= _PG_TOL_KILL:
        _pg_u.add(_pg_layout.signature)
        unwrapped_model._unsloth_prefix_grouper_grad_unsafe = _pg_u
    return getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_unsafe", None)
"""


class _Layout:
    signature = (1, (8,))


class _Model:
    pass


def _verify(source, verified, diff):
    namespace = {}
    exec(source, namespace)
    model = _Model()
    if verified is not None:
        model._unsloth_prefix_grouper_grad_verified = verified
    return namespace["grpo_accumulated_loss"](model, _Layout(), diff)


def test_grad_kill_spares_a_verified_signature():
    patched = patch_kill(_ZOO_VERIFY)
    assert patched != _ZOO_VERIFY
    assert patch_kill(patched) == patched  # idempotent
    # Never verified: a large mismatch still marks it unsafe, as before.
    assert _verify(patched, None, 6.4) == {(1, (8,))}
    assert _verify(patched, {(2, (8, 8)): (100, 50)}, 6.4) == {(1, (8,))}
    # Verified at a smaller T: the same mismatch only falls back for this batch.
    assert _verify(patched, {(1, (8,)): (5642, 2576)}, 6.4) is None
    # Borderline mismatches never kill.
    assert _verify(patched, None, 1.0) is None
    # Unpatched zoo kills the verified signature.
    assert _verify(_ZOO_VERIFY, {(1, (8,)): (5642, 2576)}, 6.4) == {(1, (8,))}


def test_installed_zoo_grad_kill_is_patched():
    zoo = pytest.importorskip("unsloth_zoo.rl_replacements")
    function = getattr(zoo, "grpo_accumulated_loss", None)
    if function is None:
        function = zoo.RL_REPLACEMENTS.get("grpo_accumulated_loss")
    source = inspect.getsource(function)
    if "_PG_TOL_KILL" not in source:
        pytest.skip("this unsloth_zoo has no gradient PrefixGrouper self-verify")
    patched = patch_kill(source)
    assert patched.count('"_unsloth_prefix_grouper_grad_verified", None) or {}):') == 1
    assert re.search(r"if\s+_pg_diff\s*>=\s*_PG_TOL_KILL\s*:", patched) is None
    compile(patched, "<patched grpo_accumulated_loss>", "exec")


def test_kill_patch_is_chained_into_the_generated_trainer():
    assert "_patch_grpo_accumulated_loss_prefix_grouper_kill(\n" in TEXT


def test_nograd_kill_spares_a_verified_signature():
    node = next(
        n
        for n in ast.parse(TEXT).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "grpo_trainer__get_per_token_logps_and_entropies"
    )
    source = ast.get_source_segment(TEXT, node)
    assert re.search(r"if _pg_diff >= _PG_TOL_KILL:", source) is None
    assert re.search(
        r"if _pg_diff >= _PG_TOL_KILL and _pg_layout\.signature not in \(\s*_pg_v if isinstance\(_pg_v, dict\) else \{\}",
        source,
    )
