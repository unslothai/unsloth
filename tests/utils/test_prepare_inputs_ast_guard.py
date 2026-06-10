"""Import-free structural guard for the batched left-padding fix (#1066, #3699).

Parses unsloth/models/llama.py with `ast` and asserts the invariants that
PR #2216 and PR #4100 established inside `_fast_prepare_inputs_for_generation`:

  1. position_ids are derived from the 2D attention mask (cumsum - 1 with pads
     masked) when such a mask exists;
  2. any position_ids assignment based on cache_position is only the fallback
     branch, never unconditional and never before the mask-derived branch;
  3. the attention mask is never truncated to its last column
     (`attention_mask[:, [-1]]`, the pre-#2216 bug);
  4. every decoder family stays wired to the shared function via
     fix_prepare_inputs_for_generation.

This file deliberately imports nothing from unsloth so it runs on any Python
with no GPU, no torch and no installed package. It is the resilient layer
behind tests/utils/test_prepare_inputs_leftpad.py (behavioral, needs torch).
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LLAMA_PY = REPO_ROOT / "unsloth" / "models" / "llama.py"

FUNC_NAME = "_fast_prepare_inputs_for_generation"

# Model files that call fix_prepare_inputs_for_generation(...) and therefore
# share the guarded function. glm4_moe and falcon_h1 are intentionally absent:
# GLM4 MoE does not patch the Llama-compatible generation path (MLA attention)
# and falcon_h1 ships its own _fast_prepare_inputs_for_generation variant.
WIRED_MODEL_FILES = [
    "mistral.py",
    "gemma.py",
    "gemma2.py",
    "qwen2.py",
    "qwen3.py",
    "qwen3_moe.py",
    "cohere.py",
    "granite.py",
]


def _load_function():
    tree = ast.parse(LLAMA_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == FUNC_NAME:
            return node
    raise AssertionError(
        f"{FUNC_NAME} not found in {LLAMA_PY}; if it was renamed or moved, "
        "update this guard so batched left-padded generation stays protected"
    )


def _names_in(node):
    """All Name ids, attribute names and string constants in a subtree."""
    found = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            found.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            found.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            found.add(sub.value)
    return found


def _mentions_attention_mask(node):
    return any("attention_mask" in name for name in _names_in(node))


def _is_kwargs_position_ids_target(target):
    return (
        isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id == "kwargs"
        and isinstance(target.slice, ast.Constant)
        and target.slice.value == "position_ids"
    )


def _walk_with_paths(node, path = ()):
    yield node, path
    for child in ast.iter_child_nodes(node):
        yield from _walk_with_paths(child, path + (node,))


def _find_mask_branch(func):
    """The If whose test checks the 2D attention mask (dim() == 2)."""
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        test_names = _names_in(node.test)
        if "dim" in test_names and any("attention_mask" in n for n in test_names):
            return node
    return None


def test_mask_derived_position_ids_branch_exists():
    func = _load_function()
    branch = _find_mask_branch(func)
    assert branch is not None, (
        f"{FUNC_NAME} no longer has a branch testing the 2D attention mask "
        "(dim() == 2); position_ids must be derived per row from the mask for "
        "left-padded batches (see PR #4100 / issues #1066, #3699)"
    )

    body_names = set()
    for stmt in branch.body:
        body_names |= _names_in(stmt)
    assert "cumsum" in body_names and _mentions_attention_mask(
        ast.Module(body = branch.body, type_ignores = [])
    ), (
        "the attention-mask branch must compute position_ids via "
        "attention_mask.cumsum(...); reintroducing cache_position-based "
        "positions breaks left-padded batched generation (issue #3699)"
    )
    assert (
        "masked_fill_" in body_names or "masked_fill" in body_names
    ), "the attention-mask branch must mask pad positions (masked_fill on mask == 0)"
    assigns_kwargs = any(
        isinstance(stmt, ast.Assign)
        and any(_is_kwargs_position_ids_target(t) for t in stmt.targets)
        for stmt in ast.walk(ast.Module(body = branch.body, type_ignores = []))
    )
    assert assigns_kwargs, (
        "the attention-mask branch must store the derived positions into " 'kwargs["position_ids"]'
    )


def test_cache_position_only_used_as_fallback_for_position_ids():
    func = _load_function()
    branch = _find_mask_branch(func)
    assert branch is not None

    orelse_nodes = set()
    for stmt in branch.orelse:
        for sub in ast.walk(stmt):
            orelse_nodes.add(id(sub))

    offenders = []
    for node, path in _walk_with_paths(func):
        if not isinstance(node, ast.Assign):
            continue
        if not any(_is_kwargs_position_ids_target(t) for t in node.targets):
            continue
        value_names = _names_in(node.value)
        # Direct use of cache_position, or the local alias `cp` the current
        # implementation builds from it inside the fallback branch.
        derives_from_cache_position = any("cache_position" in n for n in value_names) or bool(
            value_names & {"cp"}
        )
        if derives_from_cache_position and id(node) not in orelse_nodes:
            offenders.append(ast.unparse(node))

    assert not offenders, (
        'kwargs["position_ids"] must never be assigned from cache_position '
        "outside the fallback (orelse) of the 2D attention-mask branch; "
        "cache_position counts left-pad tokens, so padded rows generate "
        f"garbage (issues #1066, #3699). Offending assignments: {offenders}"
    )


def test_attention_mask_never_truncated_to_last_column():
    func = _load_function()
    offenders = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not isinstance(value, ast.Subscript):
            continue
        if not _mentions_attention_mask(value.value):
            continue
        # Match a trailing [-1]-style column selection: [:, [-1]] or [:, -1:]
        sl = value.slice
        if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
            col = sl.elts[1]
            is_last_col_list = (
                isinstance(col, ast.List)
                and len(col.elts) == 1
                and isinstance(col.elts[0], ast.UnaryOp)
            )
            is_last_col_slice = (
                isinstance(col, ast.Slice)
                and col.lower is not None
                and isinstance(col.lower, ast.UnaryOp)
                and getattr(getattr(col.lower, "operand", None), "value", None) == 1
                and col.upper is None
            )
            if is_last_col_list or is_last_col_slice:
                offenders.append(ast.unparse(node))
    assert not offenders, (
        "the 2D attention mask must not be truncated to its last column; this "
        "was the pre-#2216 bug that drops padding information in cached decode "
        f"(issue #1066). Offending assignments: {offenders}"
    )


def test_model_families_stay_wired_to_shared_prepare_inputs():
    missing = []
    for fname in WIRED_MODEL_FILES:
        path = REPO_ROOT / "unsloth" / "models" / fname
        if not path.exists():
            continue
        if "fix_prepare_inputs_for_generation(" not in path.read_text():
            missing.append(fname)
    assert not missing, (
        "these model files no longer call fix_prepare_inputs_for_generation, "
        "so they lose the guarded left-padding-safe prepare_inputs path: "
        f"{missing}"
    )
