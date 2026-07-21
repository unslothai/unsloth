"""Regression guard for batched left-padded generation (issues #1066, #3699).

Guards `_fast_prepare_inputs_for_generation` (unsloth/models/llama.py),
shared by every decoder family wired through fix_prepare_inputs_for_generation,
against two historical bugs:
  (a) 2D attention mask truncated to its last column during cached decode,
      losing padding info (fixed by #2216);
  (b) position_ids taken from cache_position (which counts left-pad tokens),
      so padded rows generated garbage (fixed by #4100).

Two CPU-only deterministic layers: (1) AST structural checks (no unsloth
import); (2) behavioral checks calling the real function with synthetic
left-padded masks and fake caches. Companion GPU check:
tests/utils/test_batched_leftpad_generation_gpu.py
"""

import ast
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LLAMA_PY = REPO_ROOT / "unsloth" / "models" / "llama.py"

FUNC_NAME = "_fast_prepare_inputs_for_generation"


# --------------------------------------------------------------------------
# Layer 1: AST structural guard (stdlib only, no unsloth import)
# --------------------------------------------------------------------------

# Model files that call fix_prepare_inputs_for_generation(...) and share the
# guarded function. glm4_moe (MLA attention, different path) and falcon_h1
# (its own variant) are intentionally absent.
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
    assert assigns_kwargs, 'the attention-mask branch must store the derived positions into kwargs["position_ids"]'


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
        derives_from_cache_position = any(
            "cache_position" in n for n in value_names
        ) or bool(value_names & {"cp"})
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


# --------------------------------------------------------------------------
# Layer 2: behavioral guard (calls the real function, lazy unsloth import)
# --------------------------------------------------------------------------

PAST_LEN = 4

# Three rows with different amounts of left padding (0 = pad).
MASK = torch.tensor(
    [
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
    ],
    dtype = torch.long,
)
BS, SEQ = MASK.shape

# Per-row positions: cumsum(-1) - 1 with pad slots filled with 1.
EXPECTED_PREFILL_POSITIONS = torch.tensor(
    [
        [1, 1, 0, 1, 2],
        [0, 1, 2, 3, 4],
        [1, 0, 1, 2, 3],
    ],
    dtype = torch.long,
)


class FakeDynamicCache:
    """Minimal stand-in for transformers DynamicCache with a non-empty cache."""

    def __init__(self, seq_length):
        self._seq_length = seq_length

    def __len__(self):
        return 1

    def get_seq_length(self):
        return self._seq_length


class FakeModel:
    """Bare-minimum `self` for _fast_prepare_inputs_for_generation."""

    dtype = torch.float32
    config = None


class FakeModelWith4DMask(FakeModel):
    """Variant exposing the HF 4D mask builder; records what it receives."""

    def __init__(self):
        self.mask_calls = []

    def _prepare_4d_causal_attention_mask_with_cache_position(
        self,
        attention_mask,
        sequence_length,
        target_length,
        dtype,
        device,
        cache_position,
        batch_size,
        config = None,
        past_key_values = None,
    ):
        self.mask_calls.append(
            {
                "mask_shape": tuple(attention_mask.shape)
                if attention_mask is not None
                else None,
                "sequence_length": sequence_length,
                "target_length": target_length,
                "batch_size": batch_size,
            }
        )
        return torch.zeros((batch_size, 1, sequence_length, target_length), dtype = dtype)


def _prepare(model, input_ids, attention_mask, **kwargs):
    from unsloth.models import llama as llama_mod
    return llama_mod._fast_prepare_inputs_for_generation(
        model, input_ids, attention_mask = attention_mask, **kwargs
    )


def test_prefill_position_ids_derived_from_left_padded_mask():
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    result = _prepare(FakeModel(), input_ids, MASK)

    position_ids = result.get("position_ids", None)
    assert (
        position_ids is not None
    ), "prefill with a left-padded 2D attention mask must populate position_ids"
    assert torch.equal(position_ids.long().cpu(), EXPECTED_PREFILL_POSITIONS), (
        "prefill position_ids must be derived per row from the attention mask "
        "(cumsum - 1, pads masked), so each row starts counting at its first "
        f"real token; got {position_ids.tolist()}"
    )
    assert result["input_ids"].shape == (BS, SEQ)


@pytest.mark.parametrize("pass_cache_position", [True, False])
def test_cached_decode_position_ids_ignore_left_padding(pass_cache_position):
    # Decode step: PAST_LEN tokens cached, current token is the mask's last
    # column. Row 0 has 2 pads, so its current token sits at logical position 2,
    # NOT at cache_position == PAST_LEN. This is exactly issue #3699.
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    kwargs = {"past_key_values": FakeDynamicCache(PAST_LEN)}
    if pass_cache_position:
        kwargs["cache_position"] = torch.arange(PAST_LEN, PAST_LEN + 1)

    result = _prepare(FakeModel(), input_ids, MASK, **kwargs)

    assert result["input_ids"].shape == (
        BS,
        1,
    ), "cached decode must slice input_ids to the last token only"
    position_ids = result.get("position_ids", None)
    assert position_ids is not None
    expected = torch.tensor([[2], [4], [3]], dtype = torch.long)
    assert torch.equal(position_ids.long().cpu().reshape(BS, 1), expected), (
        "left-padded cached decode must derive per-row position_ids from the "
        "attention mask, not from cache_position which counts pad tokens; got "
        f"{position_ids.tolist()}, expected {expected.tolist()} "
        "(row 0 has 2 pads: its position must be 2, not 4)"
    )


def test_cached_decode_does_not_truncate_2d_attention_mask():
    # Without a 4D mask builder the original 2D mask must survive untouched.
    # The historical bug replaced it with attention_mask[:, [-1]].
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    result = _prepare(
        FakeModel(), input_ids, MASK, past_key_values = FakeDynamicCache(PAST_LEN)
    )
    mask_out = result["attention_mask"]
    assert mask_out is not None
    assert mask_out.dim() != 2 or mask_out.shape[-1] == SEQ, (
        "the 2D attention mask must not be truncated to its last column during "
        f"cached decode (got shape {tuple(mask_out.shape)}); padding rows lose "
        "their pad information otherwise"
    )


def test_cached_decode_4d_mask_builder_receives_full_target_length():
    model = FakeModelWith4DMask()
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    result = _prepare(
        model, input_ids, MASK, past_key_values = FakeDynamicCache(PAST_LEN)
    )
    assert len(model.mask_calls) == 1
    call = model.mask_calls[0]
    assert call["mask_shape"] == (BS, SEQ), (
        "the 4D mask builder must receive the full 2D padding mask, not a "
        f"truncated one (got {call['mask_shape']})"
    )
    assert call["sequence_length"] == 1
    assert call["target_length"] == SEQ, (
        "target_length must cover the whole mask so padded positions stay "
        f"masked (got {call['target_length']})"
    )
    assert result["attention_mask"].dim() == 4


def test_caller_supplied_position_ids_are_passed_through():
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    custom = torch.full((BS, SEQ), 7, dtype = torch.long)
    result = _prepare(FakeModel(), input_ids, MASK, position_ids = custom)
    assert torch.equal(
        result["position_ids"], custom
    ), "caller-supplied position_ids must not be overwritten"


def test_legacy_tuple_cache_still_takes_cached_decode_path():
    # Legacy cache format: tuple of (K, V) per layer; past length from K.shape[-2].
    k = torch.zeros((BS, 1, PAST_LEN, 8))
    legacy_cache = ((k, k.clone()),)
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    result = _prepare(FakeModel(), input_ids, MASK, past_key_values = legacy_cache)
    assert result["input_ids"].shape == (BS, 1)
    expected = torch.tensor([[2], [4], [3]], dtype = torch.long)
    assert torch.equal(result["position_ids"].long().cpu().reshape(BS, 1), expected)
