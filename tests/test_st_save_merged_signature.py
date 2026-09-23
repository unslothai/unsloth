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

"""`save_pretrained_merged` must mean the same thing on every model type.

FastSentenceTransformer bound the name to `(self, save_directory, **kwargs)`
while FastLanguageModel takes `tokenizer` and `save_method` positionally, so
the documented positional call raised TypeError on embedding models.
`save_method` must be honoured, not accepted and dropped.
"""

import ast
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ST_PY = REPO_ROOT / "unsloth" / "models" / "sentence_transformer.py"
SAVE_PY = REPO_ROOT / "unsloth" / "save.py"


def _defs(name, path):
    """Matching definitions in SOURCE order.

    `ast.walk` is breadth-first, so nested closures come back in an order
    that has nothing to do with the file, and indexing into it silently
    tested the wrong one.
    """
    src = path.read_text(encoding="utf-8")
    found = [
        n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    return sorted(found, key=lambda n: n.lineno)


def _params(node):
    return [a.arg for a in node.args.args]


def test_both_definitions_exist():
    assert (
        len(_defs("_save_pretrained_merged", ST_PY)) == 2
    ), "two closures bind this name; both must be fixed or one model path keeps the old signature"


@pytest.mark.parametrize("i", [0, 1])
def test_signature_matches_fastlanguagemodel(i):
    node = _defs("_save_pretrained_merged", ST_PY)[i]
    assert _params(node)[:4] == ["self", "save_directory", "tokenizer", "save_method"]


@pytest.mark.parametrize("i", [0, 1])
def test_still_accepts_arbitrary_keywords(i):
    node = _defs("_save_pretrained_merged", ST_PY)[i]
    assert node.args.kwarg is not None, "**kwargs must survive"


@pytest.mark.parametrize("i", [0, 1])
def test_defaults_keep_the_old_keyword_call_working(i):
    node = _defs("_save_pretrained_merged", ST_PY)[i]
    defaults = {
        a.arg: d for a, d in zip(node.args.args[-len(node.args.defaults) :], node.args.defaults)
    }
    assert isinstance(defaults["tokenizer"], ast.Constant)
    assert defaults["tokenizer"].value is None
    assert defaults["save_method"].value == "merged_16bit"


def test_the_reference_signature_is_what_we_matched():
    """Guards the premise: if FastLanguageModel's signature moves, these
    two have to move with it, and the test should say so."""
    node = _defs("unsloth_save_pretrained_merged", SAVE_PY)
    assert node, "unsloth_save_pretrained_merged not found"
    assert _params(node[0])[:4] == ["self", "save_directory", "tokenizer", "save_method"]


# ---- save_method is honoured, not swallowed -------------------------------


def test_merge_and_unload_path_refuses_an_unsupported_method():
    src = ST_PY.read_text(encoding="utf-8")
    node = _defs("_save_pretrained_merged", ST_PY)[0]
    body = ast.get_source_segment(src, node)
    assert "NotImplementedError" in body
    assert "merged_16bit" in body


def test_forwarding_path_passes_save_method_through():
    src = ST_PY.read_text(encoding="utf-8")
    node = _defs("_save_pretrained_merged", ST_PY)[1]
    body = ast.get_source_segment(src, node)
    assert 'kwargs.setdefault("save_method", save_method)' in body


@pytest.mark.parametrize("i", [0, 1])
def test_an_explicit_tokenizer_is_not_shadowed_by_kwargs(i):
    """Passing it positionally AND as a keyword must not raise or silently
    prefer the wrong one."""
    src = ST_PY.read_text(encoding="utf-8")
    body = ast.get_source_segment(src, _defs("_save_pretrained_merged", ST_PY)[i])
    assert "if tokenizer is None:" in body
    assert 'pop("tokenizer", None)' in body


# ---- behavioural check with a stand-in ------------------------------------


def _extract_helper(name):
    """Pull a module-level helper out by source, so nothing imports unsloth."""
    src = ST_PY.read_text(encoding="utf-8")
    node = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert node, f"{name} not found in sentence_transformer.py"
    ns = {}
    exec(ast.get_source_segment(src, node[0]), ns)
    return ns[name]


_normalize_save_method = _extract_helper("_normalize_save_method")


_PROBE_PACKAGE = "_unsloth_st_source_probe"


def _probe_package():
    """A package for the exec'd body to resolve its deferred relative imports against.

    #11067 put `from ..save import _is_adapter_save_method` inside `_save_pretrained_merged`,
    deferred on purpose: a module-scope bind out of `unsloth.save` closes an import cycle.
    `exec` with a bare globals dict has no `__name__` and no `__package__`, so a relative
    import there raises `KeyError: "'__name__' not in globals"`, and every test that calls the
    body died on it rather than on anything it was checking.

    Satisfying it by importing unsloth would give up what this file is for: it reads the two
    closures out of the source precisely so the assertions run without an unsloth import. So
    stand up a package that owns the same relative path and put the REAL helper in it, pulled
    from unsloth/save.py by source the way `_normalize_save_method` already is. The import
    statement then executes for real and binds the shipped function, so a change to that
    function's behaviour still reaches these tests.

    Registered under a name of our own rather than "unsloth", so this cannot shadow the real
    package for anything else in the session.
    """
    import sys
    import types

    if _PROBE_PACKAGE in sys.modules:
        return

    root = types.ModuleType(_PROBE_PACKAGE)
    root.__path__ = []
    models = types.ModuleType(f"{_PROBE_PACKAGE}.models")
    models.__path__ = []
    save = types.ModuleType(f"{_PROBE_PACKAGE}.save")
    # By source, not by import, for the same reason as everything else in this file.
    save_src = SAVE_PY.read_text(encoding="utf-8")
    helper = [
        n
        for n in ast.parse(save_src).body
        if isinstance(n, ast.FunctionDef) and n.name == "_is_adapter_save_method"
    ]
    assert helper, "_is_adapter_save_method is gone from save.py; the deferred import cannot bind"
    exec(ast.get_source_segment(save_src, helper[0]), save.__dict__)

    sys.modules[_PROBE_PACKAGE] = root
    sys.modules[f"{_PROBE_PACKAGE}.models"] = models
    sys.modules[f"{_PROBE_PACKAGE}.save"] = save


def _extract(i):
    """Exec one closure body standalone so it can actually be called."""
    src = ST_PY.read_text(encoding="utf-8")
    node = _defs("_save_pretrained_merged", ST_PY)[i]
    import textwrap

    _probe_package()
    branding = []
    ns = {
        # The package coordinates of unsloth/models/sentence_transformer.py, mirrored onto the
        # probe package: `from ..save import ...` in the body resolves one level up from
        # `<probe>.models`, which is where the extracted helper was just installed.
        "__name__": f"{_PROBE_PACKAGE}.models.sentence_transformer",
        "__package__": f"{_PROBE_PACKAGE}.models",
        "os": __import__("os"),
        "print": lambda *a, **k: None,
        "_normalize_save_method": _normalize_save_method,
        "FastSentenceTransformer": type(
            "F", (), {"_add_unsloth_branding": staticmethod(lambda d: branding.append(d))}
        ),
    }
    exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
    return ns["_save_pretrained_merged"], branding


def test_the_deferred_import_binds_the_shipped_helper():
    """The harness above must resolve that import for real, not paper over it.

    A stub returning a constant would make the "lora" tests pass while testing nothing, which
    is the failure mode worth guarding: the point of routing through the real
    `_is_adapter_save_method` is that its spelling rules ("LoRA", "lora ") are the ones the
    shipped code applies.
    """
    _probe_package()
    import sys

    helper = sys.modules[f"{_PROBE_PACKAGE}.save"]._is_adapter_save_method
    for accepted in ("lora", "LoRA", "lora ", " LORA"):
        assert helper(accepted), f"the shipped helper no longer accepts {accepted!r}"
    for refused in ("merged_16bit", "lora_adapter", None, 1):
        assert not helper(refused), f"the shipped helper now accepts {refused!r}"
    # And it is the shipped source, not a copy that can drift from it.
    save_src = SAVE_PY.read_text(encoding="utf-8")
    assert "def _is_adapter_save_method(save_method):" in save_src


class _Tok:
    def __init__(self):
        self.saved = []

    def save_pretrained(self, d):
        self.saved.append(d)


class _Inner:
    """The wrapped transformer, whose LoRA gets merged away."""

    def __init__(self):
        self.merged = False
        self.saved = []
        self.forwarded = None

    def merge_and_unload(self):
        self.merged = True
        return self

    def save_pretrained(self, d, **kwargs):
        self.saved.append(d)

    def save_pretrained_merged(self, d, **kwargs):
        self.forwarded = kwargs


class _Module:
    def __init__(self, inner):
        self.auto_model = inner


class _FakeST:
    """A SentenceTransformer stand-in: a sequence of modules, module 0 of
    which wraps the transformer."""

    def __init__(self, no_modules=False):
        self.tokenizer = _Tok()
        self.saved = []
        self.inner = _Inner()
        self.no_modules = no_modules
        self._modules_list = [_Module(self.inner)]

    def __getitem__(self, i):
        return self._modules_list[i]

    def save_pretrained(self, d):
        self.saved.append(d)


def test_positional_call_completes_the_merge(tmp_path):
    """The documented positional call. Before the fix this raised TypeError
    before running a single statement."""
    fn, branding = _extract(0)
    st = _FakeST()
    tok = _Tok()
    fn(st, str(tmp_path), tok, "merged_16bit")
    assert st.saved == [str(tmp_path)]
    assert st.inner.merged, "LoRA should have been merged away"
    assert st.inner.saved == [str(tmp_path)]
    assert tok.saved == [str(tmp_path)], "the passed tokenizer must be used"
    assert branding == [str(tmp_path)]


def test_the_positional_tokenizer_wins_over_the_models_own(tmp_path):
    fn, _ = _extract(0)
    st = _FakeST()
    tok = _Tok()
    fn(st, str(tmp_path), tok)
    assert tok.saved == [str(tmp_path)]
    assert st.tokenizer.saved == [], "self.tokenizer must not be used instead"


def test_omitting_the_tokenizer_falls_back_to_the_models_own(tmp_path):
    fn, _ = _extract(0)
    st = _FakeST()
    fn(st, str(tmp_path))
    assert st.tokenizer.saved == [str(tmp_path)]


def test_keyword_tokenizer_still_works(tmp_path):
    """The call style that worked before must keep working."""
    fn, _ = _extract(0)
    st = _FakeST()
    tok = _Tok()
    fn(st, str(tmp_path), tokenizer=tok)
    assert tok.saved == [str(tmp_path)]


def test_unsupported_save_method_raises_not_implemented(tmp_path):
    fn, _ = _extract(0)
    st = _FakeST()
    with pytest.raises(NotImplementedError, match="merged_16bit"):
        fn(st, str(tmp_path), None, "lora")
    assert not st.inner.merged, "must refuse BEFORE merging anything away"
    assert st.saved == [], "and before writing a half-finished directory"


# ---- the no_modules fallback honours save_method too -----------------------


def test_no_modules_fallback_refuses_an_unsupported_method(tmp_path):
    """That branch drops save_method and merges unconditionally, so "lora"
    would return full weights for a request to write adapters."""
    fn, _ = _extract(1)
    st = _FakeST(no_modules=True)
    with pytest.raises(NotImplementedError, match="merged_16bit"):
        fn(st, str(tmp_path), None, "lora")
    assert not st.inner.merged, "must refuse BEFORE merging the adapters away"
    assert st.saved == [], "and before writing a half-finished directory"


def test_no_modules_fallback_still_does_a_16bit_merge(tmp_path):
    fn, _ = _extract(1)
    st = _FakeST(no_modules=True)
    fn(st, str(tmp_path), None, "merged_16bit")
    assert st.inner.merged


def test_the_forwarding_path_refuses_lora_for_its_own_reason(tmp_path):
    """Both branches refuse "lora", and the two refusals must stay distinct.

    This test used to assert that the forwarding path ACCEPTED "lora", on the premise that
    with modules.json present the merge understands every method. unsloth#11067 retired that
    premise: `save_pretrained_merged` writes a loadable SentenceTransformer, and an
    adapter-only save has no base weights for one, so it now refuses here too. Before that,
    "lora" matched no branch in merge_and_overwrite_lora and fell through to a 16bit merge,
    which happened to write something loadable for a request that asked for adapters.

    The invariant the old test was really protecting survives the change and is what is
    checked now: the no_modules refusal must not leak into this branch. Both raise, so a
    bare `pytest.raises` would no longer notice if one wrongly borrowed the other's reason,
    which is why the two messages are pinned apart rather than merely asserted non-empty.
    """
    fn, _ = _extract(1)
    st = _FakeST(no_modules=False)
    with pytest.raises(NotImplementedError) as raised:
        fn(st, str(tmp_path), None, "lora")

    message = str(raised.value)
    # Its own reason: no base weights for a loadable SentenceTransformer, plus the two ways
    # out. Not the no_modules reason, which is about falling back to merge_and_unload.
    assert "adapter" in message
    assert "merged_16bit" in message
    assert "no modules.json was found" not in message, (
        "the no_modules refusal leaked into the branch that HAS modules.json; the two "
        "conditions are different and a user told the wrong one looks for the wrong file"
    )
    # Refused before anything is touched, exactly as the no_modules branch is.
    assert not st.inner.merged, "must refuse BEFORE merging the adapters away"
    assert st.saved == [], "and before writing a half-finished directory"
    assert st.inner.forwarded is None, "and without forwarding the unsupported method on"


def test_the_two_lora_refusals_do_not_share_a_message(tmp_path):
    """Written because the check above can only see one of the pair at a time.

    If both branches were ever collapsed onto one message, each test would still pass on its
    own substring while the distinction that makes the advice actionable was gone.
    """
    fn, _ = _extract(1)
    messages = []
    for no_modules in (True, False):
        with pytest.raises(NotImplementedError) as raised:
            fn(_FakeST(no_modules=no_modules), str(tmp_path), None, "lora")
        messages.append(str(raised.value))
    assert messages[0] != messages[1], (
        "both 'lora' refusals now say the same thing, so the message no longer tells the "
        "user whether modules.json was the problem"
    )


# ---- spellings mean the same thing as in unsloth_save_model ----------------


def test_the_reference_normalizes_before_validating():
    """Guards the premise: save.py lowercases and strips spaces first."""
    body = ast.get_source_segment(
        SAVE_PY.read_text(encoding="utf-8"), _defs("unsloth_save_model", SAVE_PY)[0]
    )
    assert 'save_method.lower().replace(" ", "_")' in body


@pytest.mark.parametrize("method", ["MERGED_16BIT", "merged 16bit", "Merged_16bit"])
@pytest.mark.parametrize("i", [0, 1])
def test_case_and_space_variants_are_accepted(method, i, tmp_path):
    """These keyword calls worked before the signature change."""
    fn, _ = _extract(i)
    st = _FakeST(no_modules=True)
    fn(st, str(tmp_path), None, method)
    assert st.inner.merged


def test_normalization_leaves_non_strings_alone():
    assert _normalize_save_method(None) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
