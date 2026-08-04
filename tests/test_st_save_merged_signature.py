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
    src = path.read_text(encoding = "utf-8")
    found = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.FunctionDef) and n.name == name]
    return sorted(found, key = lambda n: n.lineno)


def _params(node):
    return [a.arg for a in node.args.args]


def test_both_definitions_exist():
    assert len(_defs("_save_pretrained_merged", ST_PY)) == 2, (
        "two closures bind this name; both must be fixed or one model path "
        "keeps the old signature")


@pytest.mark.parametrize("i", [0, 1])
def test_signature_matches_fastlanguagemodel(i):
    node = _defs("_save_pretrained_merged", ST_PY)[i]
    assert _params(node)[:4] == ["self", "save_directory", "tokenizer",
                                 "save_method"]


@pytest.mark.parametrize("i", [0, 1])
def test_still_accepts_arbitrary_keywords(i):
    node = _defs("_save_pretrained_merged", ST_PY)[i]
    assert node.args.kwarg is not None, "**kwargs must survive"


@pytest.mark.parametrize("i", [0, 1])
def test_defaults_keep_the_old_keyword_call_working(i):
    node = _defs("_save_pretrained_merged", ST_PY)[i]
    defaults = {a.arg: d for a, d in
                zip(node.args.args[-len(node.args.defaults):],
                    node.args.defaults)}
    assert isinstance(defaults["tokenizer"], ast.Constant)
    assert defaults["tokenizer"].value is None
    assert defaults["save_method"].value == "merged_16bit"


def test_the_reference_signature_is_what_we_matched():
    """Guards the premise: if FastLanguageModel's signature moves, these
    two have to move with it, and the test should say so."""
    node = _defs("unsloth_save_pretrained_merged", SAVE_PY)
    assert node, "unsloth_save_pretrained_merged not found"
    assert _params(node[0])[:4] == ["self", "save_directory", "tokenizer",
                                    "save_method"]


# ---- save_method is honoured, not swallowed -------------------------------

def test_merge_and_unload_path_refuses_an_unsupported_method():
    src = ST_PY.read_text(encoding = "utf-8")
    node = _defs("_save_pretrained_merged", ST_PY)[0]
    body = ast.get_source_segment(src, node)
    assert "NotImplementedError" in body
    assert "merged_16bit" in body


def test_forwarding_path_passes_save_method_through():
    src = ST_PY.read_text(encoding = "utf-8")
    node = _defs("_save_pretrained_merged", ST_PY)[1]
    body = ast.get_source_segment(src, node)
    assert 'kwargs.setdefault("save_method", save_method)' in body


@pytest.mark.parametrize("i", [0, 1])
def test_an_explicit_tokenizer_is_not_shadowed_by_kwargs(i):
    """Passing it positionally AND as a keyword must not raise or silently
    prefer the wrong one."""
    src = ST_PY.read_text(encoding = "utf-8")
    body = ast.get_source_segment(src, _defs("_save_pretrained_merged", ST_PY)[i])
    assert "if tokenizer is None:" in body
    assert 'pop("tokenizer", None)' in body


# ---- behavioural check with a stand-in ------------------------------------

def _extract(i):
    """Exec one closure body standalone so it can actually be called."""
    src = ST_PY.read_text(encoding = "utf-8")
    node = _defs("_save_pretrained_merged", ST_PY)[i]
    import textwrap
    branding = []
    ns = {
        "os": __import__("os"),
        "print": lambda *a, **k: None,
        "FastSentenceTransformer": type(
            "F", (), {"_add_unsloth_branding":
                      staticmethod(lambda d: branding.append(d))}),
    }
    exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
    return ns["_save_pretrained_merged"], branding


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

    def merge_and_unload(self):
        self.merged = True
        return self

    def save_pretrained(self, d):
        self.saved.append(d)


class _Module:
    def __init__(self, inner):
        self.auto_model = inner


class _FakeST:
    """A SentenceTransformer stand-in: a sequence of modules, module 0 of
    which wraps the transformer."""
    def __init__(self):
        self.tokenizer = _Tok()
        self.saved = []
        self.inner = _Inner()
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
    fn(st, str(tmp_path), tokenizer = tok)
    assert tok.saved == [str(tmp_path)]


def test_unsupported_save_method_raises_not_implemented(tmp_path):
    fn, _ = _extract(0)
    st = _FakeST()
    with pytest.raises(NotImplementedError, match = "merged_16bit"):
        fn(st, str(tmp_path), None, "lora")
    assert not st.inner.merged, "must refuse BEFORE merging anything away"
    assert st.saved == [], "and before writing a half-finished directory"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
