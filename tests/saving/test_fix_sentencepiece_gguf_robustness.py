import ast
import json
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from transformers.utils import sentencepiece_model_pb2

from unsloth.tokenizer_utils import fix_sentencepiece_gguf


NORMAL, CONTROL, USER_DEFINED = 1, 3, 4

_SAVE_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "unsloth", "save.py")
)
_TOK_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "unsloth", "tokenizer_utils.py")
)


def _build(pieces):
    m = sentencepiece_model_pb2.ModelProto()
    for piece, score, typ in pieces:
        p = m.pieces.add()
        p.piece = piece
        p.score = score
        p.type = typ
    return m.SerializeToString()


def _read(path):
    m = sentencepiece_model_pb2.ModelProto()
    with open(path, "rb") as f:
        m.ParseFromString(f.read())
    return [(p.piece, p.type) for p in m.pieces]


def test_user_defined_special_piece_is_not_retyped(tmp_path):
    pieces = [
        ("<s>", 0.0, CONTROL),
        ("a", -1.0, NORMAL),
        ("<ud_special>", -1.0, USER_DEFINED),
    ]
    (tmp_path / "tokenizer.model").write_bytes(_build(pieces))
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {"added_tokens": [{"id": 2, "content": "<ud_special>", "special": True}]}
        )
    )
    fix_sentencepiece_gguf(str(tmp_path))
    got = dict(_read(str(tmp_path / "tokenizer.model")))
    assert got["<ud_special>"] == USER_DEFINED


def test_malformed_entry_missing_id_does_not_raise(tmp_path):
    pieces = [("<s>", 0.0, CONTROL), ("a", -1.0, NORMAL), ("<sot>", -1.0, NORMAL)]
    (tmp_path / "tokenizer.model").write_bytes(_build(pieces))
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"content": "no_id_entry", "special": True},
                    {"id": 2, "content": "<sot>", "special": True},
                ]
            }
        )
    )
    fix_sentencepiece_gguf(str(tmp_path))
    got = dict(_read(str(tmp_path / "tokenizer.model")))
    assert got["<sot>"] == CONTROL


def test_entry_with_non_int_id_is_skipped(tmp_path):
    pieces = [("<s>", 0.0, CONTROL), ("a", -1.0, NORMAL)]
    (tmp_path / "tokenizer.model").write_bytes(_build(pieces))
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"added_tokens": [{"id": "oops", "content": "x", "special": True}]})
    )
    before = (tmp_path / "tokenizer.model").read_bytes()
    fix_sentencepiece_gguf(str(tmp_path))
    after = (tmp_path / "tokenizer.model").read_bytes()
    assert before == after


def test_save_py_except_clause_is_broad_exception():
    with open(_SAVE_PY) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "unsloth_save_pretrained_gguf"
        ):
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Try):
                    body_src = "\n".join(ast.unparse(s) for s in subnode.body)
                    if "fix_sentencepiece_gguf(" not in body_src:
                        continue
                    handler = subnode.handlers[0]
                    assert handler.type is not None
                    assert isinstance(handler.type, ast.Name)
                    assert handler.type.id == "Exception"
                    return
    raise AssertionError(
        "fix_sentencepiece_gguf try block not found in unsloth_save_pretrained_gguf"
    )


def test_tokenizer_utils_uses_import_protobuf_fallback_pattern():
    with open(_TOK_PY) as f:
        src = f.read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "fix_sentencepiece_gguf":
            fn_src = ast.unparse(node)
            assert "import_protobuf" in fn_src
            return
    raise AssertionError("fix_sentencepiece_gguf not found in tokenizer_utils.py")


def test_all_special_tokens_are_gated_by_tokenizer_json_not_by_type(tmp_path):
    pieces = [
        ("<s>", 0.0, CONTROL),
        ("a", -1.0, NORMAL),
        ("<n1>", -1.0, NORMAL),
        ("<u1>", -1.0, USER_DEFINED),
    ]
    (tmp_path / "tokenizer.model").write_bytes(_build(pieces))
    fix_sentencepiece_gguf(str(tmp_path))
    got = dict(_read(str(tmp_path / "tokenizer.model")))
    assert got["<n1>"] == NORMAL
    assert got["<u1>"] == USER_DEFINED
