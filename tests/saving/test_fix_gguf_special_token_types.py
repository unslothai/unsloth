import json
import os
import struct

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from unsloth.tokenizer_utils import fix_gguf_special_token_types


NORMAL, CONTROL, USER_DEFINED = 1, 3, 4

# GGUF metadata value/array element type ids.
_T_INT32 = 5
_T_STRING = 8
_T_ARRAY = 9


def _u64(n):
    return struct.pack("<Q", n)


def _gguf_string(s):
    b = s.encode("utf-8")
    return _u64(len(b)) + b


def _kv_string(key, value):
    return _gguf_string(key) + struct.pack("<I", _T_STRING) + _gguf_string(value)


def _kv_string_array(key, values):
    out = _gguf_string(key) + struct.pack("<I", _T_ARRAY)
    out += struct.pack("<I", _T_STRING) + _u64(len(values))
    for v in values:
        out += _gguf_string(v)
    return out


def _kv_int32_array(key, values):
    out = _gguf_string(key) + struct.pack("<I", _T_ARRAY)
    out += struct.pack("<I", _T_INT32) + _u64(len(values))
    out += b"".join(struct.pack("<i", v) for v in values)
    return out


def _write_gguf(path, tokens, token_types, trailing_merges=None):
    """Minimal GGUF v3 with tokens then token_type (mirrors llama.cpp ordering).

    A merges array is written *after* token_type so the test also exercises the
    early-break: the function must locate token_type without walking merges.
    """
    kvs = [
        _kv_string("general.architecture", "gemma4"),
        _kv_string_array("tokenizer.ggml.tokens", tokens),
        _kv_int32_array("tokenizer.ggml.token_type", token_types),
        _kv_string_array("tokenizer.ggml.merges", trailing_merges or ["a b", "c d"]),
    ]
    body = b"GGUF" + struct.pack("<I", 3) + _u64(0) + _u64(len(kvs)) + b"".join(kvs)
    with open(path, "wb") as f:
        f.write(body)


def _read_token_types(path):
    buf = open(path, "rb").read()
    pos = [4]

    def rd(fmt):
        v = struct.unpack_from(fmt, buf, pos[0])
        pos[0] += struct.calcsize(fmt)
        return v[0]

    rd("<I")  # version
    rd("<Q")  # tensor count
    n_kv = rd("<Q")
    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    for _ in range(n_kv):
        klen = rd("<Q")
        key = buf[pos[0]:pos[0] + klen].decode()
        pos[0] += klen
        vt = rd("<I")
        if vt == _T_STRING:
            slen = rd("<Q")
            pos[0] += slen
        elif vt == _T_ARRAY:
            et = rd("<I")
            n = rd("<Q")
            if et == _T_STRING:
                for _ in range(n):
                    slen = rd("<Q")
                    pos[0] += slen
            elif key == "tokenizer.ggml.token_type":
                return list(struct.unpack_from("<%di" % n, buf, pos[0]))
            else:
                pos[0] += struct.calcsize(scalar[et]) * n
        else:
            pos[0] += struct.calcsize(scalar[vt])
    raise AssertionError("token_type not found")


def _write_tokenizer_json(path, added):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"added_tokens": added}, f)


# Token layout shared by several tests: id 0 normal text, id 1 already-control
# delimiter, ids 2-3 special delimiters mistyped USER_DEFINED, id 4 a stray
# control-looking vocab token that is NOT in added_tokens.
def _gemma4_like(tmp_path):
    gguf = tmp_path / "model.gguf"
    tokens = ["hello", "<turn|>", "<|tool_response>", "<|channel>", "</s>"]
    types = [NORMAL, CONTROL, USER_DEFINED, USER_DEFINED, NORMAL]
    _write_gguf(str(gguf), tokens, types)
    _write_tokenizer_json(
        str(tmp_path / "tokenizer.json"),
        [
            {"id": 1, "content": "<turn|>", "special": True},
            {"id": 2, "content": "<|tool_response>", "special": True},
            {"id": 3, "content": "<|channel>", "special": True},
        ],
    )
    return gguf


def test_user_defined_special_delimiters_become_control(tmp_path):
    gguf = _gemma4_like(tmp_path)
    patched = fix_gguf_special_token_types(str(gguf))
    assert patched == 2
    types = _read_token_types(str(gguf))
    assert types[2] == CONTROL  # <|tool_response>
    assert types[3] == CONTROL  # <|channel>


def test_already_control_and_plain_tokens_untouched(tmp_path):
    gguf = _gemma4_like(tmp_path)
    fix_gguf_special_token_types(str(gguf))
    types = _read_token_types(str(gguf))
    assert types[0] == NORMAL    # "hello" plain text
    assert types[1] == CONTROL   # already correct
    assert types[4] == NORMAL    # "</s>" not in added_tokens -> left for llama.cpp


def test_idempotent_second_run_patches_nothing(tmp_path):
    gguf = _gemma4_like(tmp_path)
    assert fix_gguf_special_token_types(str(gguf)) == 2
    assert fix_gguf_special_token_types(str(gguf)) == 0


def test_non_control_looking_special_is_not_retyped(tmp_path):
    gguf = tmp_path / "model.gguf"
    _write_gguf(str(gguf), ["plainword"], [USER_DEFINED])
    _write_tokenizer_json(
        str(tmp_path / "tokenizer.json"),
        [{"id": 0, "content": "plainword", "special": True}],
    )
    assert fix_gguf_special_token_types(str(gguf)) == 0
    assert _read_token_types(str(gguf))[0] == USER_DEFINED


def test_id_content_mismatch_is_skipped(tmp_path):
    gguf = tmp_path / "model.gguf"
    _write_gguf(str(gguf), ["<|tool_response>"], [USER_DEFINED])
    # tokenizer.json claims a different content at id 0 -> ids drifted, skip.
    _write_tokenizer_json(
        str(tmp_path / "tokenizer.json"),
        [{"id": 0, "content": "<|something_else>", "special": True}],
    )
    assert fix_gguf_special_token_types(str(gguf)) == 0
    assert _read_token_types(str(gguf))[0] == USER_DEFINED


def test_missing_tokenizer_json_is_noop(tmp_path):
    gguf = tmp_path / "model.gguf"
    _write_gguf(str(gguf), ["<|tool_response>"], [USER_DEFINED])
    assert fix_gguf_special_token_types(str(gguf)) == 0


def test_missing_gguf_is_noop(tmp_path):
    assert fix_gguf_special_token_types(str(tmp_path / "nope.gguf")) == 0


def test_explicit_tokenizer_json_path_is_used(tmp_path):
    gguf = tmp_path / "out" / "model.gguf"
    gguf.parent.mkdir()
    _write_gguf(str(gguf), ["<|tool_call>"], [USER_DEFINED])
    tok = tmp_path / "src" / "tokenizer.json"
    tok.parent.mkdir()
    _write_tokenizer_json(str(tok), [{"id": 0, "content": "<|tool_call>", "special": True}])
    assert fix_gguf_special_token_types(str(gguf), tokenizer_json=str(tok)) == 1
    assert _read_token_types(str(gguf))[0] == CONTROL
