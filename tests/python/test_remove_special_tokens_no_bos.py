import ast
from pathlib import Path


def _load_remove_special_tokens():
    # Extract remove_special_tokens without importing unsloth (importing unsloth needs unsloth_zoo / a GPU).
    source = Path(__file__).parents[2] / "unsloth" / "chat_templates.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    funcs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "remove_special_tokens"
    ]
    namespace = {}
    module = ast.Module(body = funcs, type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["remove_special_tokens"]


class _StubTokenizer:
    def __init__(self, bos_token):
        self.bos_token = bos_token


def test_no_bos_tokenizer_does_not_crash():
    remove_special_tokens = _load_remove_special_tokens()
    assert remove_special_tokens(_StubTokenizer(None), "Hello world") == "Hello world"


def test_double_bos_is_stripped():
    # Tokenizers such as Qwen2 / Qwen2.5, GPT-2, Falcon and GPT-NeoX have no BOS token, so tokenizer.bos_token is None.
    remove_special_tokens = _load_remove_special_tokens()
    assert remove_special_tokens(_StubTokenizer("<s>"), "<s>Hello world") == "Hello world"


def test_prompt_without_leading_bos_unchanged():
    remove_special_tokens = _load_remove_special_tokens()
    assert remove_special_tokens(_StubTokenizer("<s>"), "Hello world") == "Hello world"
