"""CPO shares ORPO's row-tokenization replacements (issue #4952).

CPOTrainer reuses ORPOTrainer's build_tokenized_answer/tokenize_row/__init__
code, so the same ORPO rewriters that route tokenization through a processor's
inner tokenizer and resolve pad_token_id must be registered for cpo_trainer too.
Without them the positional self.processing_class(prompt, ...) call binds prompt
to a multimodal processor's images= arg and crashes on text[0]. These are
static, CPU-only checks so they run on Linux/macOS/Windows without torch.
"""

import ast
import os
import re


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RL_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _registrations(source):
    """Map each RL_FUNCTIONS[key] target to the appended function names."""
    tree = ast.parse(source)
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            continue
        if call.func.attr != "append":
            continue
        sub = call.func.value
        if not (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name)):
            continue
        if sub.value.id != "RL_FUNCTIONS":
            continue
        key = sub.slice
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        arg = call.args[0]
        if isinstance(arg, ast.Name):
            out.setdefault(key.value, []).append(arg.id)
    return out


def _load_rewriter(name):
    src = open(RL_PATH).read()
    tree = ast.parse(src)
    ns = {"re": re}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("_"):
                    exec(ast.get_source_segment(src, node), ns)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(ast.get_source_segment(src, node), ns)
            return ns[name]
    raise AssertionError(f"{name} not found")


def test_cpo_trainer_registers_orpo_text_tokenizer_and_pad_token():
    regs = _registrations(open(RL_PATH).read())
    cpo = regs.get("cpo_trainer", [])
    assert "orpo_trainer_text_tokenizer" in cpo
    assert "orpo_trainer_processor_pad_token" in cpo


def test_cpo_registration_matches_orpo():
    regs = _registrations(open(RL_PATH).read())
    shared = {"orpo_trainer_text_tokenizer", "orpo_trainer_processor_pad_token"}
    assert shared <= set(regs.get("orpo_trainer", []))
    assert shared <= set(regs.get("cpo_trainer", []))


def test_rewriter_drops_positional_processing_class_call():
    rewriter = _load_rewriter("orpo_trainer_text_tokenizer")
    source = """
def build_tokenized_answer(self, prompt, answer):
    full_tokenized = self.processing_class(prompt + answer, add_special_tokens=False)
    prompt_input_ids = self.processing_class(prompt, add_special_tokens=False)["input_ids"]
    return full_tokenized, prompt_input_ids
"""
    rewritten = rewriter("build_tokenized_answer", source)
    assert "self.processing_class(prompt" not in rewritten
    assert (
        'tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)'
        in rewritten
    )
