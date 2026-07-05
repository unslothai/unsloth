"""Regression test: FastLlamaModel.patch_peft_model must dispatch Qwen3-MoE.

The activation dispatch in ``patch_peft_model`` reads ``model.config.model_type``
and branches on string literals. The canonical HuggingFace ``model_type`` for
Qwen3-MoE is ``"qwen3_moe"`` (with an underscore), used everywhere else in the
repo (unsloth/models/_utils.py, the transformers.models.qwen3_moe imports,
loader.py's comment). A missing underscore ("qwen3moe") makes that branch dead,
so a Qwen3-MoE adapter falls through to
``else: raise NotImplementedError("Unsloth: qwen3_moe is not yet implemented!")``
and blocks ``get_peft_model`` on Qwen3-MoE.

Pure AST/exec so it runs on every CI OS/Python without torch: it extracts the
real dispatch chain from unsloth/models/llama.py, execs it with the
``apply_lora_mlp_*`` names stubbed, and asserts ``"qwen3_moe"`` no longer raises.
"""

from __future__ import annotations

import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LLAMA_PATH = REPO_ROOT / "unsloth" / "models" / "llama.py"


def _extract_activation_dispatch() -> ast.If:
    """Return the ``if model_type == ...`` chain from FastLlamaModel.patch_peft_model."""
    tree = ast.parse(LLAMA_PATH.read_text(encoding = "utf-8"))

    method = None
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if cls.name != "FastLlamaModel":
            continue
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == "patch_peft_model":
                method = item
                break
    assert method is not None, "FastLlamaModel.patch_peft_model not found"

    # The dispatch is the first ``if`` whose test compares ``model_type`` to a str
    # and whose orelse ultimately raises NotImplementedError.
    for node in method.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "model_type"
        ):
            return node
    raise AssertionError("activation dispatch (if model_type == ...) not found")


def _run_dispatch(model_type: str) -> str:
    """Exec the real dispatch chain for ``model_type``; return the bound apply fn name.

    ``apply_lora_mlp_*`` are stubbed with sentinel strings so no torch/kernels are
    imported; the ``else`` still raises NotImplementedError as in the real code.
    """
    dispatch = _extract_activation_dispatch()
    module = ast.Module(body = [dispatch], type_ignores = [])
    ast.fix_missing_locations(module)
    code = compile(module, filename = str(LLAMA_PATH), mode = "exec")

    namespace = {
        "model_type": model_type,
        "apply_lora_mlp_swiglu": "apply_lora_mlp_swiglu",
        "apply_lora_mlp_geglu_approx": "apply_lora_mlp_geglu_approx",
    }
    exec(code, namespace)
    return namespace["apply_lora_mlp"]


class TestPatchPeftModelDispatchesQwen3Moe(unittest.TestCase):
    def test_qwen3_moe_does_not_raise_not_implemented(self):
        """The real HF model_type "qwen3_moe" is dispatched, not rejected."""
        self.assertEqual(_run_dispatch("qwen3_moe"), "apply_lora_mlp_swiglu")

    def test_misspelled_qwen3moe_branch_is_gone(self):
        """The dead, underscore-less "qwen3moe" literal must not be in the dispatch."""
        dispatch = _extract_activation_dispatch()
        literals = {
            n.value
            for n in ast.walk(dispatch)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        self.assertIn("qwen3_moe", literals)
        self.assertNotIn("qwen3moe", literals)

    def test_unknown_model_type_still_raises(self):
        """Guard: an unrelated model_type still hits the NotImplementedError else-branch."""
        with self.assertRaises(NotImplementedError):
            _run_dispatch("definitely_not_a_real_model_type")


if __name__ == "__main__":
    unittest.main()
