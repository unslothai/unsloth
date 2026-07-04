"""FastBaseModel.from_pretrained's vLLM-standby guard message must name the
`unsloth_vllm_standby` parameter, not the `UNSLOTH_VLLM_STANDBY` env var twice.

The guard is `if unsloth_vllm_standby and os.environ.get("UNSLOTH_VLLM_STANDBY") ...`,
so the thing that is True is the PARAMETER while the env var is what is unset. The
message previously named the env var in both clauses ("UNSLOTH_VLLM_STANDBY is True,
but UNSLOTH_VLLM_STANDBY is not set to 1"), which is self-contradictory. The sibling
guard in FastLlamaModel.from_pretrained already names both correctly.
"""

import ast
import os
import unittest


def _standby_message(module_relpath):
    """Return the string raised by the `unsloth_vllm_standby` guard in a module."""
    path = os.path.join(
        os.path.dirname(__file__), os.pardir, "unsloth", "models", module_relpath
    )
    with open(path, encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.dump(node.test)
        if "unsloth_vllm_standby" not in test_src or "UNSLOTH_VLLM_STANDBY" not in test_src:
            continue
        for raise_node in ast.walk(node):
            if (
                isinstance(raise_node, ast.Raise)
                and isinstance(raise_node.exc, ast.Call)
                and raise_node.exc.args
                and isinstance(raise_node.exc.args[0], ast.Constant)
                and isinstance(raise_node.exc.args[0].value, str)
            ):
                return raise_node.exc.args[0].value
    return None


class TestStandbyErrorMessage(unittest.TestCase):
    def test_vision_message_names_parameter_not_env_var_twice(self):
        msg = _standby_message("vision.py")
        self.assertIsNotNone(msg, "could not find the unsloth_vllm_standby guard in vision.py")
        self.assertIn(
            "`unsloth_vllm_standby`",
            msg,
            f"message must name the parameter (the thing that is True), got: {msg!r}",
        )
        # The self-contradictory wording named the env var in the True clause too.
        self.assertNotIn(
            "UNSLOTH_VLLM_STANDBY is True",
            msg,
            f"the env var must not be described as the True value, got: {msg!r}",
        )

    def test_vision_matches_llama_sibling_intent(self):
        # Both from_pretrained guards should describe the same thing the same way:
        # the parameter is True, the env var is unset.
        for relpath in ("vision.py", "llama.py"):
            msg = _standby_message(relpath)
            self.assertIsNotNone(msg, f"could not find the unsloth_vllm_standby guard in {relpath}")
            self.assertIn("`unsloth_vllm_standby`", msg, f"{relpath}: {msg!r}")
            self.assertIn("`UNSLOTH_VLLM_STANDBY`", msg, f"{relpath}: {msg!r}")


if __name__ == "__main__":
    unittest.main()
