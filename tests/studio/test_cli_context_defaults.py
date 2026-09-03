"""`unsloth chat` / `unsloth inference` must default --max-seq-length to 0 like `unsloth studio run`.

A non-zero default never matches a resident llama-server's n_ctx, so attaching reloads the warm model.
"""

import ast
from pathlib import Path

_CMD_DIR = Path(__file__).resolve().parents[2] / "unsloth_cli" / "commands"


def _typer_option_default(source: str, func_name: str, long_option: str):
    tree = ast.parse(source)
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) or func.name != func_name:
            continue
        defaults = func.args.defaults + [d for d in func.args.kw_defaults if d is not None]
        for default in defaults:
            if not isinstance(default, ast.Call):
                continue
            call = default.func
            if not (
                isinstance(call, ast.Attribute)
                and call.attr == "Option"
                and isinstance(call.value, ast.Name)
                and call.value.id == "typer"
            ):
                continue
            if not default.args:
                continue
            flags = [
                a.value
                for a in default.args[1:]
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            ]
            if long_option in flags and isinstance(default.args[0], ast.Constant):
                return default.args[0].value
    return None


def test_chat_and_inference_context_default_matches_studio_run():
    run_default = _typer_option_default(
        (_CMD_DIR / "studio.py").read_text(encoding = "utf-8"), "run", "--max-seq-length"
    )
    assert run_default == 0, f"`unsloth studio run` --max-seq-length default changed to {run_default}"

    for module, func in (("chat.py", "chat"), ("inference.py", "inference")):
        default = _typer_option_default(
            (_CMD_DIR / module).read_text(encoding = "utf-8"), func, "--max-seq-length"
        )
        assert default is not None, f"no --max-seq-length typer.Option found in {module}:{func}()"
        assert default == run_default, (
            f"{module}:{func}() --max-seq-length defaults to {default}; a non-zero default "
            "reloads a model an Unsloth server already holds warm"
        )
