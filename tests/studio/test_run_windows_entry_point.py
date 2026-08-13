"""AST tests for how `unsloth run` reaches the studio venv's CLI.

Two separate regressions are pinned here, both needing a real studio venv to reach
at runtime, hence the AST:

1. The venv's entry point is `unsloth.exe` on Windows, so resolving the bare name
   made `studio_bin.is_file()` false on every Windows install and aborted with
   "Unsloth venv missing 'unsloth' entry point". The per-platform name still has to
   be chosen, because on POSIX that file is what proves the venv has a CLI at all.

2. That file must not be what Windows LAUNCHES. It is a generated, unsigned
   executable, and an Application Control policy denies it while the signed
   python.exe beside it still runs, so the respawn goes through the interpreter
   (issue #8490). POSIX keeps exec'ing the script, which is what os.execvp needs.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STUDIO = _REPO_ROOT / "unsloth_cli" / "commands" / "studio.py"


def _run_function() -> ast.FunctionDef:
    tree = ast.parse(_STUDIO.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return node
    raise AssertionError("no top-level `run` command in unsloth_cli/commands/studio.py")


def _studio_bin_value() -> ast.expr:
    for node in ast.walk(_run_function()):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "studio_bin"
                and node.value is not None
            ):
                if not (isinstance(node.value, ast.Constant) and node.value.value is None):
                    return node.value
    raise AssertionError("`run` never assigns a studio_bin path")


def test_the_entry_point_name_is_chosen_per_platform():
    value = _studio_bin_value()
    assert isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div), (
        "expected studio_bin to be built as `studio_python.parent / <name>`, got "
        f"{ast.dump(value)}"
    )
    names = {
        node.value
        for node in ast.walk(value.right)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {
        "unsloth",
        "unsloth.exe",
    } <= names, f"studio_bin must pick between 'unsloth' and 'unsloth.exe'; got {sorted(names)}"


def test_the_windows_branch_is_the_exe():
    """A swapped conditional would still hold the two names but break both platforms."""
    branch = next(
        node for node in ast.walk(_studio_bin_value().right) if isinstance(node, ast.IfExp)
    )
    assert isinstance(branch.body, ast.Constant) and branch.body.value == "unsloth.exe"
    assert isinstance(branch.orelse, ast.Constant) and branch.orelse.value == "unsloth"
    assert "Windows" in {
        node.value
        for node in ast.walk(branch.test)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }, "the .exe branch must be gated on platform.system() == 'Windows'"


def _launch_head_value() -> ast.expr:
    for node in ast.walk(_run_function()):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "launch_head":
                return node.value
    raise AssertionError("`run` never assigns a launch_head")


def test_windows_respawns_through_the_interpreter_not_the_console_script():
    """The blocked executable must not be argv[0] of the child on Windows."""
    branch = _launch_head_value()
    assert isinstance(
        branch, ast.IfExp
    ), f"expected launch_head to branch per platform, got {ast.dump(branch)}"
    # Windows arm: _managed_cli_argv(studio_python), i.e. the interpreter form.
    assert isinstance(branch.body, ast.Call), ast.dump(branch.body)
    assert isinstance(branch.body.func, ast.Name)
    assert branch.body.func.id == "_managed_cli_argv", (
        "the Windows arm must build the interpreter argv via _managed_cli_argv, got "
        f"{ast.dump(branch.body.func)}"
    )
    assert [arg.id for arg in branch.body.args if isinstance(arg, ast.Name)] == [
        "studio_python"
    ], "the interpreter argv must be built from studio_python"
    # POSIX arm: [str(studio_bin)] -- unchanged, and what os.execvp needs.
    assert isinstance(branch.orelse, ast.List) and len(branch.orelse.elts) == 1
    posix_head = branch.orelse.elts[0]
    assert isinstance(posix_head, ast.Call) and isinstance(posix_head.func, ast.Name)
    assert posix_head.func.id == "str"
    assert isinstance(posix_head.args[0], ast.Name) and posix_head.args[0].id == "studio_bin"
    assert "win32" in {
        node.value
        for node in ast.walk(branch.test)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }, "the interpreter arm must be gated on sys.platform == 'win32'"


def test_the_trampoline_is_the_one_the_rust_and_powershell_sides_use():
    """One string, three languages. A drift here silently changes argv[0] handling.

    Each side is read from its own file. An earlier version of this test only
    grepped studio.py, so drifting the Rust and PowerShell copies left it green.
    """
    # Spelled out, not imported from any of the three, so editing any single copy
    # fails here instead of quietly agreeing with itself.
    canonical = (
        "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; "
        "sys.argv[0] = 'unsloth'; from unsloth_cli import app; sys.exit(app())"
    )

    # Python: via AST, because the constant is written as adjacent literals.
    python_value = None
    for node in ast.walk(ast.parse(_STUDIO.read_text(encoding = "utf-8"))):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_WINDOWS_CLI_ENTRYPOINT" for t in node.targets
        ):
            python_value = ast.literal_eval(node.value)
    assert (
        python_value == canonical
    ), f"_WINDOWS_CLI_ENTRYPOINT in {_STUDIO.name} has drifted: {python_value!r}"

    rust = (_REPO_ROOT / "studio" / "src-tauri" / "src" / "process.rs").read_text(encoding = "utf-8")
    assert (
        f'"{canonical}"' in rust
    ), "WINDOWS_CLI_ENTRYPOINT in studio/src-tauri/src/process.rs has drifted"

    powershell = (_REPO_ROOT / "install.ps1").read_text(encoding = "utf-8")
    assert (
        f'$script:UnslothCliTrampoline = "{canonical}"' in powershell
    ), "$script:UnslothCliTrampoline in install.ps1 has drifted"


def test_the_interpreter_argv_carries_no_isolation_flag():
    """-I would imply -E and drop every PYTHON* variable the console script honours.

    The trampoline's own sys.path[:1] filter is what keeps a stray unsloth_cli in
    the working directory from shadowing the managed package, so -I is not needed
    for that either.
    """
    source = _STUDIO.read_text(encoding = "utf-8")
    assert (
        '"-X", "utf8", "-c"' in source
    ), "the interpreter argv must be `-X utf8 -c <trampoline>`, in that order"
    assert '"-I"' not in source, "-I breaks PYTHON* parity with the console script"


def test_the_windows_existence_gate_accepts_a_quarantined_venv():
    """Quarantine deletes the stub; the install behind it still runs.

    The Windows respawn goes through the interpreter and never touches this file,
    so requiring it here would abort `studio run` on an environment that works,
    which is the whole failure this change exists to remove.
    """
    gate = None
    for node in ast.walk(_run_function()):
        if not isinstance(node, ast.If):
            continue
        called = {
            child.func.attr
            for child in ast.walk(node.test)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        }
        if "is_file" in called and any(
            isinstance(child, ast.Name) and child.id == "studio_bin"
            for child in ast.walk(node.test)
        ):
            gate = node
            break
    assert gate is not None, "`run` no longer gates on studio_bin.is_file()"
    fallbacks = {
        child.func.id
        for child in ast.walk(gate.test)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }
    assert "_managed_cli_package_present" in fallbacks, (
        "a missing console script must fall back to the installed package, or a "
        "quarantined Windows install cannot start Studio"
    )
