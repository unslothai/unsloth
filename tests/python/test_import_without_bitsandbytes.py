"""`import unsloth` must survive a missing bitsandbytes.

device_type.py already tells the user "bitsandbytes is not installed - 4bit QLoRA
unallowed, but 16bit and full finetuning works", and the gfx906 install path
(#7354) deliberately removes the generic wheel because it carries no gfx906
kernels. Any module-level `import bitsandbytes` on the import chain turns that
into an unimportable package instead.

peft's 4bit LoRA layer is exported only when bnb is importable, so
`from peft.tuners.lora import Linear4bit` fails on the same hosts and is checked
here too.
"""

# Path | None below is a PEP 604 union;
# the project still supports Python 3.9.
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_MODULE = "unsloth"


def _module_path(name: str) -> Path | None:
    base = REPO_ROOT / Path(*name.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _bnb_dependent(node: ast.stmt) -> bool:
    """True for an import that raises when bitsandbytes is absent."""
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[0] == "bitsandbytes" for a in node.names)
    if isinstance(node, ast.ImportFrom) and node.level == 0:
        module = node.module or ""
        if module.split(".")[0] == "bitsandbytes":
            return True
        # peft re-exports Linear4bit only when bnb imported cleanly.
        if module.startswith("peft.tuners.lora"):
            return any(a.name == "Linear4bit" for a in node.names)
    return False


def _allow_bitsandbytes_gated(test: ast.expr) -> bool:
    """device_type.py sets ALLOW_BITSANDBYTES=False exactly when the import failed,
    so a branch keyed on it cannot run without bnb."""
    return any(isinstance(n, ast.Name) and n.id == "ALLOW_BITSANDBYTES" for n in ast.walk(test))


def _scan(path: Path, module: str):
    """Yield (lineno, source) for unguarded top-level imports.

    Imports inside a `try`, or under an ALLOW_BITSANDBYTES branch, are guarded.
    Other `if` bodies are not: the condition may well be true on a host without bnb.
    """
    is_package = path.name == "__init__.py"
    package = module if is_package else module.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    risky, edges = [], []

    def walk(body, guarded):
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if not guarded and _bnb_dependent(node):
                    risky.append((node.lineno, ast.unparse(node)))
                if isinstance(node, ast.Import):
                    edges.extend(a.name for a in node.names)
                elif node.level:
                    parts = package.split(".")
                    base = ".".join(parts[: len(parts) - (node.level - 1)])
                    edges.append(f"{base}.{node.module}" if node.module else base)
                else:
                    edges.append(node.module or "")
            elif isinstance(node, ast.Try):
                walk(node.body, True)
                for handler in node.handlers:
                    walk(handler.body, True)
                walk(node.orelse, True)
                walk(node.finalbody, guarded)
            elif isinstance(node, ast.If):
                walk(node.body, guarded or _allow_bitsandbytes_gated(node.test))
                walk(node.orelse, guarded)

    walk(tree.body, False)
    return risky, edges


def test_no_unguarded_bitsandbytes_import_on_the_unsloth_import_chain():
    seen, pending, offenders = set(), [(ROOT_MODULE, [])], []
    while pending:
        module, chain = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        path = _module_path(module)
        if path is None:
            continue
        risky, edges = _scan(path, module)
        for lineno, source in risky:
            rel = path.relative_to(REPO_ROOT).as_posix()
            offenders.append(f"{rel}:{lineno}  {source}\n    via {' -> '.join(chain + [module])}")
        pending.extend(
            (edge, chain + [module]) for edge in edges if edge.split(".")[0] == ROOT_MODULE
        )

    assert len(seen) > 20, f"import chain walk collapsed, only reached {seen}"
    assert not offenders, (
        "`import unsloth` must not hard-require bitsandbytes. Wrap these in "
        "try/except and fall back to a placeholder:\n  " + "\n  ".join(offenders)
    )


def test_missing_bnb_leaves_a_callable_that_reports_the_real_cause():
    """The 4bit ctypes handles degrade to a stub, not a NameError later on."""
    src = (REPO_ROOT / "unsloth" / "kernels" / "utils.py").read_text(encoding = "utf-8")
    assert "def _bnb_required(" in src
    assert "get_ptr = _bnb_required" in src
    for name in (
        "cdequantize_blockwise_fp32",
        "cdequantize_blockwise_fp16_nf4",
        "cdequantize_blockwise_bf16_nf4",
        "cgemm_4bit_inference_naive_fp16",
        "cgemm_4bit_inference_naive_bf16",
    ):
        assert f"{name} = _bnb_required" in src, f"{name} has no bnb-less fallback"


def test_capability_flags_come_from_a_guarded_import_not_find_spec():
    """kernels/utils.py and _gpu_init.py treat any import failure as unavailable.
    device_type.py must agree, or an installed-but-unusable wheel leaves
    ALLOW_BITSANDBYTES true while the kernels fall back to the stub."""
    src = (REPO_ROOT / "unsloth" / "device_type.py").read_text(encoding = "utf-8")
    head = src.split('if DEVICE_TYPE == "hip":')[0]
    assert "import bitsandbytes as _bnb_probe" in head
    assert 'find_spec("bitsandbytes")' not in head, "find_spec cannot see a broken wheel"
    assert head.count("ALLOW_BITSANDBYTES = False") >= 1


def _bnb_guards():
    src = (REPO_ROOT / "unsloth" / "models" / "loader.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    return src, [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(n, ast.Name) and n.id == "ALLOW_BITSANDBYTES" for n in ast.walk(node.test)
        )
    ]


def test_bitsandbytes_guard_is_not_gated_on_use_exact_model_name():
    """use_exact_model_name suppresses repo-name remapping; it cannot make bnb
    available. Gating on it left the default load_in_4bit=True set on a host
    without bitsandbytes."""
    _, guards = _bnb_guards()
    assert len(guards) == 2, f"expected both loader guards, found {len(guards)}"
    for guard in guards:
        names = {n.id for n in ast.walk(guard.test) if isinstance(n, ast.Name)}
        assert (
            "use_exact_model_name" not in names
        ), f"guard at line {guard.lineno} still gates the capability check on naming"


def test_bitsandbytes_guard_drops_a_bnb_quantization_config():
    """A BitsAndBytesConfig in kwargs re-sets the flags downstream, so clearing
    load_in_4bit/8bit alone still builds the bnb quantizer in Transformers. A
    non-bnb config (GPTQ/AWQ/fp8) must not be touched."""
    _, guards = _bnb_guards()
    for guard in guards:
        # ast.unparse normalises quotes, so match on the call shape instead.
        def _is_pop(node):
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pop"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "kwargs"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "quantization_config"
            )

        assert any(
            _is_pop(n) for n in ast.walk(guard)
        ), f"guard at line {guard.lineno} leaves the bnb config in kwargs"
        # the pop must be conditional on the config actually asking for bnb
        pops = [
            node
            for node in ast.walk(guard)
            if isinstance(node, ast.If) and any(_is_pop(n) for n in ast.walk(node))
        ]
        assert pops, f"guard at line {guard.lineno} pops unconditionally"
        assert any(
            isinstance(n, ast.Name) and n.id == "_wants_bnb"
            for node in pops
            for n in ast.walk(node.test)
        ), f"guard at line {guard.lineno} does not gate the pop on a bnb request"


def test_bitsandbytes_guard_clears_8bit_as_well_as_4bit():
    """8bit is bitsandbytes too: leaving load_in_8bit set sends the request to
    Transformers, which builds the bnb quantizer and fails there instead."""
    src = (REPO_ROOT / "unsloth" / "models" / "loader.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(n, ast.Name) and n.id == "ALLOW_BITSANDBYTES" for n in ast.walk(node.test)
        )
    ]
    assert len(guards) == 2, f"expected both loader guards, found {len(guards)}"
    for guard in guards:
        cleared = {
            target.id
            for stmt in guard.body
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
            if isinstance(target, ast.Name)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is False
        }
        assert {
            "load_in_4bit",
            "load_in_8bit",
        } <= cleared, f"guard at line {guard.lineno} clears only {sorted(cleared)}"


def test_capability_fallback_precedes_the_mutually_exclusive_mode_check():
    """load_in_4bit defaults to True, so load_in_16bit=True trips the
    "can only load in 4bit or 8bit or 16bit" RuntimeError unless the unavailable
    4bit request is cleared first. That check must come after the fallback."""
    src, _ = _bnb_guards()
    tree = ast.parse(src)
    checked = 0
    # Scope to the enclosing function:
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        raises = [
            node.lineno
            for node in ast.walk(func)
            if isinstance(node, ast.Raise)
            and "Can only load in 4bit or 8bit or 16bit" in ast.unparse(node)
        ]
        if not raises:
            continue
        guards = [
            node.lineno
            for node in ast.walk(func)
            if isinstance(node, ast.If)
            and any(
                isinstance(n, ast.Name) and n.id == "ALLOW_BITSANDBYTES"
                for n in ast.walk(node.test)
            )
        ]
        for lineno in raises:
            checked += 1
            assert any(g < lineno for g in guards), (
                f"{func.name}: the mode check at line {lineno} runs before this "
                "function's ALLOW_BITSANDBYTES fallback, so load_in_16bit=True on a "
                "bnb-less host raises instead of taking the 16bit path"
            )
    assert checked, "mode-exclusivity check not found"


def test_bitsandbytes_compile_patch_is_never_called_unguarded():
    """unsloth_zoo's patch_compiling_bitsandbytes imports bitsandbytes
    unconditionally, so an unwrapped call raises on a bnb-less host before any
    fallback can run."""
    src = (REPO_ROOT / "unsloth" / "models" / "loader.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "patch_compiling_bitsandbytes"
    ]
    assert calls, "call sites not found"
    guarded = {
        call.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "patch_compiling_bitsandbytes"
    }
    unguarded = sorted({c.lineno for c in calls} - guarded)
    assert not unguarded, f"patch_compiling_bitsandbytes called unguarded at {unguarded}"
