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
