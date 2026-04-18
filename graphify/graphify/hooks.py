# git hook integration - install/uninstall graphify post-commit and post-checkout hooks
from __future__ import annotations
from pathlib import Path

_HOOK_MARKER = "# graphify-hook"
_CHECKOUT_MARKER = "# graphify-checkout-hook"

_HOOK_SCRIPT = """\
#!/bin/bash
# graphify-hook
# Auto-rebuilds the knowledge graph after each commit (code files only, no LLM needed).
# Installed by: graphify hook install

CHANGED=$(git diff --name-only HEAD~1 HEAD 2>/dev/null || git diff --name-only HEAD 2>/dev/null)
if [ -z "$CHANGED" ]; then
    exit 0
fi

export GRAPHIFY_CHANGED="$CHANGED"
python3 -c "
import os, sys
from pathlib import Path

CODE_EXTS = {
    '.py', '.ts', '.js', '.go', '.rs', '.java', '.cpp', '.c', '.rb', '.swift',
    '.kt', '.cs', '.scala', '.php', '.cc', '.cxx', '.hpp', '.h', '.kts',
}

changed_raw = os.environ.get('GRAPHIFY_CHANGED', '')
changed = [Path(f.strip()) for f in changed_raw.strip().splitlines() if f.strip()]
code_changed = [f for f in changed if f.suffix.lower() in CODE_EXTS and f.exists()]

if not code_changed:
    sys.exit(0)

print(f'[graphify hook] {len(code_changed)} code file(s) changed - rebuilding graph...')

try:
    from graphify.watch import _rebuild_code
    _rebuild_code(Path('.'))
except Exception as exc:
    print(f'[graphify hook] Rebuild failed: {exc}')
    sys.exit(0)
"
"""


_CHECKOUT_SCRIPT = """\
#!/bin/bash
# graphify-checkout-hook
# Auto-rebuilds the knowledge graph (code only) when switching branches.
# Installed by: graphify hook install

PREV_HEAD=$1
NEW_HEAD=$2
BRANCH_SWITCH=$3

# Only run on branch switches, not file checkouts
if [ "$BRANCH_SWITCH" != "1" ]; then
    exit 0
fi

# Only run if graphify-out/ exists (graph has been built before)
if [ ! -d "graphify-out" ]; then
    exit 0
fi

echo "[graphify] Branch switched - rebuilding knowledge graph (code files)..."
python3 -c "
from graphify.watch import _rebuild_code
from pathlib import Path
try:
    _rebuild_code(Path('.'))
except Exception as exc:
    print(f'[graphify] Rebuild failed: {exc}')
"
"""


def _git_root(path: Path) -> Path | None:
    """Walk up to find .git directory."""
    current = path.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _install_hook(hooks_dir: Path, name: str, script: str, marker: str) -> str:
    """Install a single git hook, appending if an existing hook is present."""
    hook_path = hooks_dir / name
    if hook_path.exists():
        content = hook_path.read_text()
        if marker in content:
            return f"already installed at {hook_path}"
        hook_path.write_text(content.rstrip() + "\n\n" + script)
        return f"appended to existing {name} hook at {hook_path}"
    hook_path.write_text(script)
    hook_path.chmod(0o755)
    return f"installed at {hook_path}"


def _uninstall_hook(hooks_dir: Path, name: str, marker: str) -> str:
    """Remove graphify section from a git hook."""
    hook_path = hooks_dir / name
    if not hook_path.exists():
        return f"no {name} hook found - nothing to remove."
    content = hook_path.read_text()
    if marker not in content:
        return f"graphify hook not found in {name} - nothing to remove."
    before = content.split(marker)[0].rstrip()
    non_empty = [l for l in before.splitlines() if l.strip() and not l.startswith("#!")]
    if not non_empty:
        hook_path.unlink()
        return f"removed {name} hook at {hook_path}"
    hook_path.write_text(before + "\n")
    return f"graphify removed from {name} at {hook_path} (other hook content preserved)"


def install(path: Path = Path(".")) -> str:
    """Install graphify post-commit and post-checkout hooks in the nearest git repo."""
    root = _git_root(path)
    if root is None:
        raise RuntimeError(f"No git repository found at or above {path.resolve()}")

    hooks_dir = root / ".git" / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    commit_msg = _install_hook(hooks_dir, "post-commit", _HOOK_SCRIPT, _HOOK_MARKER)
    checkout_msg = _install_hook(hooks_dir, "post-checkout", _CHECKOUT_SCRIPT, _CHECKOUT_MARKER)

    return f"post-commit: {commit_msg}\npost-checkout: {checkout_msg}"


def uninstall(path: Path = Path(".")) -> str:
    """Remove graphify post-commit and post-checkout hooks."""
    root = _git_root(path)
    if root is None:
        raise RuntimeError(f"No git repository found at or above {path.resolve()}")

    hooks_dir = root / ".git" / "hooks"
    commit_msg = _uninstall_hook(hooks_dir, "post-commit", _HOOK_MARKER)
    checkout_msg = _uninstall_hook(hooks_dir, "post-checkout", _CHECKOUT_MARKER)

    return f"post-commit: {commit_msg}\npost-checkout: {checkout_msg}"


def status(path: Path = Path(".")) -> str:
    """Check if graphify hooks are installed."""
    root = _git_root(path)
    if root is None:
        return "Not in a git repository."
    hooks_dir = root / ".git" / "hooks"

    def _check(name: str, marker: str) -> str:
        p = hooks_dir / name
        if not p.exists():
            return "not installed"
        return "installed" if marker in p.read_text() else "not installed (hook exists but graphify not found)"

    commit = _check("post-commit", _HOOK_MARKER)
    checkout = _check("post-checkout", _CHECKOUT_MARKER)
    return f"post-commit: {commit}\npost-checkout: {checkout}"
