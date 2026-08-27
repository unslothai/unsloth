"""A value imported from the @/features/chat barrel must not be consumed at
module scope.

features/chat is inside an import cycle -- chat-runtime-store -> presets/
preset-load-config -> features/model-picker -> ... -> features/chat -- so a
module importing from the barrel can be evaluated while chat-runtime-store is
still initializing. Naming one of its `const` exports at module scope then reads
it in its temporal dead zone and throws at import time, which takes the whole
page down rather than failing anything locally:

    [ansi-smoke] pageerror: Cannot access 'CHAT_GPU_MEMORY_MODE_KEY'
                            before initialization

That shipped from hooks/use-model-memory.ts, whose WATCHED_STORAGE_KEYS array
listed the key at module scope. Reading inside a function is safe: by call time
every module has finished loading.

Breaking the cycle would be the deeper fix, but it spans three features, and
this guard holds while that is true or false -- the rule it enforces (do not
consume a possibly-uninitialized binding eagerly) is worth keeping either way.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "studio/frontend/src"

_BARREL_IMPORT = re.compile(r"""import\s*\{(?P<names>[^}]*)\}\s*from\s*["']@/features/chat["']""")
# Top-level `const X = ...;` / `let X = ...;` only: a leading space would make it
# a nested declaration, and those run when their enclosing function is called.
_MODULE_SCOPE_BINDING = re.compile(r"^(?:const|let)\s+(\w+)\s*=\s*(.*?);", re.M | re.S)


def _imported_values(src: str) -> list[str]:
    """Value names this module pulls out of the chat barrel.

    `type` specifiers are erased before the code runs, so they cannot trip a
    temporal dead zone and are not the guard's business.
    """
    match = _BARREL_IMPORT.search(src)
    if match is None:
        return []
    names: list[str] = []
    for raw in match.group("names").split(","):
        name = raw.strip()
        if not name or name.startswith("type "):
            continue
        names.append(name.split(" as ")[0].strip())
    return names


def _is_deferred(body: str) -> bool:
    """True when the initialiser only *captures* the names rather than reading
    them, so evaluation is deferred to call time."""
    head = body.lstrip()
    if head.startswith(("function", "async", "(")):
        return True
    # `() => ...`, `key => ...`: an arrow anywhere in the first line means the
    # names below it are inside the body, not evaluated to build it.
    return "=>" in head.split("\n", 1)[0]


def test_no_module_scope_read_of_a_chat_barrel_value():
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.ts*")):
        if path.suffix not in (".ts", ".tsx"):
            continue
        src = path.read_text(encoding = "utf-8", errors = "replace")
        names = _imported_values(src)
        if not names:
            continue
        for binding in _MODULE_SCOPE_BINDING.finditer(src):
            body = binding.group(2)
            if _is_deferred(body):
                continue
            for name in names:
                if re.search(rf"\b{re.escape(name)}\b", body):
                    rel = path.relative_to(REPO)
                    offenders.append(f"{rel}: `{binding.group(1)}` reads {name}")

    assert not offenders, (
        "these module-scope bindings consume a value imported from the "
        "@/features/chat barrel, which can still be in its temporal dead zone "
        "when the import cycle re-enters:\n  "
        + "\n  ".join(sorted(set(offenders)))
        + "\nWrap the initialiser in a function so the read happens on call, as "
        "hooks/use-model-memory.ts does with watchedStorageKeys()."
    )


def test_the_scan_still_sees_the_barrel_importers_it_claims_to():
    """Anti-vacuity: a rename of the barrel path would silently pass the test
    above by finding nothing at all to check."""
    importers = [
        path
        for path in SRC.rglob("*.ts*")
        if path.suffix in (".ts", ".tsx")
        and _imported_values(path.read_text(encoding = "utf-8", errors = "replace"))
    ]
    assert len(importers) >= 5, (
        f"only {len(importers)} modules import values from @/features/chat; the "
        "barrel path probably moved and this guard is no longer scanning anything"
    )
