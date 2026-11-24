"""typer 0.29 imports pydantic. `unsloth studio setup` runs as a typer CLI, so with typer 0.29 it holds
pydantic_core's native module open while the Studio requirements re-pin pydantic, and Windows refuses to
replace a loaded .pyd ("Access is denied"). Every install that runs before setup keeps typer below 0.29.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
FIRST_PYDANTIC_TYPER = "0.29.0"


def _typer(requirements):
    found = [Requirement(r) for r in requirements if re.match(r"\s*typer\b", r)]
    assert len(found) == 1, requirements
    return found[0]


def test_the_package_dependency_keeps_typer_below_the_pydantic_release():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding = "utf-8"))["project"]
    req = _typer(project["dependencies"])
    assert FIRST_PYDANTIC_TYPER not in req.specifier, req


def test_the_no_torch_runtime_keeps_typer_below_the_pydantic_release():
    path = ROOT / "studio" / "backend" / "requirements" / "no-torch-runtime.txt"
    lines = [l.split("#", 1)[0].strip() for l in path.read_text(encoding = "utf-8").splitlines()]
    req = _typer([l for l in lines if l])
    assert FIRST_PYDANTIC_TYPER not in req.specifier, req


def test_every_resolving_base_install_in_install_ps1_carries_the_cap():
    source = (ROOT / "install.ps1").read_text(encoding = "utf-8")
    cap = re.search(r'^\s*\$_typerCapSpec\s*=\s*"([^"]+)"', source, re.M)
    assert cap, "install.ps1 no longer defines $_typerCapSpec"
    assert FIRST_PYDANTIC_TYPER not in Requirement(cap.group(1)).specifier, cap.group(1)
    installs = re.findall(r'-Label "install unsloth[^"]*" \{([^\n]*)\}', source)
    resolving = [cmd for cmd in installs if "--no-deps" not in cmd]
    assert len(resolving) >= 4, installs
    missing = [cmd.strip() for cmd in resolving if '"$_typerCapSpec"' not in cmd]
    assert not missing, missing
