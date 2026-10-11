"""Hub remote code imports these, nothing in this repo does, so an unused-package sweep drops them."""

from __future__ import annotations

import re
from pathlib import Path

EXTRAS_TXT = (
    Path(__file__).resolve().parents[2] / "studio" / "backend" / "requirements" / "extras.txt"
)

REMOTE_CODE_PACKAGES = {
    "addict": ("unsloth/DeepSeek-OCR", "unsloth/DeepSeek-OCR-2"),  # #8732
}


def _requirement_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding = "utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line)
        if match:
            names.add(re.sub(r"[-_.]+", "-", match.group(0)).lower())
    return names


def test_remote_code_packages_are_installed():
    names = _requirement_names(EXTRAS_TXT)
    missing = {pkg: models for pkg, models in REMOTE_CODE_PACKAGES.items() if pkg not in names}
    assert not missing, f"extras.txt dropped packages that model remote code imports: {missing}"
