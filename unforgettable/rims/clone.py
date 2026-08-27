# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure filesystem clone of a world sandbox into a sim sandbox."""

from __future__ import annotations

import shutil
from pathlib import Path

IGNORE_NAMES = frozenset({".unsloth_sandbox", ".unsloth_sandbox_remap.json"})


def _ignore(directory: str, names: list[str]) -> set[str]:
    skipped = set()
    for name in names:
        if name in IGNORE_NAMES or ".deleting-" in name:
            skipped.add(name)
    return skipped


def clone_tree(src: str | Path, dst: str | Path) -> Path:
    source = Path(src).resolve()
    dest = Path(dst)
    if dest.exists() or dest.is_symlink():
        dest = dest.resolve()
    else:
        dest = dest.expanduser()
        if dest.parent.exists() or dest.parent.is_symlink():
            dest = dest.parent.resolve() / dest.name
        else:
            dest = dest.resolve()
    if source == dest:
        raise ValueError("clone_tree refuses to copy a tree onto itself")
    try:
        dest.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("clone_tree refuses to copy a tree into itself")
    if not source.is_dir():
        raise FileNotFoundError(f"world sandbox missing: {source}")
    dest.mkdir(parents = True, exist_ok = True)
    # Copy symlink nodes; do not dereference a world link into ~/.ssh or /etc.
    shutil.copytree(source, dest, dirs_exist_ok = True, ignore = _ignore, symlinks = True)
    return dest
