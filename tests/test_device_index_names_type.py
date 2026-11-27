# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""torch.device(<int index>) means CUDA, so an XPU / NPU card index must name its device type."""

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "unsloth"


def test_device_index_names_its_type():
    bare = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "torch.device"
                and len(node.args) == 1
                and not node.keywords
            ):
                continue
            arg = node.args[0]
            name = arg.id if isinstance(arg, ast.Name) else None
            if (
                isinstance(arg, ast.Constant)
                and type(arg.value) is int
                or (name and ("idx" in name or "index" in name))
            ):
                bare.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    assert not bare, f"torch.device(index) without a device type: {bare}"
