# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Minimal OLE compound file writer (version 3) for building .doc/.xls/.ppt/.msg fixtures."""

from __future__ import annotations

import math
import struct

_SECTOR, _MINI, _CUTOFF = 512, 64, 4096
_FREE, _END, _FATSECT = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD


def _entry(name: str, kind: int, child: int, right: int, start: int, size: int) -> bytes:
    encoded = (name + "\0").encode("utf-16-le")
    return (
        encoded.ljust(64, b"\0")
        + struct.pack("<HBB", len(encoded), kind, 1)
        + struct.pack("<III", _FREE, right, child)
        + b"\0" * 36
        + struct.pack("<IQ", start, size)
    )


def compound_file(streams: dict[tuple[str, ...], bytes]) -> bytes:
    """Streams keyed by path; storages are implied. Streams under 4096 bytes use the mini stream."""
    # Directory tree: each storage's children chained through right-sibling links.
    nodes = [{"name": "Root Entry", "kind": 5, "children": []}]
    index = {(): 0}
    for path in streams:
        for depth in range(1, len(path) + 1):
            key = path[:depth]
            if key not in index:
                index[key] = len(nodes)
                kind = 2 if depth == len(path) else 1
                nodes.append({"name": key[-1], "kind": kind, "children": [], "path": key})
                nodes[index[key[:-1]]]["children"].append(index[key])

    mini = b""
    big: list[tuple[int, bytes]] = []
    placement = {}
    for path, data in streams.items():
        i = index[path]
        if len(data) < _CUTOFF:
            placement[i] = ("mini", len(mini) // _MINI, len(data))
            mini += (
                data.ljust(math.ceil(len(data) / _MINI) * _MINI or _MINI, b"\0") if data else b""
            )
        else:
            big.append((i, data))

    def sectors(n_bytes):
        return math.ceil(n_bytes / _SECTOR)

    mini_fat = []
    for i, (where, start, size) in sorted(placement.items(), key = lambda kv: kv[1][1]):
        count = math.ceil(size / _MINI)
        mini_fat += [start + k + 1 for k in range(count - 1)] + ([_END] if count else [])
    mini_fat_bytes = struct.pack(f"<{len(mini_fat)}I", *mini_fat) if mini_fat else b""

    layout = []  # (label, n_sectors)
    n_dir = sectors(len(nodes) * 128)
    layout.append(("dir", n_dir))
    layout.append(("minifat", sectors(len(mini_fat_bytes))))
    layout.append(("ministream", sectors(len(mini))))
    for i, data in big:
        layout.append((i, sectors(len(data))))
    data_sectors = sum(n for _, n in layout)
    n_fat = 1
    while n_fat * 128 < data_sectors + n_fat:
        n_fat += 1

    start_of, cursor = {}, n_fat
    for label, n in layout:
        start_of[label] = cursor if n else _END
        cursor += n
    fat = [_FATSECT] * n_fat
    for label, n in layout:
        fat += [start_of[label] + k + 1 for k in range(n - 1)] + ([_END] if n else [])
    fat += [_FREE] * (n_fat * 128 - len(fat))

    entries = []
    for i, node in enumerate(nodes):
        kids = node["children"]
        child = kids[0] if kids else _FREE
        parent = next((n for n in nodes if i in n["children"]), None)
        right = _FREE
        if parent is not None:
            pos = parent["children"].index(i)
            right = parent["children"][pos + 1] if pos + 1 < len(parent["children"]) else _FREE
        if i == 0:
            start, size = start_of["ministream"], len(mini)
        elif node["kind"] == 2:
            if i in placement:
                start, size = placement[i][1], placement[i][2]
            else:
                start, size = start_of[i], len(streams[node["path"]])
        else:
            start, size = 0, 0
        entries.append(_entry(node["name"], node["kind"], child, right, start, size))
    directory = b"".join(entries)

    header = bytearray(512)
    header[:8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    struct.pack_into("<HHHHH", header, 0x18, 0x3E, 3, 0xFFFE, 9, 6)
    struct.pack_into("<I", header, 0x2C, n_fat)
    struct.pack_into("<I", header, 0x30, start_of["dir"])
    struct.pack_into("<I", header, 0x38, _CUTOFF)
    struct.pack_into("<II", header, 0x3C, start_of["minifat"], sectors(len(mini_fat_bytes)))
    struct.pack_into("<II", header, 0x44, _END, 0)
    difat = list(range(n_fat)) + [_FREE] * (109 - n_fat)
    struct.pack_into("<109I", header, 0x4C, *difat)

    body = struct.pack(f"<{len(fat)}I", *fat)
    for label, n in layout:
        if label == "dir":
            chunk = directory
        elif label == "minifat":
            chunk = mini_fat_bytes
        elif label == "ministream":
            chunk = mini
        else:
            chunk = streams[nodes[label]["path"]]
        body += chunk.ljust(n * _SECTOR, b"\0")
    return bytes(header) + body
