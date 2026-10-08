# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only OLE compound file reader ([MS-CFB]), the container of .doc, .xls, .ppt and .msg."""

from __future__ import annotations

import struct

SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_FREE, _END, _FAT, _DIFAT = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFC
_NO_STREAM = 0xFFFFFFFF
_STORAGE, _STREAM, _ROOT = 1, 2, 5


class CompoundFileError(ValueError):
    pass


class _Entry:
    __slots__ = ("name", "kind", "left", "right", "child", "start", "size")

    def __init__(self, raw: bytes, major: int):
        name_len = struct.unpack_from("<H", raw, 64)[0]
        self.name = raw[: max(0, min(name_len, 64) - 2)].decode("utf-16-le", "replace")
        self.kind = raw[66]
        self.left, self.right, self.child = struct.unpack_from("<III", raw, 68)
        self.start = struct.unpack_from("<I", raw, 116)[0]
        size = struct.unpack_from("<Q", raw, 120)[0]
        # Version 3 files may leave junk in the high half.
        self.size = size & 0xFFFFFFFF if major == 3 else size


class CompoundFile:
    """Streams of a compound file, addressed by path: ``open("WordDocument")``,
    ``open("__attach_version1.0_#00000000", "__substg1.0_3707001F")``."""

    def __init__(self, data: bytes):
        if len(data) < 512 or not data.startswith(SIGNATURE):
            raise CompoundFileError("not an OLE compound file")
        self._data = data
        major = struct.unpack_from("<H", data, 0x1A)[0]
        shift = struct.unpack_from("<H", data, 0x1E)[0]
        mini_shift = struct.unpack_from("<H", data, 0x20)[0]
        if (major, shift) not in ((3, 9), (4, 12)) or mini_shift != 6:
            raise CompoundFileError("unsupported compound file version")
        self._sector = 1 << shift
        self._mini_cutoff = struct.unpack_from("<I", data, 0x38)[0]
        (n_fat,) = struct.unpack_from("<I", data, 0x2C)
        (first_dir,) = struct.unpack_from("<I", data, 0x30)
        first_mini_fat, n_mini_fat = struct.unpack_from("<II", data, 0x3C)
        first_difat, n_difat = struct.unpack_from("<II", data, 0x44)

        fat_sectors = [s for s in struct.unpack_from("<109I", data, 0x4C) if s < _DIFAT]
        per = self._sector // 4
        sector, seen = first_difat, set()
        for _ in range(n_difat):
            if sector >= _DIFAT or sector in seen:
                break
            seen.add(sector)
            words = struct.unpack(f"<{per}I", self._raw_sector(sector))
            fat_sectors += [s for s in words[:-1] if s < _DIFAT]
            sector = words[-1]
        fat_sectors = fat_sectors[:n_fat] if n_fat else fat_sectors
        self._fat: list[int] = []
        for s in fat_sectors:
            self._fat.extend(struct.unpack(f"<{per}I", self._raw_sector(s)))

        dir_bytes = self._read_chain(first_dir, self._fat, self._raw_sector, self._sector)
        self._entries = [
            _Entry(dir_bytes[i : i + 128], major) for i in range(0, len(dir_bytes) - 127, 128)
        ]
        if not self._entries or self._entries[0].kind != _ROOT:
            raise CompoundFileError("missing root entry")
        root = self._entries[0]
        mini_fat_bytes = (
            self._read_chain(first_mini_fat, self._fat, self._raw_sector, self._sector)
            if n_mini_fat
            else b""
        )
        self._mini_fat = list(struct.unpack(f"<{len(mini_fat_bytes) // 4}I", mini_fat_bytes))
        self._mini_stream = self._read_chain(
            root.start, self._fat, self._raw_sector, self._sector, root.size
        )

    def _raw_sector(self, sector: int) -> bytes:
        offset = (sector + 1) * self._sector
        if offset + self._sector > len(self._data):
            # A short final sector is common; anything past the end is not.
            if offset >= len(self._data):
                raise CompoundFileError("sector out of range")
            return self._data[offset:].ljust(self._sector, b"\0")
        return self._data[offset : offset + self._sector]

    def _mini_sector(self, sector: int) -> bytes:
        offset = sector * 64
        if offset >= len(self._mini_stream):
            raise CompoundFileError("mini sector out of range")
        return self._mini_stream[offset : offset + 64]

    @staticmethod
    def _read_chain(
        start,
        table,
        read,
        size,
        limit = None,
    ) -> bytes:
        parts, seen, sector = [], set(), start
        while sector < _DIFAT:
            if sector in seen:
                raise CompoundFileError("broken sector chain")
            seen.add(sector)
            parts.append(read(sector))
            if limit is not None and len(parts) * size >= limit:
                break
            # Some writers leave the file's last sectors out of the FAT; end the chain there.
            sector = table[sector] if sector < len(table) else _END
        out = b"".join(parts)
        return out[:limit] if limit is not None else out

    def _children(self, index: int) -> dict[str, int]:
        """Children of a storage, keyed by lowercased name (the red-black tree flattened)."""
        found: dict[str, int] = {}
        stack, seen = [self._entries[index].child], set()
        while stack:
            i = stack.pop()
            if i == _NO_STREAM or i >= len(self._entries) or i in seen:
                continue
            seen.add(i)
            entry = self._entries[i]
            found[entry.name.lower()] = i
            stack += [entry.left, entry.right]
        return found

    def _find(self, path: tuple[str, ...]) -> _Entry | None:
        index = 0
        for name in path:
            index = self._children(index).get(name.lower())
            if index is None:
                return None
        return self._entries[index]

    def exists(self, *path: str) -> bool:
        return self._find(path) is not None

    def listdir(self, *path: str) -> list[str]:
        entry_index = 0
        for name in path:
            entry_index = self._children(entry_index).get(name.lower())
            if entry_index is None:
                return []
        return [self._entries[i].name for i in self._children(entry_index).values()]

    def open(self, *path: str) -> bytes:
        entry = self._find(path)
        if entry is None or entry.kind != _STREAM:
            raise CompoundFileError(f"missing stream: {'/'.join(path)}")
        if entry.size == 0:
            return b""
        if entry.size > len(self._data):
            raise CompoundFileError("stream larger than the file")
        if entry.size < self._mini_cutoff:
            return self._read_chain(entry.start, self._mini_fat, self._mini_sector, 64, entry.size)
        return self._read_chain(entry.start, self._fat, self._raw_sector, self._sector, entry.size)
