# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Bounded data-only native host input; not runtime trust or qualification.

The launch owner supplies canonical, leased paths and an admitted generation.
Encoding this record cannot approve a runtime or give it startup authority.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import ntpath
import struct

from .content import _relative
from .content_access import appcontainer_sid
from .profiles import select_abi_adapter, WindowsRuntimeError

MAGIC = b"USLPCF1\0"
HEADER = struct.Struct("<8s8I32s32s32s")
MAX_BYTES = 65536
MAX_EXPANDED_BYTES = MAX_BYTES + 1024 * 1024 + 4
MAX_STRING_BYTES = 32766
MAX_LIST = 64


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_PROTOCOL_MISMATCH", message)


def _path(value):
    if (
        type(value) is not str
        or len(value) < 4
        or value[1:3] != ":\\"
        or not value[0].isascii()
        or not value[0].isalpha()
        or ntpath.normpath(value) != value
        or "/" in value
    ):
        raise _invalid("Host paths must be canonical absolute local paths, not roots or aliases.")
    _relative(value[3:].replace("\\", "/"))
    return value


def _string(value):
    if type(value) is not str or "\0" in value:
        raise _invalid("Invalid native host string.")
    try:
        encoded = value.encode("utf-16-le", errors = "strict")
    except UnicodeError as error:
        raise _invalid("Invalid native host Unicode.") from error
    if len(encoded) > MAX_STRING_BYTES:
        raise _invalid("Native host string exceeds its bound.")
    return struct.pack("<I", len(encoded)) + encoded


@dataclass(frozen = True)
class HostPaths:
    runtime_dll: str
    stdlib: str
    native_dir: str
    runtime_home: str
    executable: str
    prefix: str
    base_prefix: str
    policy: str
    shim: str
    workdir: str
    script: str
    private_temp: str
    aap_sentinel: str


@dataclass(frozen = True)
class HostConfiguration:
    paths: HostPaths
    package_sid: str
    version: tuple[int, int, int]
    nonce: bytes
    profile_digest: bytes
    content_digest: bytes
    packages: tuple[str, ...] = ()
    arguments: tuple[str, ...] = ()
    native_images: tuple[str, ...] = ()
    activation_plan: bytes | None = None

    def encode(self):
        if type(self.paths) is not HostPaths or type(self.version) is not tuple:
            raise _invalid("Invalid host paths or version.")
        select_abi_adapter(implementation = "cpython", version = self.version, architecture = "x64")
        if any(part > 0xFFFFFFFF for part in self.version):
            raise _invalid("Native host version is out of range.")
        for digest in (self.nonce, self.profile_digest, self.content_digest):
            if type(digest) is not bytes or len(digest) != 32:
                raise _invalid("Invalid host launch binding.")
        for values in (self.packages, self.arguments, self.native_images):
            if type(values) is not tuple or len(values) > MAX_LIST:
                raise _invalid("Native host list bound exceeded.")
        names = [_path(getattr(self.paths, field.name)) for field in fields(HostPaths)]
        appcontainer_sid(self.package_sid)
        for path in (*self.packages, *self.native_images):
            _path(path)
        for path in self.native_images:
            if not path.casefold().startswith(self.paths.runtime_home.casefold() + "\\"):
                raise _invalid("Native startup images must belong to the protected runtime.")
        body = b"".join(
            _string(value)
            for value in [
                *names,
                self.package_sid,
                *self.packages,
                *self.arguments,
                *self.native_images,
            ]
        )
        version = 1
        if self.activation_plan is not None:
            from .activation_plan import ActivationPlan

            plan = ActivationPlan.decode(
                self.activation_plan,
                nonce = self.nonce,
                profile_digest = self.profile_digest,
                content_digest = self.content_digest,
                runtime_home = ntpath.dirname(self.paths.runtime_home),
            )
            if (plan.nonce, plan.profile_digest, plan.content_digest) != (
                self.nonce,
                self.profile_digest,
                self.content_digest,
            ):
                raise _invalid("Activation plan does not match the host launch binding.")
            version = 2
        size = HEADER.size + len(body)
        if size > MAX_BYTES:
            raise _invalid("Native host configuration exceeds its bound.")
        if version == 2:
            body += struct.pack("<I", len(self.activation_plan)) + self.activation_plan
            size = HEADER.size + len(body)
            if size > MAX_EXPANDED_BYTES:
                raise _invalid("Expanded native host configuration exceeds its bound.")
        return (
            HEADER.pack(
                MAGIC,
                version,
                size,
                *self.version,
                len(self.packages),
                len(self.arguments),
                len(self.native_images),
                self.nonce,
                self.profile_digest,
                self.content_digest,
            )
            + body
        )
