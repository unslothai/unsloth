# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pairing token. It carries a secret: never log it, never put it in a URL."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import re
from dataclasses import dataclass, field
from typing import List

PREFIX = "unsloth-pair-1."
VERSION = 1
SECRET_BYTES = 32
_HEX = re.compile(r"^[0-9a-f]+$")
_ADDR = re.compile(r"^[A-Za-z0-9.:\-]{1,253}$")


class TokenError(ValueError):
    """Not a pairing token this version understands."""


@dataclass
class Invite:
    cluster_id: str
    invite_id: str
    addrs: List[str]
    port: int
    master_fp: str
    secret: bytes = field(repr = False)
    exp: float
    v: int = VERSION


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError):
        raise TokenError("bad base64") from None


def check_addrs(addrs) -> list:
    """The address list a token may carry; the same rule decode() enforces."""
    addrs = list(addrs)
    if not _addrs_ok(addrs):
        raise TokenError(
            "bad addresses: 1 to 16 host names or IPs, letters, digits, '.', ':' and '-' only"
        )
    if any(_is_ipv6(a) for a in addrs):
        raise TokenError("IPv6 addresses are not supported yet: use an IPv4 address or a host name")
    return addrs


def _addrs_ok(addrs: list) -> bool:
    return 0 < len(addrs) <= 16 and all(isinstance(a, str) and _ADDR.match(a) for a in addrs)


def _is_ipv6(addr: str) -> bool:
    try:
        return ipaddress.ip_address(addr.strip("[]")).version == 6
    except ValueError:
        return False


def encode(invite: Invite) -> str:
    check_addrs(invite.addrs)
    body = {
        "v": invite.v,
        "c": invite.cluster_id,
        "i": invite.invite_id,
        "a": list(invite.addrs),
        "p": invite.port,
        "f": invite.master_fp,
        "s": _b64e(invite.secret),
        "e": int(invite.exp),
    }
    return PREFIX + _b64e(json.dumps(body, separators = (",", ":")).encode("utf-8"))


def _hex(value, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length or not _HEX.match(value):
        raise TokenError(f"bad {name}")
    return value


def decode(token: str) -> Invite:
    token = (token or "").strip()
    if not token.startswith(PREFIX) or len(token) > 4096:
        raise TokenError("not an unsloth pairing token")
    try:
        body = json.loads(_b64d(token[len(PREFIX) :]).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise TokenError("unreadable pairing token") from None
    if not isinstance(body, dict) or body.get("v") != VERSION:
        raise TokenError("unsupported pairing token version")
    addrs = body.get("a")
    if not isinstance(addrs, list) or not _addrs_ok(addrs):
        raise TokenError("bad addresses")
    port = body.get("p")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 < port < 65536:
        raise TokenError("bad port")
    exp = body.get("e")
    if not isinstance(exp, int) or isinstance(exp, bool):
        raise TokenError("bad expiry")
    secret_text = body.get("s")
    secret = _b64d(secret_text) if isinstance(secret_text, str) else b""
    if len(secret) != SECRET_BYTES:
        raise TokenError("bad secret")
    return Invite(
        cluster_id = _hex(body.get("c"), "cluster id", 32),
        invite_id = _hex(body.get("i"), "invite id", 32),
        addrs = addrs,
        port = port,
        master_fp = _hex(body.get("f"), "master fingerprint", 64),
        secret = secret,
        exp = float(exp),
    )
