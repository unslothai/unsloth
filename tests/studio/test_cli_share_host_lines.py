# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression for #8868: CLI URLs use the LAN host, not the public IP.

The command bodies are checked structurally because they bind, load, and block.
"""

from __future__ import annotations

import ast
from pathlib import Path

from unsloth_cli.commands.studio import (
    _display_host_for_bind,
    _network_share_host_for_bind,
    _openable_host_for_bind,
)

_STUDIO_CLI = Path(__file__).resolve().parents[2] / "unsloth_cli" / "commands" / "studio.py"

PUBLIC_IP = "104.32.48.18"
LAN_IP = "192.168.1.50"


class _RunModule:
    """Current backend resolver surface."""

    @staticmethod
    def _display_host_for_bind(host):
        return PUBLIC_IP if host in ("0.0.0.0", "::") else host

    @staticmethod
    def _network_share_host_for_bind(host):
        return LAN_IP if host in ("0.0.0.0", "::") else host


class _OlderRunModule:
    """Compatibility surface for older backends."""

    @staticmethod
    def _display_host_for_bind(host):
        return PUBLIC_IP if host in ("0.0.0.0", "::") else host


class _NoLanRunModule:
    """WSL NAT or a loopback-only host: nothing a LAN peer could open."""

    @staticmethod
    def _display_host_for_bind(host):
        return PUBLIC_IP if host in ("0.0.0.0", "::") else host

    @staticmethod
    def _network_share_host_for_bind(host):
        return host


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(_STUDIO_CLI.read_text(encoding = "utf-8"))
    matches = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected exactly one {name}() in studio.py, found {len(matches)}"
    return matches[0]


def _called_names(node: ast.AST) -> list[str]:
    return [
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    ]


def test_share_host_shim_returns_the_lan_address():
    assert _network_share_host_for_bind(_RunModule, "0.0.0.0") == LAN_IP
    assert _display_host_for_bind(_RunModule, "0.0.0.0") == PUBLIC_IP


def test_share_host_shim_leaves_a_specific_bind_alone():
    assert _network_share_host_for_bind(_RunModule, "192.168.1.7") == "192.168.1.7"


def test_share_host_shim_keeps_the_only_answer_an_older_run_py_has():
    assert _network_share_host_for_bind(_OlderRunModule, "0.0.0.0") == PUBLIC_IP


def test_openable_host_prefers_the_lan_address():
    assert _openable_host_for_bind(_RunModule, "0.0.0.0") == LAN_IP


def test_openable_host_leaves_a_specific_bind_alone():
    assert _openable_host_for_bind(_RunModule, "192.168.1.7") == "192.168.1.7"


def test_openable_host_falls_back_to_loopback_with_no_lan_address():
    """A URL to open is never the wildcard: no browser can reach it."""
    assert _openable_host_for_bind(_NoLanRunModule, "0.0.0.0") == "127.0.0.1"
    assert _openable_host_for_bind(_NoLanRunModule, "::") == "::1"


def test_launch_line_never_resolves_the_internet_facing_address():
    called = _called_names(_function("studio_default"))
    assert "_display_host_for_bind" not in called
    assert "_openable_host_for_bind" in called


def test_run_banner_keeps_the_public_address_for_the_reachability_probe_only():
    called = _called_names(_function("run"))
    # The remaining call feeds public reachability.
    assert called.count("_display_host_for_bind") == 1
    assert "_openable_host_for_bind" in called
