# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Consistent SSH restrictions for sandboxed terminal and python tools."""

from __future__ import annotations

import ast
import re
from typing import Iterable, Optional

from state.ssh_approvals import approved_hosts, is_host_approved, normalize_host

_SSH_COMMANDS = frozenset({"ssh", "slogin", "scp", "sftp"})

_SSH_PY_ROOT_MODULES = frozenset(
    {
        "paramiko",
        "asyncssh",
        "fabric",
    }
)

_SSH_PY_CONNECT_ATTRS = frozenset({"connect", "connect_ssh"})

_SSH_PY_CONNECT_FQ = (
    "paramiko.SSHClient.connect",
    "paramiko.Transport.connect",
    "asyncssh.connect",
    "asyncssh.SSHClient.connect",
    "fabric.Connection",
    "fabric.connection.Connection",
)

def _parse_host_token(token: str) -> Optional[str]:
    token = token.strip().strip("'\"")
    if not token or token.startswith("-") or token.startswith("$") or token.startswith("${"):
        return None
    if "/" in token and "@" not in token:
        return None
    if "@" in token:
        _, host_part = token.split("@", 1)
        if ":" in host_part:
            host_part = host_part.split(":", 1)[0]
        host = normalize_host(host_part)
        return host or None
    host = normalize_host(token)
    if not host or host in {"localhost", "127.0.0.1", "::1"}:
        return None
    return host


def _hosts_from_ssh_segment(name: str, tokens: list[str]) -> tuple[set[str], bool]:
    """Extract literal hosts from one ssh/scp/sftp command segment."""
    literal_hosts: set[str] = set()
    dynamic = False
    skip_next = False
    host_tokens: list[str] = []
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in {"-p", "-i", "-F", "-l", "-o", "-b", "-c", "-J", "-L", "-R", "-D", "-W"}:
            skip_next = True
            continue
        if tok.startswith("-"):
            if "=" in tok:
                continue
            skip_next = True
            continue
        host_tokens.append(tok)
    if name.lower() in {"ssh", "slogin"}:
        # ssh [-options] [user@]host [command] -- host is the first target token
        candidates = []
        for tok in host_tokens:
            if _parse_host_token(tok):
                candidates = [tok]
                break
        if not candidates and host_tokens:
            candidates = [host_tokens[0]]
    else:
        # scp/sftp: remote target is usually the last user@host[:path] token
        candidates = [
            t for t in host_tokens if "@" in t or (":" in t and "@" in t.split(":", 1)[0])
        ]
        if not candidates and host_tokens:
            candidates = [host_tokens[-1]]
    if not candidates:
        return literal_hosts, True
    for cand in candidates:
        host = _parse_host_token(cand)
        if host:
            literal_hosts.add(host)
        else:
            dynamic = True
    return literal_hosts, dynamic


def extract_ssh_hosts_from_command(command: str) -> tuple[set[str], bool]:
    """Return literal SSH hosts and whether a dynamic/unparsed target exists."""
    if not command or not command.strip():
        return set(), False
    # Lazy import: tools imports ssh_policy at module load.
    from core.inference.tools import _find_ssh_command_segments

    hosts: set[str] = set()
    dynamic = False
    for name, arg_tokens in _find_ssh_command_segments(command):
        if name not in _SSH_COMMANDS:
            continue
        segment_hosts, segment_dynamic = _hosts_from_ssh_segment(name, arg_tokens)
        hosts.update(segment_hosts)
        dynamic = dynamic or segment_dynamic
    return hosts, dynamic


def _literal_host_from_ast(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        host = normalize_host(node.value)
        return host or None
    return None


def _module_is_ssh_root(module: Optional[str]) -> bool:
    return bool(module and module.split(".", 1)[0] in _SSH_PY_ROOT_MODULES)


def _ssh_imports_in_tree(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _SSH_PY_ROOT_MODULES:
                    roots.add(root)
        elif isinstance(node, ast.ImportFrom) and _module_is_ssh_root(node.module):
            roots.add((node.module or "").split(".", 1)[0])
    return roots


def _call_is_ssh_client_connect(func: ast.AST, ssh_imports: set[str]) -> bool:
    if not isinstance(func, ast.Attribute) or func.attr not in _SSH_PY_CONNECT_ATTRS:
        return False
    if not ssh_imports:
        return False
    receiver = func.value
    if isinstance(receiver, ast.Call):
        inner = receiver.func
        if isinstance(inner, ast.Attribute) and inner.attr in {"SSHClient", "Transport"}:
            return True
        if isinstance(inner, ast.Name) and inner.id in {"SSHClient", "Transport", "Connection"}:
            return True
    if isinstance(receiver, ast.Name):
        return True
    if isinstance(receiver, ast.Attribute) and receiver.attr in {
        "SSHClient",
        "Transport",
        "Connection",
    }:
        return True
    return bool(ssh_imports)


def _call_is_ssh_factory(func: ast.AST, ssh_imports: set[str]) -> bool:
    parts: list[str] = []
    cur = func
    while isinstance(cur, ast.Attribute):
        parts.insert(0, cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.insert(0, cur.id)
    fq = ".".join(parts)
    if fq in _SSH_PY_CONNECT_FQ or fq.endswith(".Connection"):
        return bool(ssh_imports)
    if isinstance(func, ast.Attribute) and func.attr == "Connection":
        return bool(ssh_imports)
    if isinstance(func, ast.Name) and func.id == "Connection":
        return "fabric" in ssh_imports
    if fq.endswith(".connect") and fq.split(".", 1)[0] in ssh_imports:
        return True
    return False


def extract_ssh_hosts_from_python(code: str) -> tuple[set[str], bool, bool]:
    """Return (literal_hosts, dynamic_target, uses_ssh_library)."""
    if not code or not code.strip():
        return set(), False, False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(), True, bool(re.search(r"\b(?:paramiko|asyncssh|fabric)\b", code))
    ssh_imports = _ssh_imports_in_tree(tree)
    hosts: set[str] = set()
    dynamic = False
    uses_ssh = bool(ssh_imports)

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            nonlocal dynamic, uses_ssh
            is_ssh = _call_is_ssh_client_connect(node.func, ssh_imports) or _call_is_ssh_factory(
                node.func, ssh_imports
            )
            if not is_ssh:
                self.generic_visit(node)
                return
            uses_ssh = True
            host_lit: Optional[str] = None
            if node.args:
                host_lit = _literal_host_from_ast(node.args[0])
            if host_lit is None:
                for kw in node.keywords or []:
                    if kw.arg in {"hostname", "host"}:
                        host_lit = _literal_host_from_ast(kw.value)
                        break
            if host_lit:
                hosts.add(host_lit)
            else:
                dynamic = True
            self.generic_visit(node)

    _Visitor().visit(tree)
    return hosts, dynamic, uses_ssh


def _unapproved(hosts: Iterable[str], session_id: Optional[str]) -> set[str]:
    return {h for h in hosts if h and not is_host_approved(session_id, h)}


def check_ssh_command_access(command: str, session_id: Optional[str]) -> Optional[str]:
    """Return an error message when SSH is not allowed, else None."""
    hosts, dynamic = extract_ssh_hosts_from_command(command)
    if not hosts and not dynamic:
        return None
    unapproved = _unapproved(hosts, session_id)
    if dynamic and not hosts:
        return (
            "Blocked: SSH command requires a literal, approved target server. "
            "Approve the server when prompted before connecting."
        )
    if unapproved:
        listed = ", ".join(sorted(unapproved))
        return (
            f"Blocked: SSH access to unapproved server(s): {listed}. "
            "Approve the server when prompted to allow deployment workflows."
        )
    if dynamic and unapproved:
        return (
            "Blocked: SSH command includes a non-literal host that is not approved. "
            "Use a literal hostname for an approved server."
        )
    if dynamic:
        # Literal host approved, but other parts are dynamic: allow.
        return None
    return None


def check_ssh_python_access(code: str, session_id: Optional[str]) -> Optional[str]:
    """Return an error message when python SSH usage is not allowed, else None."""
    hosts, dynamic, uses_ssh = extract_ssh_hosts_from_python(code)
    if not uses_ssh and not hosts and not dynamic:
        return None
    if uses_ssh and not hosts and dynamic:
        return (
            "Blocked: SSH library usage with a non-literal host requires approving "
            "the target server first."
        )
    if uses_ssh and not hosts and not dynamic:
        return None
    unapproved = _unapproved(hosts, session_id)
    if unapproved:
        listed = ", ".join(sorted(unapproved))
        return (
            f"Blocked: SSH library access to unapproved server(s): {listed}. "
            "Approve the server when prompted to allow deployment workflows."
        )
    if dynamic:
        return (
            "Blocked: SSH library usage with a non-literal host requires approving "
            "the target server first."
        )
    return None


def collect_ssh_hosts_for_approval(name: str, arguments: dict) -> set[str]:
    """Hosts to auto-approve when the user allows a gated tool call."""
    if name == "terminal":
        hosts, _dynamic = extract_ssh_hosts_from_command(str(arguments.get("command", "")))
        return hosts
    if name == "python":
        hosts, _dynamic, _uses = extract_ssh_hosts_from_python(str(arguments.get("code", "")))
        return hosts
    return set()


def list_approved_ssh_hosts(session_id: Optional[str]) -> list[str]:
    return sorted(approved_hosts(session_id))


def filter_ssh_approved_network_blocks(
    code: str, session_id: Optional[str], analysis_info: dict
) -> dict:
    """Drop informational-host blocks for SSH targets the session already approved."""
    hosts, dynamic, uses_ssh = extract_ssh_hosts_from_python(code)
    if not uses_ssh or dynamic or not hosts:
        return analysis_info
    if not all(is_host_approved(session_id, host) for host in hosts):
        return analysis_info
    filtered = dict(analysis_info)
    filtered["network_calls"] = [
        item
        for item in analysis_info.get("network_calls", [])
        if item.get("type") != "untrusted_host_blocked"
    ]
    return filtered
