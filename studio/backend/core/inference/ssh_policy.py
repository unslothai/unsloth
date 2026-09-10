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

# OpenSSH boolean flags (do not consume the next token).
_SSH_NO_ARG_FLAGS = frozenset(
    {
        "4",
        "6",
        "A",
        "a",
        "C",
        "f",
        "G",
        "g",
        "K",
        "k",
        "M",
        "N",
        "n",
        "q",
        "s",
        "T",
        "t",
        "V",
        "v",
        "X",
        "x",
        "Y",
        "y",
    }
)

# OpenSSH options that take a separate value token.
_SSH_VALUE_FLAGS = frozenset(
    {
        "b",
        "c",
        "D",
        "E",
        "e",
        "F",
        "I",
        "i",
        "J",
        "L",
        "l",
        "m",
        "O",
        "o",
        "p",
        "Q",
        "R",
        "S",
        "W",
        "w",
    }
)

_SSH_REDIRECT_OPTIONS = frozenset({"hostname", "proxyjump"})
_SSH_DYNAMIC_OPTIONS = frozenset({"proxycommand"})

_SHELL_EXEC_FUNCS = frozenset(
    {
        "os.system",
        "os.popen",
        "os.popen2",
        "os.popen3",
        "os.popen4",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "subprocess.getoutput",
        "subprocess.getstatusoutput",
    }
)

_CMD_KWARGS = frozenset({"args", "command", "executable", "path", "file"})


def _extract_host_from_endpoint(token: str) -> Optional[str]:
    """Parse a user@host, host:port, host:path, or bracketed IPv6 operand."""
    token = token.strip().strip("'\"")
    if not token or token.startswith("-") or token.startswith("$") or token.startswith("${"):
        return None
    if "/" in token and "@" not in token and ":" not in token:
        return None

    host_part = token.split("@", 1)[-1]

    if host_part.startswith("[") and "]" in host_part:
        host = host_part[1 : host_part.index("]")]
        host = normalize_host(host)
        return host or None

    if ":" in host_part:
        left, right = host_part.split(":", 1)
        if right.isdigit() and "/" not in left:
            host_part = left
        elif "/" not in left:
            host_part = left

    host = normalize_host(host_part)
    if not host or host in {"localhost", "127.0.0.1", "::1"}:
        return None
    return host


def _host_from_ssh_option(key: str, value: Optional[str]) -> tuple[set[str], bool]:
    """Extract hosts from one ``-o`` key/value pair."""
    hosts: set[str] = set()
    dynamic = False
    key = (key or "").strip().lower()
    value = (value or "").strip()
    if key in _SSH_DYNAMIC_OPTIONS:
        return hosts, True
    if key in _SSH_REDIRECT_OPTIONS and value:
        host = _extract_host_from_endpoint(value)
        if host:
            hosts.add(host)
        else:
            dynamic = True
    return hosts, dynamic


def _parse_ssh_option_token(tok: str) -> tuple[Optional[str], Optional[str]]:
    if "=" in tok:
        key, value = tok.split("=", 1)
        return key.strip().lower(), value.strip()
    return tok.strip().lower(), None


def _consume_ssh_flag(
    flag: str, tokens: list[str], index: int
) -> tuple[int, set[str], bool]:
    """Advance past one ssh flag and return any hosts it names."""
    hosts: set[str] = set()
    dynamic = False
    flag_body = flag[1:]
    if not flag_body:
        return index, hosts, dynamic

    if flag_body[0] in _SSH_NO_ARG_FLAGS and all(ch in _SSH_NO_ARG_FLAGS for ch in flag_body):
        return index, hosts, dynamic

    if flag_body[0] in _SSH_VALUE_FLAGS:
        opt = flag_body[0]
        attached = flag_body[1:]
        if opt == "o":
            if attached:
                key, value = _parse_ssh_option_token(attached)
                opt_hosts, opt_dynamic = _host_from_ssh_option(key, value)
                hosts.update(opt_hosts)
                dynamic = dynamic or opt_dynamic
            elif index < len(tokens):
                key, value = _parse_ssh_option_token(tokens[index])
                if value is None and index + 1 < len(tokens):
                    value = tokens[index + 1]
                    index += 2
                else:
                    index += 1
                opt_hosts, opt_dynamic = _host_from_ssh_option(key, value)
                hosts.update(opt_hosts)
                dynamic = dynamic or opt_dynamic
            return index, hosts, dynamic
        if attached:
            if opt == "J":
                host = _extract_host_from_endpoint(attached)
                if host:
                    hosts.add(host)
                else:
                    dynamic = True
            return index, hosts, dynamic
        if index < len(tokens):
            value = tokens[index]
            index += 1
            if opt == "J":
                host = _extract_host_from_endpoint(value)
                if host:
                    hosts.add(host)
                else:
                    dynamic = True
            elif opt == "o":
                key, opt_value = _parse_ssh_option_token(value)
                if opt_value is None and index < len(tokens):
                    opt_value = tokens[index]
                    index += 1
                opt_hosts, opt_dynamic = _host_from_ssh_option(key, opt_value)
                hosts.update(opt_hosts)
                dynamic = dynamic or opt_dynamic
        return index, hosts, dynamic

    return index, hosts, dynamic


def _parse_ssh_cli_options(tokens: list[str]) -> tuple[list[str], set[str], bool]:
    """Return positional tokens plus hosts named by ssh options."""
    positional: list[str] = []
    hosts: set[str] = set()
    dynamic = False
    index = 0
    while index < len(tokens):
        tok = tokens[index]
        if tok == "--":
            positional.extend(tokens[index + 1 :])
            break
        if tok.startswith("-") and tok != "-":
            index += 1
            index, opt_hosts, opt_dynamic = _consume_ssh_flag(tok, tokens, index)
            hosts.update(opt_hosts)
            dynamic = dynamic or opt_dynamic
            continue
        positional.append(tok)
        index += 1
    return positional, hosts, dynamic


def _scp_remote_candidates(tokens: list[str]) -> list[str]:
    """Return scp/sftp operands that name a remote host."""
    candidates: list[str] = []
    for tok in tokens:
        if "@" in tok:
            candidates.append(tok)
            continue
        if ":" in tok:
            left, right = tok.split(":", 1)
            if left and "/" not in left and not re.fullmatch(r"[A-Za-z]:", left):
                candidates.append(tok)
    return candidates


def _hosts_from_ssh_segment(name: str, tokens: list[str]) -> tuple[set[str], bool]:
    """Extract literal hosts from one ssh/scp/sftp command segment."""
    literal_hosts: set[str] = set()
    dynamic = False
    cmd = name.lower()
    if cmd in {"ssh", "slogin"}:
        positional, opt_hosts, opt_dynamic = _parse_ssh_cli_options(tokens)
        literal_hosts.update(opt_hosts)
        dynamic = dynamic or opt_dynamic
        candidates: list[str] = []
        for tok in positional:
            if _extract_host_from_endpoint(tok):
                candidates = [tok]
                break
        if not candidates and positional:
            candidates = [positional[0]]
    else:
        candidates = _scp_remote_candidates(tokens)
        if not candidates and tokens:
            candidates = [tokens[-1]]
    if not candidates:
        return literal_hosts, True
    for cand in candidates:
        host = _extract_host_from_endpoint(cand)
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


def _ssh_import_bindings(tree: ast.AST) -> dict[str, str]:
    """Map local names to fully-qualified SSH symbols."""
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _SSH_PY_ROOT_MODULES:
                    local = alias.asname or root
                    bindings[local] = alias.name
        elif isinstance(node, ast.ImportFrom) and _module_is_ssh_root(node.module):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                bindings[local] = f"{module}.{alias.name}"
    return bindings


def _ssh_client_bindings(tree: ast.AST) -> dict[str, str]:
    """Map variables assigned from SSH client factories."""
    clients: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Call):
            continue
        fq = _fq_name(node.value.func)
        if fq.endswith(".SSHClient") or fq.endswith(".Transport") or fq.endswith(".Connection"):
            clients[target.id] = fq
    return clients


def _fq_name(node: ast.AST) -> str:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.insert(0, cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.insert(0, cur.id)
    return ".".join(parts)


def _resolve_ssh_call(
    func: ast.AST, bindings: dict[str, str], clients: dict[str, str]
) -> Optional[str]:
    """Return a canonical SSH call name when ``func`` is an SSH connect/factory."""
    if isinstance(func, ast.Name):
        bound = bindings.get(func.id, func.id)
        if bound in _SSH_PY_CONNECT_FQ:
            return bound
        if bound.endswith(".connect") and bound.split(".", 1)[0] in _SSH_PY_ROOT_MODULES:
            return bound
        if bound.endswith(".Connection"):
            return bound
        if func.id == "Connection" and any(v.endswith(".Connection") for v in bindings.values()):
            return "fabric.Connection"
        return None

    if not isinstance(func, ast.Attribute):
        return None

    fq = _fq_name(func)
    if fq in _SSH_PY_CONNECT_FQ or fq.endswith(".Connection"):
        root = fq.split(".", 1)[0]
        if root in bindings or root in _SSH_PY_ROOT_MODULES:
            return fq
        return None

    if func.attr not in _SSH_PY_CONNECT_ATTRS and func.attr != "Connection":
        return None

    if isinstance(func.value, ast.Name):
        client_fq = clients.get(func.value.id)
        if client_fq:
            root = client_fq.split(".", 1)[0]
            if root in _SSH_PY_ROOT_MODULES:
                return f"{root}.{func.attr}"
        root = bindings.get(func.value.id, func.value.id).split(".", 1)[0]
        if root in _SSH_PY_ROOT_MODULES:
            if func.attr == "Connection":
                return f"{root}.Connection"
            return f"{root}.{func.attr}"
        return None

    if isinstance(func.value, ast.Call):
        inner = func.value.func
        if isinstance(inner, ast.Attribute) and inner.attr in {"SSHClient", "Transport", "Connection"}:
            receiver_root = _fq_name(inner.value).split(".", 1)[0] if isinstance(inner.value, ast.AST) else ""
            if receiver_root in _SSH_PY_ROOT_MODULES or receiver_root in bindings:
                return f"{receiver_root}.{func.attr}"
        if isinstance(inner, ast.Name):
            root = bindings.get(inner.id, inner.id).split(".", 1)[0]
            if root in _SSH_PY_ROOT_MODULES and inner.id in {"SSHClient", "Transport", "Connection"}:
                return f"{root}.{func.attr}"

    if isinstance(func.value, ast.Attribute) and func.value.attr in {
        "SSHClient",
        "Transport",
        "Connection",
    }:
        root = _fq_name(func.value).split(".", 1)[0]
        if root in _SSH_PY_ROOT_MODULES or root in bindings:
            return f"{root}.{func.attr}"

    return None


def _literal_strings_from_node(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple)):
        out: list[str] = []
        for elt in node.elts:
            out.extend(_literal_strings_from_node(elt))
        return out
    return []


def _ssh_from_argv_literals(strings: list[str]) -> tuple[set[str], bool]:
    if not strings:
        return set(), False
    cmd = strings[0].rsplit("/", 1)[-1].lower()
    if cmd not in _SSH_COMMANDS:
        return set(), False
    return _hosts_from_ssh_segment(cmd, strings[1:])


def _extract_ssh_from_shell_literal(literal: str) -> tuple[set[str], bool]:
    hosts, dynamic = extract_ssh_hosts_from_command(literal)
    return hosts, dynamic


def _shell_exec_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    aliases[alias.asname or "os"] = "os"
                elif alias.name == "subprocess":
                    aliases[alias.asname or "subprocess"] = "subprocess"
        elif isinstance(node, ast.ImportFrom) and node.module in {"os", "subprocess"}:
            module = node.module
            for alias in node.names:
                if alias.name == "*":
                    continue
                aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
    return aliases


def _scan_ssh_python_usage(code: str) -> tuple[set[str], bool, bool, set[int]]:
    """Return (literal_hosts, dynamic_target, uses_ssh_connect, connect_line_numbers)."""
    if not code or not code.strip():
        return set(), False, False, set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(), True, bool(re.search(r"\b(?:paramiko|asyncssh|fabric)\b", code)), set()
    bindings = _ssh_import_bindings(tree)
    clients = _ssh_client_bindings(tree)
    shell_aliases = _shell_exec_aliases(tree)
    hosts: set[str] = set()
    dynamic = False
    uses_ssh = False
    connect_lines: set[int] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            nonlocal dynamic, uses_ssh
            ssh_call = _resolve_ssh_call(node.func, bindings, clients)
            if ssh_call:
                uses_ssh = True
                connect_lines.add(getattr(node, "lineno", -1))
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
                return

            shell_func: Optional[str] = None
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                module = shell_aliases.get(node.func.value.id)
                if module in {"os", "subprocess"}:
                    shell_func = f"{module}.{node.func.attr}"
            elif isinstance(node.func, ast.Name):
                shell_func = shell_aliases.get(node.func.id)

            if shell_func and shell_func in _SHELL_EXEC_FUNCS:
                expanded_kwargs: dict[str, ast.AST] = {}
                for kw in node.keywords or []:
                    if kw.arg is not None:
                        expanded_kwargs[kw.arg] = kw.value
                cmd_args = list(node.args) + [
                    expanded_kwargs[k] for k in _CMD_KWARGS if k in expanded_kwargs
                ]
                for arg in cmd_args:
                    argv = _literal_strings_from_node(arg)
                    if isinstance(arg, (ast.List, ast.Tuple)) and argv:
                        seg_hosts, seg_dynamic = _ssh_from_argv_literals(argv)
                        if seg_hosts or seg_dynamic:
                            uses_ssh = True
                            connect_lines.add(getattr(node, "lineno", -1))
                            hosts.update(seg_hosts)
                            dynamic = dynamic or seg_dynamic
                        continue
                    for literal in argv:
                        seg_hosts, seg_dynamic = _extract_ssh_from_shell_literal(literal)
                        if seg_hosts or seg_dynamic:
                            uses_ssh = True
                            connect_lines.add(getattr(node, "lineno", -1))
                            hosts.update(seg_hosts)
                            dynamic = dynamic or seg_dynamic

            self.generic_visit(node)

    _Visitor().visit(tree)
    return hosts, dynamic, uses_ssh, connect_lines


def extract_ssh_hosts_from_python(code: str) -> tuple[set[str], bool, bool]:
    """Return (literal_hosts, dynamic_target, uses_ssh_connect)."""
    hosts, dynamic, uses_ssh, _lines = _scan_ssh_python_usage(code)
    return hosts, dynamic, uses_ssh


def _approved_ssh_connect_lines(code: str, session_id: Optional[str]) -> set[int]:
    """Lines with approved SSH connect calls, for network-block filtering."""
    hosts, dynamic, uses_ssh, connect_lines = _scan_ssh_python_usage(code)
    if not uses_ssh or dynamic or not hosts:
        return set()
    if not all(is_host_approved(session_id, host) for host in hosts):
        return set()
    return connect_lines


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
    if dynamic:
        return (
            "Blocked: SSH command includes a non-literal or redirected target that "
            "is not approved. Use a literal hostname for an approved server."
        )
    return None


def check_ssh_python_access(code: str, session_id: Optional[str]) -> Optional[str]:
    """Return an error message when python SSH usage is not allowed, else None."""
    hosts, dynamic, uses_ssh = extract_ssh_hosts_from_python(code)
    if not uses_ssh:
        return None
    if uses_ssh and not hosts and dynamic:
        return (
            "Blocked: SSH library usage with a non-literal host requires approving "
            "the target server first."
        )
    if uses_ssh and not hosts and not dynamic:
        return (
            "Blocked: SSH usage detected without a literal, approved target server."
        )
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
    """Drop host blocks only on lines with already-approved SSH connect calls."""
    approved_lines = _approved_ssh_connect_lines(code, session_id)
    if not approved_lines:
        return analysis_info
    filtered = dict(analysis_info)
    filtered["network_calls"] = [
        item
        for item in analysis_info.get("network_calls", [])
        if not (
            item.get("type") == "untrusted_host_blocked"
            and item.get("line") in approved_lines
        )
    ]
    return filtered
