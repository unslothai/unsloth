# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Consistent SSH restrictions for sandboxed terminal and python tools."""

from __future__ import annotations

import ast
import ipaddress
import re
import shlex
from typing import Iterable, Optional
from urllib.parse import urlsplit

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
    "paramiko.Transport",
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
        "B",
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
        "P",
        "p",
        "Q",
        "R",
        "S",
        "W",
        "w",
    }
)

_SSH_REDIRECT_OPTIONS = frozenset({"hostname", "proxyjump"})

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
    if not token or token.startswith("-") or any(c in token for c in "$`%"):
        return None
    if "/" in token and "@" not in token and ":" not in token:
        return None

    if token.startswith(("ssh://", "scp://", "sftp://")):
        try:
            return normalize_host(urlsplit(token).hostname or "") or None
        except ValueError:
            return None
    host_part = token.split("@", 1)[-1]
    if host_part.count(":") > 1 and not host_part.startswith("["):
        try:
            return str(ipaddress.IPv6Address(host_part))
        except ValueError:
            return None

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
    if not host:
        return None
    return host


def _host_from_ssh_option(key: str, value: Optional[str]) -> tuple[set[str], bool]:
    """Extract destinations or reject configuration that hides them."""
    key = key.strip().lower()
    value = (value or "").strip()
    if key in {"proxycommand", "include", "localcommand", "knownhostscommand"}:
        return set(), True
    if key not in _SSH_REDIRECT_OPTIONS:
        return set(), False
    if key == "proxyjump" and value.lower() == "none":
        return set(), False
    hosts: set[str] = set()
    dynamic = False
    for endpoint in value.split(",") if key == "proxyjump" else [value]:
        host = _extract_host_from_endpoint(endpoint)
        if host:
            hosts.add(host)
        else:
            dynamic = True
    return hosts, dynamic


def _parse_ssh_cli_options(tokens: list[str], command: str) -> tuple[list[str], set[str], bool]:
    """Parse client option arity before interpreting destination operands."""
    value_flags = _SSH_VALUE_FLAGS
    no_arg_flags = _SSH_NO_ARG_FLAGS
    if command == "scp":
        value_flags = frozenset("cDFiJloPSX")
        no_arg_flags = frozenset("346ABCOpqRrsTv")
    elif command == "sftp":
        value_flags = frozenset("BbcDFiJloPRSsX")
        no_arg_flags = frozenset("46AaCfNpqrv")
    positional: list[str] = []
    hosts: set[str] = set()
    dynamic = False
    configuration_disabled = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if token == "--":
            positional.extend(tokens[index:])
            break
        if not token.startswith("-") or token == "-":
            positional.append(token)
            if command in {"ssh", "slogin", "sftp"}:
                break
            continue
        flags = token[1:]
        while flags:
            option, flags = flags[0], flags[1:]
            if option in no_arg_flags:
                continue
            if option not in value_flags:
                dynamic = True
                break
            if flags:
                value = flags
            elif index < len(tokens):
                value = tokens[index]
                index += 1
            else:
                dynamic = True
                break
            flags = ""
            if option == "F":
                configuration_disabled = value == "none"
                dynamic |= not configuration_disabled
            elif command in {"scp", "sftp"} and option in {"D", "S"}:
                dynamic = True
            elif option == "J":
                found, unknown = _host_from_ssh_option("proxyjump", value)
                hosts.update(found)
                dynamic |= unknown
            elif option == "o":
                parts = re.split(r"[=\s]+", value.strip(), maxsplit = 1)
                found, unknown = _host_from_ssh_option(
                    parts[0], parts[1] if len(parts) == 2 else None
                )
                hosts.update(found)
                dynamic |= unknown
    return positional, hosts, dynamic or not configuration_disabled


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
    positional, opt_hosts, opt_dynamic = _parse_ssh_cli_options(tokens, cmd)
    literal_hosts.update(opt_hosts)
    dynamic = dynamic or opt_dynamic
    if cmd in {"ssh", "slogin", "sftp"}:
        candidates = positional[:1]
    else:
        candidates = _scp_remote_candidates(positional)
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


def _ssh_client_bindings(tree: ast.AST, bindings: dict[str, str]) -> dict[str, str]:
    """Map variables assigned from SSH client factories."""
    clients: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        else:
            continue
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Call):
            continue
        fq = _fq_name(node.value.func)
        root, sep, rest = fq.partition(".")
        fq = bindings.get(root, root) + (sep + rest if sep else "")
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
    root, sep, rest = fq.partition(".")
    fq = bindings.get(root, root) + (sep + rest if sep else "")
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
            if client_fq == "paramiko.Transport":
                return None
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
        if isinstance(inner, ast.Attribute) and inner.attr in {
            "SSHClient",
            "Transport",
            "Connection",
        }:
            receiver_root = (
                _fq_name(inner.value).split(".", 1)[0] if isinstance(inner.value, ast.AST) else ""
            )
            if receiver_root in _SSH_PY_ROOT_MODULES or receiver_root in bindings:
                return f"{receiver_root}.{func.attr}"
        if isinstance(inner, ast.Name):
            root = bindings.get(inner.id, inner.id).split(".", 1)[0]
            if root in _SSH_PY_ROOT_MODULES and inner.id in {
                "SSHClient",
                "Transport",
                "Connection",
            }:
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
    return extract_ssh_hosts_from_command(shlex.join(strings))


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


def _ssh_call_span(node: ast.Call) -> tuple[int, int, int, int]:
    lineno = getattr(node, "lineno", -1)
    col = getattr(node, "col_offset", 0)
    end_lineno = getattr(node, "end_lineno", None) or lineno
    end_col = getattr(node, "end_col_offset", None) or col
    return lineno, col, end_lineno, end_col


def _ssh_python_configuration_is_explicit(
    node: ast.Call, ssh_call: str, bindings: dict[str, str]
) -> bool:
    """Reject library configuration which can hide destinations."""
    if not ssh_call.startswith(("asyncssh.", "fabric.")):
        return True
    if any(kw.arg is None for kw in node.keywords):
        return False
    kwargs = {kw.arg: kw.value for kw in node.keywords}
    config = kwargs.get("config")
    if ssh_call.startswith("asyncssh."):
        if not isinstance(config, ast.Constant) or config.value is not None:
            return False
        return all(
            isinstance(kwargs[key], ast.Constant) and kwargs[key].value is None
            for key in ("options", "tunnel", "proxy_command", "sock")
            if key in kwargs
        )
    if not isinstance(config, ast.Call) or config.args or len(config.keywords) != 1:
        return False
    root, sep, rest = _fq_name(config.func).partition(".")
    factory = bindings.get(root, root) + (sep + rest if sep else "")
    lazy = config.keywords[0]
    return (
        factory in {"fabric.Config", "fabric.config.Config"}
        and lazy.arg == "lazy"
        and isinstance(lazy.value, ast.Constant)
        and lazy.value.value is True
        and (
            "gateway" not in kwargs
            or isinstance(kwargs["gateway"], ast.Constant)
            and kwargs["gateway"].value in (None, False)
        )
    )


def _scan_ssh_python_usage(
    code: str,
) -> tuple[set[str], bool, bool, list[tuple[int, int, int, int]]]:
    """Return (literal_hosts, dynamic_target, uses_ssh_connect, connect_spans)."""
    if not code or not code.strip():
        return set(), False, False, []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(), True, bool(re.search(r"\b(?:paramiko|asyncssh|fabric)\b", code)), []
    bindings = _ssh_import_bindings(tree)
    clients = _ssh_client_bindings(tree, bindings)
    shell_aliases = _shell_exec_aliases(tree)
    hosts: set[str] = set()
    dynamic = False
    uses_ssh = False
    connect_spans: list[tuple[int, int, int, int]] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            nonlocal dynamic, uses_ssh
            ssh_call = _resolve_ssh_call(node.func, bindings, clients)
            if ssh_call:
                uses_ssh = True
                dynamic |= not _ssh_python_configuration_is_explicit(node, ssh_call, bindings)
                connect_spans.append(_ssh_call_span(node))
                host_lit: Optional[str] = None
                if node.args:
                    endpoint = node.args[0]
                    if (
                        ssh_call == "paramiko.Transport"
                        and isinstance(endpoint, ast.Tuple)
                        and endpoint.elts
                    ):
                        endpoint = endpoint.elts[0]
                    host_lit = _literal_host_from_ast(endpoint)
                if host_lit is None:
                    for kw in node.keywords or []:
                        if kw.arg in {"hostname", "host"}:
                            host_lit = _literal_host_from_ast(kw.value)
                            break
                if host_lit and ssh_call in {
                    "fabric.Connection",
                    "fabric.connection.Connection",
                    "paramiko.Transport",
                }:
                    host_lit = _extract_host_from_endpoint(host_lit)
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
                            connect_spans.append(_ssh_call_span(node))
                            hosts.update(seg_hosts)
                            dynamic = dynamic or seg_dynamic
                        continue
                    for literal in argv:
                        seg_hosts, seg_dynamic = _extract_ssh_from_shell_literal(literal)
                        if seg_hosts or seg_dynamic:
                            uses_ssh = True
                            connect_spans.append(_ssh_call_span(node))
                            hosts.update(seg_hosts)
                            dynamic = dynamic or seg_dynamic

            self.generic_visit(node)

    _Visitor().visit(tree)
    return hosts, dynamic, uses_ssh, connect_spans


def extract_ssh_hosts_from_python(code: str) -> tuple[set[str], bool, bool]:
    """Return (literal_hosts, dynamic_target, uses_ssh_connect)."""
    hosts, dynamic, uses_ssh, _lines = _scan_ssh_python_usage(code)
    return hosts, dynamic, uses_ssh


def _approved_ssh_connect_spans(
    code: str, session_id: Optional[str]
) -> list[tuple[int, int, int, int]]:
    """AST spans for approved SSH connect calls (network-block filtering)."""
    hosts, dynamic, uses_ssh, connect_spans = _scan_ssh_python_usage(code)
    if not uses_ssh or dynamic or not hosts:
        return []
    if not all(is_host_approved(session_id, host) for host in hosts):
        return []
    return connect_spans


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
            "is not approved. Use -F none to disable implicit SSH configuration and "
            "a literal hostname for an approved server."
        )
    return None


def check_ssh_python_access(code: str, session_id: Optional[str]) -> Optional[str]:
    """Return an error message when python SSH usage is not allowed, else None."""
    hosts, dynamic, uses_ssh = extract_ssh_hosts_from_python(code)
    if not uses_ssh:
        return None
    if uses_ssh and not hosts and dynamic:
        return (
            "Blocked: SSH usage with a non-literal host or implicit configuration is not allowed. "
            "Disable configuration with -F none for OpenSSH, config=None for AsyncSSH, "
            "or config=fabric.Config(lazy=True) for Fabric."
        )
    if uses_ssh and not hosts and not dynamic:
        return "Blocked: SSH usage detected without a literal, approved target server."
    unapproved = _unapproved(hosts, session_id)
    if unapproved:
        listed = ", ".join(sorted(unapproved))
        return (
            f"Blocked: SSH library access to unapproved server(s): {listed}. "
            "Approve the server when prompted to allow deployment workflows."
        )
    if dynamic:
        return (
            "Blocked: SSH usage with a non-literal host or implicit configuration is not allowed. "
            "Disable configuration with -F none for OpenSSH, config=None for AsyncSSH, "
            "or config=fabric.Config(lazy=True) for Fabric."
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
    """Drop host blocks only for approved SSH connect call sites."""
    approved_spans = _approved_ssh_connect_spans(code, session_id)
    if not approved_spans:
        return analysis_info
    filtered = dict(analysis_info)
    kept: list[dict] = []
    for item in analysis_info.get("network_calls", []):
        if item.get("type") != "untrusted_host_blocked":
            kept.append(item)
            continue
        line = item.get("line", -1)
        col = item.get("col_offset", -1)
        if col < 0:
            kept.append(item)
            continue
        if any((line, col) == span[:2] for span in approved_spans):
            continue
        kept.append(item)
    filtered["network_calls"] = kept
    return filtered
