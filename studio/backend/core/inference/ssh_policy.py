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

_FABRIC_GROUP_FQ = frozenset({"fabric.Group", "fabric.SerialGroup", "fabric.ThreadingGroup"})

_SSH_PY_CONNECT_FQ = (
    *_FABRIC_GROUP_FQ,
    "paramiko.SSHClient.connect",
    "paramiko.Transport",
    "asyncssh.connect",
    "asyncssh.connect_reverse",
    "asyncssh.create_connection",
    "asyncssh.get_server_host_key",
    "asyncssh.get_server_auth_methods",
    "asyncssh.SSHClient.connect",
    "fabric.Connection",
    "fabric.connection.Connection",
)

_PYTHON_API_SYMBOLS = {
    name: name
    for name in (
        *_SSH_PY_CONNECT_FQ,
        "paramiko.SSHClient",
        "asyncssh.SSHClient",
        "fabric.Config",
        "paramiko.ProxyCommand",
    )
}
_PYTHON_API_SYMBOLS.update(
    {
        "asyncio.subprocess.create_subprocess_exec": "asyncio.create_subprocess_exec",
        "asyncio.subprocess.create_subprocess_shell": "asyncio.create_subprocess_shell",
        "asyncssh.connection.connect": "asyncssh.connect",
        "asyncssh.connection.connect_reverse": "asyncssh.connect_reverse",
        "asyncssh.connection.create_connection": "asyncssh.create_connection",
        "asyncssh.connection.get_server_host_key": "asyncssh.get_server_host_key",
        "asyncssh.connection.get_server_auth_methods": "asyncssh.get_server_auth_methods",
        "paramiko.proxy.ProxyCommand": "paramiko.ProxyCommand",
        "paramiko.transport.Transport": "paramiko.Transport",
        "paramiko.client.SSHClient": "paramiko.SSHClient",
        "fabric.config.Config": "fabric.Config",
        "fabric.group.Group": "fabric.Group",
        "fabric.group.SerialGroup": "fabric.SerialGroup",
        "fabric.group.ThreadingGroup": "fabric.ThreadingGroup",
    }
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
        "pty.spawn",
        "os.system",
        "os.popen",
        "os.popen2",
        "os.popen3",
        "os.popen4",
        "os.execl",
        "os.execle",
        "os.execlp",
        "os.execlpe",
        "os.execv",
        "os.execve",
        "os.execvp",
        "os.execvpe",
        "os.spawnl",
        "os.spawnle",
        "os.spawnlp",
        "os.spawnlpe",
        "os.spawnv",
        "os.spawnve",
        "os.spawnvp",
        "os.spawnvpe",
        "os.posix_spawn",
        "os.posix_spawnp",
        "paramiko.ProxyCommand",
        "asyncio.create_subprocess_exec",
        "asyncio.create_subprocess_shell",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "subprocess.getoutput",
        "subprocess.getstatusoutput",
    }
)

_CMD_KWARGS = frozenset(
    {"args", "argv", "command", "executable", "path", "file", "program", "cmd", "command_line"}
)


def _extract_host_from_endpoint(token: str, *, remote_path: bool = False) -> Optional[str]:
    """Parse a user@host, host:port, host:path, or bracketed IPv6 operand."""
    token = token.strip().strip("'\"")
    if not token or token.startswith("-") or any(c in token for c in "$`%"):
        return None
    if "/" in token and "@" not in token and ":" not in token:
        return None

    if token.startswith(("ssh://", "scp://", "sftp://")):
        try:
            authority = token.split("://", 1)[1].split("/", 1)[0]
            if any(c in authority for c in "*?{}~"):
                return None
            return normalize_host(urlsplit(token).hostname or "") or None
        except ValueError:
            return None
    if remote_path:
        bracketed = False
        for index, char in enumerate(token):
            if char == "[":
                bracketed = True
            elif char == "]":
                bracketed = False
            elif char == ":" and not bracketed:
                token = token[:index]
                break
    if "@" in token and any(c in token.rsplit("@", 1)[0] for c in "*?[]{}~"):
        return None
    host_part = token.rsplit("@", 1)[-1]
    if host_part.count(":") > 1 and not host_part.startswith("["):
        try:
            return str(ipaddress.IPv6Address(host_part))
        except ValueError:
            return None

    if host_part.startswith("[") and "]" in host_part:
        host = host_part[1 : host_part.index("]")]
        try:
            return str(ipaddress.IPv6Address(host))
        except ValueError:
            return None

    if ":" in host_part:
        left, right = host_part.split(":", 1)
        if right.isdigit() and "/" not in left:
            host_part = left
        elif "/" not in left:
            host_part = left

    if any(c in host_part for c in "*?[]{}~"):
        return None
    host = normalize_host(host_part)
    if not host:
        return None
    return host


def _host_from_ssh_option(key: str, value: Optional[str]) -> tuple[set[str], bool]:
    """Extract destinations or reject configuration that hides them."""
    key = key.strip().lower()
    value = (value or "").strip()
    if key == "canonicalizehostname":
        return set(), value.lower() != "no"
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
            if command == "sftp" or command in {"ssh", "slogin"} and len(positional) > 1:
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
            elif command == "sftp" and option == "b":
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
        host = _extract_host_from_endpoint(cand, remote_path = cmd in {"scp", "sftp"})
        if host:
            literal_hosts.add(host)
        else:
            dynamic = True
    return literal_hosts, dynamic


def extract_ssh_hosts_from_command(
    command: str, *, _stdin_supplied: bool = False
) -> tuple[set[str], bool]:
    """Return literal SSH hosts and whether a dynamic/unparsed target exists."""
    if not command or not command.strip():
        return set(), False
    # Lazy import: tools imports ssh_policy at module load.
    from core.inference.tools import _find_ssh_command_segments

    hosts: set[str] = set()
    dynamic = False
    for name, arg_tokens in _find_ssh_command_segments(command):
        if name == "git":
            segment_hosts, segment_dynamic = _hosts_from_git_segment(arg_tokens)
            hosts.update(segment_hosts)
            dynamic |= segment_dynamic
            continue
        if name not in _SSH_COMMANDS:
            continue
        segment_hosts, segment_dynamic = _hosts_from_ssh_segment(name, arg_tokens)
        hosts.update(segment_hosts)
        dynamic = dynamic or segment_dynamic or name == "sftp" and _stdin_supplied
    return hosts, dynamic


def _hosts_from_git_segment(tokens: list[str]) -> tuple[set[str], bool]:
    """Reject Git transports whose SSH destinations depend on configuration."""
    from core.inference.tools import _GIT_GLOBAL_VALUE_FLAGS

    index = 0
    configured_transport = False
    while index < len(tokens) and tokens[index].startswith("-"):
        option = tokens[index]
        index += 1
        value = option
        if option in _GIT_GLOBAL_VALUE_FLAGS:
            if index == len(tokens):
                return set(), True
            value = tokens[index]
            index += 1
        if option == "-c" or option.startswith(("-c", "--config-env")):
            configured_transport |= any(
                key in value.lower() for key in ("ssh", "url.", "remote.", "include")
            )
    if index == len(tokens):
        return set(), False
    subcommand, operands = tokens[index], tokens[index + 1 :]
    if any(char in subcommand for char in "$`"):
        return set(), True
    if subcommand not in {
        "clone",
        "fetch",
        "pull",
        "push",
        "ls-remote",
        "submodule",
        "remote",
        "archive",
    }:
        return set(), False
    if subcommand == "remote" and not any(
        arg in {"update", "prune", "-f", "--fetch"} for arg in operands
    ):
        return set(), False
    if subcommand == "archive" and not any(arg.startswith("--remote") for arg in operands):
        return set(), False
    if subcommand in {"remote", "submodule"} or any(
        arg in {"--all", "--multiple", "-m"} or arg.startswith("--recurse-submodules")
        for arg in operands
    ):
        return set(), True
    value_flags = {
        "--template",
        "--reference",
        "--reference-if-able",
        "--origin",
        "--branch",
        "--revision",
        "--upload-pack",
        "--depth",
        "--shallow-since",
        "--shallow-exclude",
        "--separate-git-dir",
        "--ref-format",
        "--server-option",
        "--filter",
        "--jobs",
        "--deepen",
        "--refmap",
        "--negotiation-tip",
        "--receive-pack",
        "--exec",
        "--push-option",
        "--sort",
        "-o",
        "-j",
    }
    if subcommand == "clone":
        value_flags.update({"-b", "-u"})
    boolean_flags = {
        "-v",
        "-q",
        "-n",
        "-f",
        "-t",
        "-4",
        "-6",
        "-u",
        "-b",
        "--quiet",
        "--verbose",
        "--bare",
        "--mirror",
        "--tags",
        "--refs",
        "--symref",
        "--exit-code",
        "--dry-run",
        "--no-tags",
        "--no-checkout",
    }
    index = 0
    endpoint = None
    while index < len(operands):
        arg = operands[index]
        index += 1
        if arg == "--":
            endpoint = operands[index] if index < len(operands) else None
            break
        if arg.startswith(("--remote=", "--repo=")):
            endpoint = arg.split("=", 1)[1]
            break
        if arg in {"--remote", "--repo"}:
            endpoint = operands[index] if index < len(operands) else None
            break
        if arg.split("=", 1)[0] in value_flags:
            if "=" not in arg:
                index += 1
            continue
        if arg in boolean_flags:
            continue
        if arg.startswith("-"):
            return set(), True
        endpoint = arg
        break
    if not endpoint:
        return set(), True
    if endpoint.startswith(("http://", "https://", "git://", "file://", "/", "./", "../")):
        return set(), configured_transport
    host = _extract_host_from_endpoint(endpoint) if ":" in endpoint else None
    # git can rewrite even literal ssh urls and override core.sshcommand through the environment.
    return {host} if host else set(), True


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
                if root in _SSH_PY_ROOT_MODULES or root in {"importlib", "builtins"}:
                    local = alias.asname or root
                    bindings[local] = alias.name if alias.asname else root
        elif isinstance(node, ast.ImportFrom) and (
            _module_is_ssh_root(node.module) or node.module in {"importlib", "builtins"}
        ):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    for symbol, canonical in _PYTHON_API_SYMBOLS.items():
                        owner, _, name = symbol.rpartition(".")
                        if owner == module:
                            bindings[name] = canonical
                    continue
                local = alias.asname or alias.name
                bindings[local] = f"{module}.{alias.name}"
    for target, value in _assignment_pairs(tree):
        name = _fq_name(target)
        symbol = _bound_name(value, bindings)
        if name and symbol.split(".", 1)[0] in _SSH_PY_ROOT_MODULES:
            bindings[name] = symbol
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                factory = _bound_name(base, bindings)
                if factory.split(".", 1)[0] in _SSH_PY_ROOT_MODULES and factory.endswith(
                    (
                        ".SSHClient",
                        ".Transport",
                        ".Connection",
                        ".Group",
                        ".SerialGroup",
                        ".ThreadingGroup",
                    )
                ):
                    bindings[node.name] = factory
                    break
    clients = _ssh_client_bindings(tree, bindings)
    for target, value in _assignment_pairs(tree):
        name = _fq_name(target)
        call = _resolve_ssh_call(value, bindings, clients)
        if name and call:
            bindings[name] = call
    return bindings


def _assignment_pairs(tree: ast.AST):
    """Yield individual assignment targets, including chained and unpacked forms."""
    method_receivers: dict[int, tuple[str, str]] = {}

    def scoped_nodes(
        node: ast.AST,
        scope: str = "",
        receiver: Optional[tuple[str, str]] = None,
    ):
        if receiver:
            method_receivers[id(node)] = receiver
        yield node, scope
        if isinstance(node, ast.ClassDef):
            scope = f"{scope}.{node.name}" if scope else node.name
            receiver = None
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args.posonlyargs + node.args.args
            if (
                scope
                and arguments
                and not any(
                    _fq_name(item) == "staticmethod" for item in getattr(node, "decorator_list", ())
                )
            ):
                receiver = arguments[0].arg, scope
            scope = ""
        for child in ast.iter_child_nodes(node):
            yield from scoped_nodes(child, scope, receiver)

    nodes = list(scoped_nodes(tree))
    functions = {
        f"{scope}.{node.name}" if scope else node.name: node
        for node, scope in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {
        f"{scope}.{node.name}" if scope else node.name
        for node, scope in nodes
        if isinstance(node, ast.ClassDef)
    }
    instances: dict[str, str] = {}
    callable_refs: dict[str, ast.AST] = {}
    assigned_values: dict[str, list[ast.AST]] = {}
    builtin_aliases: dict[str, str] = {}
    for node, scope in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "builtins":
                    builtin_aliases[alias.asname or alias.name] = "builtins"
        elif isinstance(node, ast.ImportFrom) and node.module == "builtins":
            for alias in node.names:
                builtin_aliases[alias.asname or alias.name] = f"builtins.{alias.name}"
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            targets = [node.target]
        else:
            continue
        value = node.value
        owner = (
            _fq_name(value.func)
            if isinstance(value, ast.Call)
            else instances.get(_fq_name(value), "")
        )
        for target in targets:
            name = _fq_name(target)
            if name:
                assigned_values.setdefault(name, []).append(value)
            if owner in classes:
                instances[name] = owner
            if isinstance(value, (ast.Name, ast.Attribute)):
                callable_refs[name] = value

    def iterator_builtin(node: ast.AST) -> str:
        name = _fq_name(node)
        seen: set[str] = set()
        while name in callable_refs and name not in seen:
            seen.add(name)
            name = _fq_name(callable_refs[name])
        if name in functions:
            return ""
        root, separator, attribute = name.partition(".")
        name = builtin_aliases.get(name, builtin_aliases.get(root, root) + separator + attribute)
        return name.removeprefix("builtins.")

    def iterated_values(value: ast.AST, seen: frozenset[str] = frozenset()):
        name = _fq_name(value)
        if not isinstance(value, ast.Call) and name in assigned_values and name not in seen:
            for assigned in assigned_values[name]:
                yield from iterated_values(assigned, seen | {name})
        elif isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            yield from value.elts
        elif (
            isinstance(value, ast.Call)
            and iterator_builtin(value.func) in {"iter", "reversed"}
            and value.args
        ):
            yield from iterated_values(value.args[0], seen)
        else:
            yield ast.Subscript(value = value, slice = ast.Constant("*"))

    for node, scope in nodes:
        function = None
        if isinstance(node, ast.Call):
            callee = node.func
            seen: set[str] = set()
            while (name := _fq_name(callee)) in callable_refs and name not in seen:
                seen.add(name)
                callee = callable_refs[name]
            function = functions.get(_fq_name(callee))
            call_args = list(node.args)
            if _fq_name(callee) in classes:
                function = functions.get(f"{_fq_name(callee)}.__init__")
                call_args.insert(0, ast.Call(func = callee, args = [], keywords = []))
            if isinstance(callee, ast.Attribute) and _fq_name(callee) not in classes:
                receiver = callee.value
                owner = (
                    _fq_name(receiver.func)
                    if isinstance(receiver, ast.Call)
                    else instances.get(_fq_name(receiver), _fq_name(receiver))
                )
                function = functions.get(f"{owner}.{callee.attr}")
                if function is not None:
                    decorators = {_fq_name(item) for item in function.decorator_list}
                    if (
                        "classmethod" in decorators
                        or "staticmethod" not in decorators
                        and (isinstance(receiver, ast.Call) or _fq_name(receiver) not in classes)
                    ):
                        call_args.insert(0, receiver)
        if function is not None:
            parameters = function.args
            positional = parameters.posonlyargs + parameters.args
            defaults = (
                dict(
                    zip(
                        [arg.arg for arg in positional[-len(parameters.defaults) :]],
                        parameters.defaults,
                    )
                )
                if parameters.defaults
                else {}
            )
            defaults.update(
                (arg.arg, value)
                for arg, value in zip(parameters.kwonlyargs, parameters.kw_defaults)
                if value is not None
            )
            defaults.update((arg.arg, value) for arg, value in zip(positional, call_args))
            keywords, _ = _call_keyword_values(node)
            defaults.update(keywords)
            for arg in positional + parameters.kwonlyargs:
                if arg.arg in defaults:
                    yield ast.Name(id = arg.arg), defaults[arg.arg]
            continue
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    yield item.optional_vars, item.context_expr
            continue
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            pending = [(node.target, item) for item in iterated_values(node.iter)]
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            targets = [node.target]
            pending = [(target, node.value) for target in targets]
        elif isinstance(node, ast.Assign):
            targets = node.targets
            pending = [(target, node.value) for target in targets]
        else:
            continue
        while pending:
            target, value = pending.pop()
            if (
                isinstance(value, ast.Call)
                and iterator_builtin(value.func) == "next"
                and value.args
            ):
                pending.extend((target, item) for item in iterated_values(value.args[0]))
                pending.extend((target, item) for item in value.args[1:])
                continue
            if isinstance(value, ast.IfExp):
                pending.extend((target, branch) for branch in (value.body, value.orelse))
                continue
            if isinstance(value, ast.BoolOp):
                pending.extend((target, branch) for branch in value.values)
                continue
            receiver = method_receivers.get(id(node))
            name = _fq_name(target)
            if receiver and name.startswith(receiver[0] + "."):
                suffix = name[len(receiver[0]) :]
                owners = [
                    receiver[1],
                    *(key for key, owner in instances.items() if owner == receiver[1]),
                ]
                pending.extend((ast.Name(id = owner + suffix), value) for owner in owners)
            if scope and isinstance(target, ast.Name):
                pending.append((ast.Attribute(value = ast.Name(id = scope), attr = target.id), value))
            if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
                value, (ast.Tuple, ast.List)
            ):
                if len(target.elts) == len(value.elts):
                    pending.extend(zip(target.elts, value.elts))
            else:
                yield target, value
                if isinstance(value, (ast.List, ast.Tuple)):
                    for index, item in enumerate(value.elts):
                        pending.append(
                            (ast.Subscript(value = target, slice = ast.Constant(index)), item)
                        )
                        yield (
                            ast.Subscript(
                                value = target, slice = ast.Constant(index - len(value.elts))
                            ),
                            item,
                        )
                elif isinstance(value, ast.Dict):
                    for key, item in zip(value.keys, value.values):
                        if key is not None:
                            pending.append((ast.Subscript(value = target, slice = key), item))


def _ssh_client_bindings(tree: ast.AST, bindings: dict[str, str]) -> dict[str, str]:
    """Map variables and attributes assigned from SSH client factories."""
    clients: dict[str, str] = {}
    pairs = list(_assignment_pairs(tree))

    def carries_ssh_client(value: ast.AST) -> bool:
        for child in ast.walk(value):
            if isinstance(child, ast.Call) and _module_is_ssh_root(
                _bound_name(child.func, bindings)
            ):
                return True
            name = _fq_name(child)
            if name and any(
                (key == name or key.startswith((name + "[", name + ".")))
                and (factory == "unresolved" or _module_is_ssh_root(factory))
                for key, factory in clients.items()
            ):
                return True
        return False

    previous_count = -1
    while len(clients) > previous_count:
        previous_count = len(clients)
        for target, value in pairs:
            name = _fq_name(target)
            if not name:
                continue
            if isinstance(value, ast.Call):
                factory = _bound_name(value.func, bindings)
                if factory.endswith(
                    (
                        ".SSHClient",
                        ".Transport",
                        ".Connection",
                        ".Group",
                        ".SerialGroup",
                        ".ThreadingGroup",
                    )
                ):
                    clients[name] = factory
                else:
                    inputs = list(value.args) + [keyword.value for keyword in value.keywords]
                    if isinstance(value.func, ast.Attribute):
                        inputs.append(value.func.value)
                    if any(carries_ssh_client(argument) for argument in inputs):
                        clients[name] = "unresolved"
            elif _fq_name(value) in clients:
                clients[name] = clients[_fq_name(value)]
            elif isinstance(value, ast.Subscript):
                clients[name] = "unresolved"
    return clients


def _fq_name(node: Optional[ast.AST]) -> str:
    if isinstance(node, ast.NamedExpr):
        return _fq_name(node.target)
    if isinstance(node, ast.Subscript):
        container = _fq_name(node.value)
        try:
            key = ast.literal_eval(node.slice)
        except (ValueError, TypeError, SyntaxError):
            key = None
        suffix = repr(key) if isinstance(key, (str, int)) else "*"
        return f"{container}[{suffix}]" if container else ""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.insert(0, cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.insert(0, cur.id)
    elif isinstance(cur, ast.Call):
        parts.insert(0, _fq_name(cur.func))
    return ".".join(parts)


def _literal_getattr(node: Optional[ast.AST]) -> Optional[ast.Attribute]:
    if (
        isinstance(node, ast.Call)
        and _fq_name(node.func) in {"getattr", "builtins.getattr"}
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return ast.Attribute(value = node.args[0], attr = node.args[1].value)
    return None


def _bound_name(node: Optional[ast.AST], bindings: dict[str, str]) -> str:
    """Resolve imported or assigned symbols to the public SSH API names."""
    if isinstance(node, ast.Lambda) and isinstance(node.body, ast.Call):
        return _bound_name(node.body.func, bindings)
    reflected = _literal_getattr(node)
    if reflected is not None:
        node = reflected
    if isinstance(node, ast.Call):
        importer = _bound_name(node.func, bindings)
        if importer in {
            "__import__",
            "builtins.__import__",
            "importlib.__import__",
            "importlib.import_module",
        }:
            kwargs, _ = _call_keyword_values(node)
            module = node.args[0] if node.args else kwargs.get("name")
            if isinstance(module, ast.Constant) and isinstance(module.value, str):
                if importer.endswith("__import__"):
                    fromlist = node.args[3] if len(node.args) > 3 else kwargs.get("fromlist")
                    if (
                        fromlist is None
                        or isinstance(fromlist, (ast.List, ast.Tuple))
                        and not fromlist.elts
                    ):
                        return module.value.split(".", 1)[0]
                return module.value
    if isinstance(node, ast.Attribute):
        owner = _bound_name(node.value, bindings)
        name = f"{owner}.{node.attr}" if owner else node.attr
    else:
        name = _fq_name(node)
    root, sep, rest = name.partition(".")
    name = bindings.get(name, bindings.get(root, root) + (sep + rest if sep else ""))
    return _PYTHON_API_SYMBOLS.get(name, name)


def _ssh_factory_helpers(tree: ast.AST, bindings: dict[str, str]) -> None:
    """Resolve helpers returning known SSH client objects."""
    functions = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    returns: dict[str, list[ast.AST]] = {}
    for function in functions:
        pending = list(function.body)
        values = []
        while pending:
            node = pending.pop()
            if isinstance(node, ast.Return) and node.value is not None:
                values.append(node.value)
            elif not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
            ):
                pending.extend(ast.iter_child_nodes(node))
        returns[function.name] = values
    for target, value in _assignment_pairs(tree):
        if isinstance(value, ast.Lambda) and (name := _fq_name(target)):
            returns[name] = [value.body]
    while returns:
        clients = _ssh_client_bindings(tree, bindings)
        resolved = []
        for name, values in returns.items():
            for value in values:
                factory = (
                    _bound_name(value.func, bindings)
                    if isinstance(value, ast.Call)
                    else clients.get(_fq_name(value), "")
                )
                if factory.split(".", 1)[0] in _SSH_PY_ROOT_MODULES and factory.endswith(
                    (
                        ".SSHClient",
                        ".Transport",
                        ".Connection",
                        ".Group",
                        ".SerialGroup",
                        ".ThreadingGroup",
                    )
                ):
                    bindings[name] = factory
                    resolved.append(name)
                    break
        if not resolved:
            break
        for name in resolved:
            del returns[name]


def _resolve_ssh_call(
    func: ast.AST,
    bindings: dict[str, str],
    clients: dict[str, str],
    client_context: Optional[str] = None,
) -> Optional[str]:
    """Return a canonical SSH call name when ``func`` is an SSH connect/factory."""
    reflected = _literal_getattr(func)
    if reflected is not None:
        func = reflected
    if isinstance(func, ast.Name):
        bound = _bound_name(func, bindings)
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

    if client_context and isinstance(func.value, ast.Call) and _fq_name(func.value.func) == "super":
        if func.attr == "__init__" and client_context in _SSH_PY_CONNECT_FQ:
            return client_context
        if func.attr == "connect" and client_context != "paramiko.Transport":
            return f"{client_context.split('.', 1)[0]}.connect"

    fq = _bound_name(func, bindings)
    if fq in _SSH_PY_CONNECT_FQ or fq.endswith(".Connection"):
        root = fq.split(".", 1)[0]
        if root in bindings or root in _SSH_PY_ROOT_MODULES:
            return fq
        return None

    if func.attr not in _SSH_PY_CONNECT_ATTRS and func.attr != "Connection":
        return None

    client_fq = clients.get(_fq_name(func.value))
    if client_fq:
        if client_fq == "paramiko.Transport":
            return None
        root = client_fq.split(".", 1)[0]
        if root in _SSH_PY_ROOT_MODULES:
            return f"{root}.{func.attr}"
    if isinstance(func.value, ast.Name):
        root = bindings.get(func.value.id, func.value.id).split(".", 1)[0]
        if root in _SSH_PY_ROOT_MODULES:
            if func.attr == "Connection":
                return f"{root}.Connection"
            return f"{root}.{func.attr}"
        return None

    if isinstance(func.value, ast.Call):
        factory = _bound_name(func.value.func, bindings)
        root = factory.split(".", 1)[0]
        if factory == "paramiko.Transport":
            return None
        if root in _SSH_PY_ROOT_MODULES and factory.endswith((".SSHClient", ".Connection")):
            return f"{root}.{func.attr}"

    if isinstance(func.value, ast.Attribute) and func.value.attr in {
        "SSHClient",
        "Transport",
        "Connection",
    }:
        root = _fq_name(func.value).split(".", 1)[0]
        root = bindings.get(root, root).split(".", 1)[0]
        if root in _SSH_PY_ROOT_MODULES:
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


def _ssh_from_argv_literals(
    strings: list[str], stdin_supplied: bool = False
) -> tuple[set[str], bool]:
    if not strings:
        return set(), False
    return extract_ssh_hosts_from_command(shlex.join(strings), _stdin_supplied = stdin_supplied)


def _extract_ssh_from_shell_literal(
    literal: str, stdin_supplied: bool = False
) -> tuple[set[str], bool]:
    hosts, dynamic = extract_ssh_hosts_from_command(literal, _stdin_supplied = stdin_supplied)
    return hosts, dynamic


def _shell_exec_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {
                    "pty",
                    "os",
                    "subprocess",
                    "asyncio",
                    "asyncio.subprocess",
                    "paramiko",
                    "paramiko.proxy",
                }:
                    local = alias.asname or alias.name.split(".")[0]
                    aliases[local] = alias.name if alias.asname else local
        elif isinstance(node, ast.ImportFrom) and node.module in {
            "pty",
            "os",
            "subprocess",
            "asyncio",
            "asyncio.subprocess",
            "paramiko",
            "paramiko.proxy",
        }:
            module = node.module
            for alias in node.names:
                if alias.name == "*":
                    exported_module = {
                        "paramiko.proxy": "paramiko",
                        "asyncio.subprocess": "asyncio",
                    }.get(module, module)
                    for function in _SHELL_EXEC_FUNCS:
                        owner, _, name = function.rpartition(".")
                        if owner == exported_module:
                            aliases[name] = function
                    continue
                aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
    for target, value in _assignment_pairs(tree):
        name = _fq_name(target)
        symbol = _bound_name(value, aliases)
        if name and (
            symbol
            in {
                "pty",
                "os",
                "subprocess",
                "asyncio",
                "asyncio.subprocess",
                "paramiko",
                "paramiko.proxy",
            }
            or symbol in _SHELL_EXEC_FUNCS
        ):
            aliases[name] = symbol
    return aliases


def _call_keyword_values(node: ast.Call) -> tuple[dict[str, ast.AST], bool]:
    """Expand literal keyword dictionaries and identify opaque keys or mappings."""
    values: dict[str, ast.AST] = {}
    opaque = False

    def merge(mapping: ast.AST) -> None:
        nonlocal opaque
        if not isinstance(mapping, ast.Dict):
            opaque = True
            return
        for key, value in zip(mapping.keys, mapping.values):
            if key is None:
                merge(value)
            elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                values[key.value] = value
            else:
                opaque = True

    for keyword in node.keywords:
        if keyword.arg is None:
            merge(keyword.value)
        else:
            values[keyword.arg] = keyword.value
    return values, opaque


def _os_process_argv(node: ast.Call, function: str) -> list[ast.AST]:
    """Use the actual executable, ignoring argv[0] and the spawn mode/environment."""
    kwargs, _opaque = _call_keyword_values(node)
    offset = 1 if function.startswith("os.spawn") else 0
    program = (
        node.args[offset] if len(node.args) > offset else kwargs.get("path", kwargs.get("file"))
    )
    tail = list(node.args[offset + 1 :])
    vector = function.startswith(("os.execv", "os.spawnv", "os.posix_spawn"))
    if vector:
        argv = tail[0] if tail else kwargs.get("argv", kwargs.get("args"))
        tail = list(argv.elts[1:]) if isinstance(argv, (ast.List, ast.Tuple)) else [argv]
    else:
        tail = tail[1:]
        if function.endswith("e"):
            tail = tail[:-1]
    return [part for part in [program, *tail] if part is not None]


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
    if ssh_call.startswith("paramiko.") and ssh_call != "paramiko.Transport":
        if len(node.args) > 10 and not (
            isinstance(node.args[10], ast.Constant) and node.args[10].value is None
        ):
            return False
        return all(
            kw.arg is not None
            and (kw.arg != "sock" or isinstance(kw.value, ast.Constant) and kw.value.value is None)
            for kw in node.keywords
        )
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
    connect_kwargs = kwargs.get("connect_kwargs")
    if connect_kwargs is not None and not (
        isinstance(connect_kwargs, ast.Constant) and connect_kwargs.value is None
    ):
        if not isinstance(connect_kwargs, ast.Dict):
            return False
        for key, value in zip(connect_kwargs.keys, connect_kwargs.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return False
            if key.value == "sock" and not (
                isinstance(value, ast.Constant) and value.value is None
            ):
                return False
    factory = _bound_name(config.func, bindings)
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


def _expand_partial_calls(tree: ast.AST) -> ast.AST:
    """Analyze partial invocations with their effective arguments and original call spans."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name == "functools":
                    aliases[name.asname or name.name] = name.name
        elif isinstance(node, ast.ImportFrom) and node.module == "functools":
            for name in node.names:
                if name.name in {"partial", "*"}:
                    aliases[name.asname or "partial"] = "functools.partial"
    partials: dict[str, ast.Call] = {}

    def template(node: ast.AST) -> Optional[ast.Call]:
        if (
            isinstance(node, ast.Call)
            and _bound_name(node.func, aliases) == "functools.partial"
            and node.args
        ):
            return node
        return partials.get(_fq_name(node))

    for target, value in _assignment_pairs(tree):
        name = _fq_name(target)
        if _bound_name(value, aliases) in {"functools", "functools.partial"}:
            aliases[name] = _bound_name(value, aliases)
        saved = template(value)
        if saved is not None:
            partials[name] = saved

    class _Expand(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            node = self.generic_visit(node)
            seen: set[int] = set()
            while (saved := template(node.func)) is not None and id(saved) not in seen:
                seen.add(id(saved))
                old_kwargs, old_opaque = _call_keyword_values(saved)
                new_kwargs, new_opaque = _call_keyword_values(node)
                keywords = [
                    ast.keyword(arg = key, value = value)
                    for key, value in (old_kwargs | new_kwargs).items()
                ]
                if old_opaque or new_opaque:
                    keywords.extend(kw for kw in saved.keywords + node.keywords if kw.arg is None)
                node = ast.copy_location(
                    ast.Call(
                        func = saved.args[0], args = saved.args[1:] + node.args, keywords = keywords
                    ),
                    node,
                )
            return node

    return _Expand().visit(tree)


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
    tree = _expand_partial_calls(tree)
    bindings = _ssh_import_bindings(tree)
    _ssh_factory_helpers(tree, bindings)
    clients = _ssh_client_bindings(tree, bindings)
    shell_aliases = _shell_exec_aliases(tree)
    hosts: set[str] = set()
    dynamic = False
    uses_ssh = False
    connect_spans: list[tuple[int, int, int, int]] = []

    class _Visitor(ast.NodeVisitor):
        client_context: Optional[str] = None

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            previous = self.client_context
            self.client_context = bindings.get(node.name)
            self.generic_visit(node)
            self.client_context = previous

        def visit_Call(self, node: ast.Call) -> None:
            nonlocal dynamic, uses_ssh
            if (
                _fq_name(node.func) in {"getattr", "builtins.getattr"}
                and len(node.args) >= 2
                and _literal_getattr(node) is None
            ):
                receiver = node.args[0]
                owner = clients.get(_fq_name(receiver), "") or _bound_name(
                    receiver.func if isinstance(receiver, ast.Call) else receiver, bindings
                )
                if owner.split(".", 1)[0] in _SSH_PY_ROOT_MODULES:
                    uses_ssh = dynamic = True
            ssh_call = _resolve_ssh_call(node.func, bindings, clients, self.client_context)
            if ssh_call:
                uses_ssh = True
                dynamic |= not _ssh_python_configuration_is_explicit(node, ssh_call, bindings)
                connect_spans.append(_ssh_call_span(node))
                if ssh_call in _FABRIC_GROUP_FQ:
                    dynamic |= not node.args
                    for endpoint in node.args:
                        host = _literal_host_from_ast(endpoint)
                        host = _extract_host_from_endpoint(host) if host else None
                        if host:
                            hosts.add(host)
                        else:
                            dynamic = True
                    self.generic_visit(node)
                    return
                host_lit: Optional[str] = None
                host_index = 1 if ssh_call == "asyncssh.create_connection" else 0
                if len(node.args) > host_index:
                    endpoint = node.args[host_index]
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

            if (
                any(_module_is_ssh_root(value) for value in bindings.values())
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _SSH_PY_CONNECT_ATTRS
                and (
                    isinstance(node.func.value, ast.Subscript)
                    and _fq_name(node.func.value) not in clients
                    or clients.get(_fq_name(node.func.value)) == "unresolved"
                )
            ):
                uses_ssh = dynamic = True

            shell_func = _bound_name(node.func, shell_aliases)
            call_kwargs, _opaque = _call_keyword_values(node)
            stdin_supplied = any(
                key in call_kwargs
                and not (
                    isinstance(call_kwargs[key], ast.Constant) and call_kwargs[key].value is None
                )
                for key in ("stdin", "input")
            )

            if (
                shell_func
                and (
                    shell_func.startswith(("os.exec", "os.spawn", "os.posix_spawn"))
                    or shell_func == "asyncio.create_subprocess_exec"
                )
                and shell_func in _SHELL_EXEC_FUNCS
            ):
                if shell_func == "asyncio.create_subprocess_exec":
                    keyword_values, _opaque = _call_keyword_values(node)
                    program = keyword_values.get("program")
                    argv_nodes = list(node.args) or ([program] if program is not None else [])
                else:
                    argv_nodes = _os_process_argv(node, shell_func)
                argv = [text for arg in argv_nodes for text in _literal_strings_from_node(arg)]
                found, unknown = _ssh_from_argv_literals(argv, stdin_supplied)
                if found or unknown:
                    uses_ssh = True
                    hosts.update(found)
                    dynamic |= unknown or any(
                        not isinstance(arg, ast.Constant) or not isinstance(arg.value, str)
                        for arg in argv_nodes
                    )
                    connect_spans.append(_ssh_call_span(node))
                self.generic_visit(node)
                return

            if shell_func and shell_func in _SHELL_EXEC_FUNCS:
                expanded_kwargs, _opaque = _call_keyword_values(node)
                cmd_args = list(node.args) + [
                    expanded_kwargs[k] for k in _CMD_KWARGS if k in expanded_kwargs
                ]
                for arg in cmd_args:
                    argv = _literal_strings_from_node(arg)
                    if isinstance(arg, (ast.List, ast.Tuple)) and argv:
                        seg_hosts, seg_dynamic = _ssh_from_argv_literals(argv, stdin_supplied)
                        if seg_hosts or seg_dynamic:
                            uses_ssh = True
                            connect_spans.append(_ssh_call_span(node))
                            hosts.update(seg_hosts)
                            dynamic = dynamic or seg_dynamic
                        continue
                    for literal in argv:
                        seg_hosts, seg_dynamic = _extract_ssh_from_shell_literal(
                            literal, stdin_supplied
                        )
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
            "Blocked: SSH usage with a non-literal host or indirect connection settings is not allowed. "
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
            "Blocked: SSH usage with a non-literal host or indirect connection settings is not allowed. "
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
