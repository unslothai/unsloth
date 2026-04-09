# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Tool definitions and executors for LLM tool calling.

Supports web search (DuckDuckGo), Python code execution, and terminal commands.
"""

import ast
import http.client
import os

os.environ["UNSLOTH_IS_PRESENT"] = "1"

import random
import re
import shlex
import ssl
import subprocess
import sys
import tempfile
import threading
import urllib.request

from loggers import get_logger

logger = get_logger(__name__)

_EXEC_TIMEOUT = 300  # 5 minutes

# Pre-import modules used in _sandbox_preexec at module level so that
# the preexec_fn closure does not trigger the import machinery in the
# forked child (which can deadlock in multi-threaded servers).
_libc = None
if sys.platform == "linux":
    try:
        import ctypes
        import ctypes.util

        _libc_name = ctypes.util.find_library("c")
        if _libc_name:
            _libc = ctypes.CDLL(_libc_name, use_errno = True)
    except (OSError, AttributeError):
        pass

_resource = None
if sys.platform != "win32":
    try:
        import resource as _resource
    except ImportError:
        pass

# Strict raster-image allowlist for sandbox file serving.
# No .svg (XSS risk via embedded scripts), no .html, no .pdf.
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_MAX_OUTPUT_CHARS = 8000  # truncate long output
_BLOCKED_COMMANDS_COMMON = frozenset(
    {
        "rm",
        "sudo",
        "su",
        "dd",
        "chmod",
        "chown",
        "mkfs",
        "shutdown",
        "reboot",
        "passwd",
        "mount",
        "umount",
        "fdisk",
        "kill",
        "killall",
        "pkill",
    }
)
_BLOCKED_COMMANDS_WIN = frozenset(
    {
        "rmdir",
        "takeown",
        "icacls",
        "runas",
        "powershell",
        "pwsh",
    }
)
_BLOCKED_COMMANDS = (
    _BLOCKED_COMMANDS_COMMON | _BLOCKED_COMMANDS_WIN
    if sys.platform == "win32"
    else _BLOCKED_COMMANDS_COMMON
)


def _find_blocked_commands(command: str) -> set[str]:
    """Detect blocked commands using shlex tokenization and regex scanning.

    Catches: full paths (/usr/bin/sudo), quoted strings ("sudo"),
    split-quotes (su""do), backslash escapes (\\rm), and command-position
    words after ;, |, &&, $().
    """
    blocked = set()

    # 1. shlex tokenization (handles quotes, escapes, concatenation)
    try:
        tokens = (
            shlex.split(command)
            if sys.platform != "win32"
            else shlex.split(command, posix = False)
        )
    except ValueError:
        tokens = command.split()

    for token in tokens:
        base = os.path.basename(token).lower()
        # Strip common Windows executable extensions so that
        # runas.exe, shutdown.bat, etc. match the blocklist.
        stem, ext = os.path.splitext(base)
        if ext in {".exe", ".com", ".bat", ".cmd"}:
            base = stem
        if base in _BLOCKED_COMMANDS:
            blocked.add(base)

    # 2. Regex: catch blocked words at shell command boundaries
    #    (semicolons, pipes, &&, ||, backticks, $(), <(), subshells, newlines)
    #    Uses a single combined pattern for all blocked words.
    #    Handles optional Unix path prefix (/usr/bin/) and Windows drive
    #    letter prefix (C:\Windows\...\).
    lowered = command.lower()
    if _BLOCKED_COMMANDS:
        words_alt = "|".join(re.escape(w) for w in sorted(_BLOCKED_COMMANDS))
        pattern = (
            rf"(?:^|[;&|`\n(]\s*|[$]\(\s*|<\(\s*)"
            rf"(?:[\w./\\-]*/|[a-zA-Z]:[/\\][\w./\\-]*)?"
            rf"({words_alt})(?:\.(?:exe|com|bat|cmd))?\b"
        )
        blocked.update(re.findall(pattern, lowered))

    # 3. Check for nested shell invocations (bash -c 'sudo whoami',
    #    bash -lc '...', bash --login -c '...', cmd /c '...').
    #    When a -c or /c flag is found, look backwards for a shell name
    #    (skipping intermediate flags like --login, -l, -x) and recursively
    #    scan the nested command string.
    _SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "csh", "tcsh", "fish"}
    _SHELLS_WIN = {"cmd", "cmd.exe"}
    for i, token in enumerate(tokens):
        tok_lower = token.lower()
        # Match -c exactly, or combined flags ending in c (e.g. -lc, -xc)
        is_unix_c = tok_lower == "-c" or (
            tok_lower.startswith("-")
            and tok_lower.endswith("c")
            and not tok_lower.startswith("--")
        )
        is_win_c = tok_lower == "/c"
        if not (is_unix_c or is_win_c) or i < 1 or i + 1 >= len(tokens):
            continue
        # Look backwards past any flags to find the shell binary.
        # On Unix, flags start with - (skip those). On Windows, flags
        # start with / but so do absolute paths, so only skip short
        # single-char /X flags (not /bin/bash style paths).
        for j in range(i - 1, -1, -1):
            prev = tokens[j]
            if prev.startswith("-"):
                continue  # skip Unix flags like --login, -l
            if is_win_c and prev.startswith("/") and len(prev) <= 3:
                continue  # skip Windows flags like /s, /q (not /bin/bash)
            prev_base = os.path.basename(prev).lower()
            if is_unix_c and prev_base in _SHELLS:
                blocked |= _find_blocked_commands(tokens[i + 1])
            elif is_win_c and prev_base in _SHELLS_WIN:
                blocked |= _find_blocked_commands(tokens[i + 1])
            break  # stop at first non-flag token

    return blocked


def _build_safe_env(workdir: str) -> dict[str, str]:
    """Build a minimal, credential-free environment for sandboxed subprocesses.

    Strips HF_TOKEN, WANDB_API_KEY, AWS_*, GH_TOKEN, LD_PRELOAD, DYLD_*, etc.
    Preserves the active Python interpreter and virtualenv directories in PATH
    so that pip, uv, and packages installed in the Studio runtime remain
    accessible.
    """
    # Start with the directory containing the running Python interpreter
    # so that subprocess calls to 'python', 'pip', etc. resolve to the
    # same environment the Studio server is running in.
    exe_dir = os.path.dirname(sys.executable)
    path_entries = [exe_dir] if exe_dir else []

    # If a virtualenv is active, include its bin/Scripts directory.
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        venv_bin = os.path.join(venv, "Scripts" if sys.platform == "win32" else "bin")
        if venv_bin not in path_entries:
            path_entries.append(venv_bin)

    if sys.platform == "win32":
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        path_entries.extend([os.path.join(sysroot, "System32"), sysroot])
    else:
        path_entries.extend(["/usr/local/bin", "/usr/bin", "/bin"])

    # Deduplicate while preserving order
    deduped = list(dict.fromkeys(p for p in path_entries if p))

    env = {
        "PATH": os.pathsep.join(deduped),
        "HOME": workdir,
        "TMPDIR": workdir,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TERM": "dumb",
        "PYTHONIOENCODING": "utf-8",
    }
    if venv:
        env["VIRTUAL_ENV"] = venv
    # Windows needs SystemRoot for Python/subprocess to work
    if sys.platform == "win32":
        env["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")
    return env


def _sandbox_preexec():
    """Pre-exec hook: drop privilege escalation ability and set resource limits.

    On Linux, applies PR_SET_NO_NEW_PRIVS so sudo/su/pkexec fail at the
    kernel level. On Linux and macOS, sets RLIMIT_FSIZE.
    No-op on Windows (use creationflags instead).

    Note: RLIMIT_NPROC is intentionally NOT set because Linux enforces it
    per real UID, not per process tree, so it would starve the Studio
    server and other sessions sharing the same user account.

    All modules and handles are resolved at import time (module level) so
    this function does not trigger Python imports in the forked child,
    avoiding potential deadlocks in multi-threaded servers.
    """
    if _libc is not None:
        try:
            # PR_SET_NO_NEW_PRIVS = 38, arg2 = 1 (enable)
            _libc.prctl(38, 1, 0, 0, 0)
        except (OSError, AttributeError):
            pass  # Not available (container, old kernel, etc.)

    if _resource is not None:
        try:
            # Limit file size to 100MB (prevents disk filling)
            _resource.setrlimit(
                _resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024)
            )
        except (ValueError, OSError):
            pass


def _get_shell_cmd(command: str) -> list[str]:
    """Return the platform-appropriate shell invocation for a command string."""
    if sys.platform == "win32":
        return ["cmd", "/c", command]
    return ["bash", "-c", command]


# Per-session working directories so each chat thread gets its own sandbox.
# Falls back to a shared ~/studio_sandbox/_default for API callers without a
# session_id.
_workdirs: dict[str, str] = {}


def _get_workdir(session_id: str | None = None) -> str:
    """Return (and lazily create) a persistent working directory for tool execution."""
    global _workdirs
    key = session_id or "_default"
    if key not in _workdirs or not os.path.isdir(_workdirs[key]):
        home = os.path.expanduser("~")
        sandbox_root = os.path.join(home, "studio_sandbox")
        if session_id:
            # Sanitize: strip path separators and parent-dir references
            safe_id = os.path.basename(session_id.replace("..", ""))
            if not safe_id:
                safe_id = "_invalid"
            workdir = os.path.join(sandbox_root, safe_id)
            # Verify resolved path stays under sandbox root
            if not os.path.realpath(workdir).startswith(os.path.realpath(sandbox_root)):
                workdir = os.path.join(sandbox_root, "_invalid")
        else:
            workdir = os.path.join(sandbox_root, "_default")
        os.makedirs(workdir, exist_ok = True)
        _workdirs[key] = workdir
    return _workdirs[key]


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web and fetch page content. Returns snippets for all results. "
            "Use the url parameter to fetch full page text from a specific URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "url": {
                    "type": "string",
                    "description": "A URL to fetch full page content from (instead of searching). Use this to read a page found in search results.",
                },
            },
            "required": [],
        },
    },
}

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute Python code in a sandbox and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to run",
                }
            },
            "required": ["code"],
        },
    },
}

TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Execute a terminal command and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to run",
                }
            },
            "required": ["command"],
        },
    },
}

ALL_TOOLS = [WEB_SEARCH_TOOL, PYTHON_TOOL, TERMINAL_TOOL]


_TIMEOUT_UNSET = object()


def execute_tool(
    name: str,
    arguments: dict,
    cancel_event = None,
    timeout: int | None = _TIMEOUT_UNSET,
    session_id: str | None = None,
) -> str:
    """Execute a tool by name with the given arguments. Returns result as a string.

    ``timeout``: int sets per-call limit in seconds, ``None`` means no limit,
    unset (default) uses ``_EXEC_TIMEOUT`` (300 s).
    ``session_id``: optional thread/session ID for per-conversation sandbox isolation.
    """
    logger.info(
        f"execute_tool: name={name}, session_id={session_id}, timeout={timeout}"
    )
    effective_timeout = _EXEC_TIMEOUT if timeout is _TIMEOUT_UNSET else timeout
    if name == "web_search":
        return _web_search(
            arguments.get("query", ""),
            url = arguments.get("url"),
            timeout = effective_timeout,
        )
    if name == "python":
        return _python_exec(
            arguments.get("code", ""), cancel_event, effective_timeout, session_id
        )
    if name == "terminal":
        return _bash_exec(
            arguments.get("command", ""), cancel_event, effective_timeout, session_id
        )
    return f"Unknown tool: {name}"


_MAX_PAGE_CHARS = 16000  # limit fetched page text (after HTML-to-MD conversion)
# Raw download cap.  Must be larger than _MAX_PAGE_CHARS because SSR pages
# embed large <head> sections (CSS, JS, SVGs) that are stripped during
# HTML-to-Markdown conversion.  512 KB is enough to reach article content
# on GitBook / Next.js / Docusaurus pages whose <head> alone can be 200 KB.
_MAX_FETCH_BYTES = 512 * 1024

_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
)

_tls_ctx = ssl.create_default_context()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection that connects to a pinned IP but uses a different
    hostname for SNI and certificate verification.

    The SSRF IP-pinning rewrites URLs to raw IPs.  A normal HTTPSConnection
    would then send no SNI and verify the cert against the IP, both of which
    fail.  This subclass splits the two concerns: TCP connects to the pinned
    IP (``host`` parameter) while TLS uses ``sni_hostname`` for the
    ClientHello and cert check.
    """

    def __init__(self, host: str, *, sni_hostname: str, **kwargs):
        super().__init__(host, **kwargs)
        self._sni_hostname = sni_hostname

    def connect(self):
        # TCP connect to the pinned IP stored in self.host (+ tunnel if
        # a proxy is configured via set_tunnel, though we do not use one).
        http.client.HTTPConnection.connect(self)
        # TLS handshake with the real hostname for SNI + cert verification.
        self.sock = self._context.wrap_socket(
            self.sock,
            server_hostname = self._sni_hostname,
        )


class _SNIHTTPSHandler(urllib.request.HTTPSHandler):
    """HTTPS handler that sends the correct SNI hostname during TLS handshake.

    The SSRF IP-pinning rewrites URLs to raw IPs, which breaks SNI and cert
    verification.  This handler returns a ``_PinnedHTTPSConnection`` that
    connects to the pinned IP but verifies TLS against the original hostname.
    """

    def __init__(self, hostname: str):
        super().__init__(context = _tls_ctx)
        self._sni_hostname = hostname

    def https_open(self, req):
        return self.do_open(self._sni_connection, req)

    def _sni_connection(self, host, **kwargs):
        kwargs["context"] = _tls_ctx
        return _PinnedHTTPSConnection(host, sni_hostname = self._sni_hostname, **kwargs)


def _validate_and_resolve_host(hostname: str, port: int) -> tuple[bool, str, str]:
    """Resolve *hostname*, reject non-public IPs, return a pinned IP string.

    Returns ``(ok, reason_or_empty, resolved_ip)``.  The caller should
    connect to *resolved_ip* (with a ``Host`` header) to prevent DNS
    rebinding between validation and the actual fetch.
    """
    import ipaddress
    import socket

    try:
        infos = socket.getaddrinfo(hostname, port, type = socket.SOCK_STREAM)
    except OSError as e:
        return False, f"Failed to resolve host: {e}", ""

    if not infos:
        return False, f"Failed to resolve host: no addresses for {hostname!r}", ""

    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False, f"Blocked: refusing to fetch non-public address {ip}.", ""

    # Return the first resolved address for pinning
    first_ip = infos[0][4][0]
    return True, "", first_ip


def _fetch_page_text(
    url: str, max_chars: int = _MAX_PAGE_CHARS, timeout: int = 30
) -> str:
    """Fetch a URL and return plain text content (HTML tags stripped).

    Blocks private/loopback/link-local targets (SSRF protection) and caps
    the download size to avoid unbounded memory usage.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Blocked: only http/https URLs are allowed (got {parsed.scheme!r})."
    if not parsed.hostname:
        return "Blocked: URL is missing a hostname."

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    ok, reason, pinned_ip = _validate_and_resolve_host(parsed.hostname, port)
    if not ok:
        return reason

    try:
        from urllib.error import HTTPError as _HTTPError
        from urllib.parse import urljoin, urlunparse

        max_bytes = _MAX_FETCH_BYTES
        current_url = url
        current_host = parsed.hostname
        ua = random.choice(_USER_AGENTS)

        for _hop in range(5):
            # Pin to the validated IP to prevent DNS rebinding.
            # Rewrite the URL to use the IP and set the Host header.
            cp = urlparse(current_url)
            # Bracket IPv6 addresses so the netloc is valid in a URL.
            ip_str = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
            ip_netloc = f"{ip_str}:{cp.port}" if cp.port else ip_str
            pinned_url = urlunparse(cp._replace(netloc = ip_netloc))

            opener = urllib.request.build_opener(
                _NoRedirect,
                _SNIHTTPSHandler(current_host),
            )

            req = urllib.request.Request(
                pinned_url,
                headers = {
                    "User-Agent": ua,
                    "Host": current_host,
                },
            )
            try:
                resp = opener.open(req, timeout = timeout)
            except _HTTPError as e:
                if e.code not in (301, 302, 303, 307, 308):
                    return (
                        f"Failed to fetch URL: HTTP {e.code} {getattr(e, 'reason', '')}"
                    )
                location = e.headers.get("Location")
                if not location:
                    return "Failed to fetch URL: redirect missing Location header."
                current_url = urljoin(current_url, location)
                rp = urlparse(current_url)
                if rp.scheme not in ("http", "https") or not rp.hostname:
                    return "Blocked: redirect target is not a valid http/https URL."
                rp_port = rp.port or (443 if rp.scheme == "https" else 80)
                ok2, reason2, pinned_ip = _validate_and_resolve_host(
                    rp.hostname,
                    rp_port,
                )
                if not ok2:
                    return reason2
                current_host = rp.hostname
                continue
            # Success -- read capped body
            raw_bytes = resp.read(max_bytes)
            break
        else:
            return "Failed to fetch URL: too many redirects."

        charset = resp.headers.get_content_charset() or "utf-8"
        raw_html = raw_bytes.decode(charset, errors = "replace")
    except _HTTPError as e:
        return f"Failed to fetch URL: HTTP {e.code} {getattr(e, 'reason', '')}"
    except Exception as e:
        return f"Failed to fetch URL: {e}"

    # Convert HTML to Markdown using the builtin converter (no external deps)
    from ._html_to_md import html_to_markdown

    text = html_to_markdown(raw_html)

    if not text:
        return "(page returned no readable text)"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... (truncated, {len(text)} chars total)"
    return text


def _web_search(
    query: str,
    max_results: int = 5,
    timeout: int = _EXEC_TIMEOUT,
    url: str | None = None,
) -> str:
    """Search the web using DuckDuckGo and return formatted results.

    If ``url`` is provided, fetches that page directly instead of searching.
    """
    # Direct URL fetch mode
    if url and url.strip():
        fetch_timeout = 60 if timeout is None else min(timeout, 60)
        return _fetch_page_text(url.strip(), timeout = fetch_timeout)

    if not query or not query.strip():
        return "No query provided."
    try:
        from ddgs import DDGS

        results = DDGS(timeout = timeout).text(query, max_results = max_results)
        if not results:
            return "No results found."
        parts = []
        for r in results:
            parts.append(
                f"Title: {r.get('title', '')}\n"
                f"URL: {r.get('href', '')}\n"
                f"Snippet: {r.get('body', '')}"
            )
        text = "\n\n---\n\n".join(parts)
        text += (
            "\n\n---\n\nIMPORTANT: These are only short snippets. "
            "To get the full page content, call web_search with "
            'the url parameter (e.g. {"url": "<URL>"}).'
        )
        return text
    except Exception as e:
        return f"Search failed: {e}"


def _check_signal_escape_patterns(code: str):
    """
    Check if code contains patterns that could escape signal-based timeouts.

    Vendored from unsloth_zoo.rl_environments to avoid importing unsloth_zoo
    (which requires GPU drivers and fails on Mac/Apple Silicon).

    Returns (safe: bool, details: dict)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, {
            "error": f"SyntaxError: {e}",
            "signal_tampering": [],
            "exception_catching": [],
            "warnings": [],
        }

    signal_tampering = []
    exception_catching = []
    shell_escapes = []
    warnings = []

    def _ast_name_matches(node, names):
        if isinstance(node, ast.Name):
            return node.id in names
        elif isinstance(node, ast.Attribute):
            full_name = []
            current = node
            while isinstance(current, ast.Attribute):
                full_name.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                full_name.append(current.id)
            full_name = ".".join(reversed(full_name))
            return full_name in names
        return False

    # Dangerous os/subprocess functions that can execute shell commands
    _SHELL_EXEC_FUNCS = frozenset(
        {
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
            "subprocess.run",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "subprocess.Popen",
            "subprocess.getoutput",
            "subprocess.getstatusoutput",
        }
    )

    def _extract_string_from_node(node):
        """Extract a plain string value from an AST node, if it is a constant."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _extract_strings_from_list(node):
        """Extract string elements from an AST List or Tuple node."""
        if isinstance(node, (ast.List, ast.Tuple)):
            parts = []
            for elt in node.elts:
                s = _extract_string_from_node(elt)
                if s is not None:
                    parts.append(s)
            return parts
        return []

    # Keyword argument names that carry command content (as opposed to
    # control flags like check=True, text=True, capture_output=True).
    _CMD_KWARGS = frozenset({"args", "command", "executable", "path", "file"})

    def _check_args_for_blocked(args_nodes):
        """Check if any call arguments contain blocked commands."""
        found = set()
        for arg in args_nodes:
            s = _extract_string_from_node(arg)
            if s is not None:
                found |= _find_blocked_commands(s)
            strs = _extract_strings_from_list(arg)
            for s in strs:
                found |= _find_blocked_commands(s)
        return found

    class SignalEscapeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.imports_signal = False
            self.signal_aliases = {"signal"}
            self.os_aliases = {"os"}
            self.subprocess_aliases = {"subprocess"}
            # Maps bare function names to their fully-qualified form
            # for from-import tracking (e.g. "system" -> "os.system")
            self.shell_exec_aliases: dict[str, str] = {}
            self.loop_depth = 0

        def visit_Import(self, node):
            for alias in node.names:
                if alias.name == "signal":
                    self.imports_signal = True
                    if alias.asname:
                        self.signal_aliases.add(alias.asname)
                elif alias.name == "os":
                    self.os_aliases.add(alias.asname or "os")
                elif alias.name == "subprocess":
                    self.subprocess_aliases.add(alias.asname or "subprocess")
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module == "signal":
                self.imports_signal = True
                for alias in node.names:
                    if alias.name in (
                        "signal",
                        "SIGALRM",
                        "SIG_IGN",
                        "setitimer",
                        "ITIMER_REAL",
                        "pthread_sigmask",
                        "SIG_BLOCK",
                        "alarm",
                    ):
                        self.signal_aliases.add(alias.asname or alias.name)
            elif node.module in ("os", "subprocess"):
                if node.module == "os":
                    self.os_aliases.add("os")
                else:
                    self.subprocess_aliases.add("subprocess")
                # Track from-imports of dangerous functions
                for alias in node.names:
                    fq = f"{node.module}.{alias.name}"
                    if fq in _SHELL_EXEC_FUNCS:
                        self.shell_exec_aliases[alias.asname or alias.name] = fq
            self.generic_visit(node)

        def visit_While(self, node):
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_For(self, node):
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_Call(self, node):
            func = node.func
            func_name = None
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name):
                    if func.value.id in self.signal_aliases:
                        func_name = f"signal.{func.attr}"
            elif isinstance(func, ast.Name):
                if func.id in ("signal", "setitimer", "alarm", "pthread_sigmask"):
                    func_name = func.id

            if func_name:
                if func_name in ("signal.signal", "signal"):
                    if len(node.args) >= 1:
                        if _ast_name_matches(
                            node.args[0], ("SIGALRM", "signal.SIGALRM")
                        ):
                            signal_tampering.append(
                                {
                                    "type": "signal_handler_override",
                                    "line": node.lineno,
                                    "description": "Overrides SIGALRM handler",
                                }
                            )
                elif func_name in ("signal.setitimer", "setitimer"):
                    if len(node.args) >= 1:
                        if _ast_name_matches(
                            node.args[0], ("ITIMER_REAL", "signal.ITIMER_REAL")
                        ):
                            signal_tampering.append(
                                {
                                    "type": "timer_manipulation",
                                    "line": node.lineno,
                                    "description": "Manipulates ITIMER_REAL timer",
                                }
                            )
                elif func_name in ("signal.alarm", "alarm"):
                    signal_tampering.append(
                        {
                            "type": "alarm_manipulation",
                            "line": node.lineno,
                            "description": "Manipulates alarm timer",
                        }
                    )
                elif func_name in ("signal.pthread_sigmask", "pthread_sigmask"):
                    signal_tampering.append(
                        {
                            "type": "signal_mask",
                            "line": node.lineno,
                            "description": "Modifies signal mask (may block SIGALRM)",
                        }
                    )

            # --- Shell escape detection ---
            # Resolve the fully qualified function name for os.*/subprocess.*
            shell_func = None
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name):
                    if func.value.id in self.os_aliases:
                        shell_func = f"os.{func.attr}"
                    elif func.value.id in self.subprocess_aliases:
                        shell_func = f"subprocess.{func.attr}"
            elif isinstance(func, ast.Name):
                # Check from-import aliases: from os import system; system(...)
                shell_func = self.shell_exec_aliases.get(func.id)

            if shell_func and shell_func in _SHELL_EXEC_FUNCS:
                # Expand **kwargs dicts to inspect their keys
                expanded_kwargs: dict[str, ast.AST] = {}
                has_opaque_kwargs = False
                for kw in node.keywords:
                    if kw.arg is not None:
                        expanded_kwargs[kw.arg] = kw.value
                    elif isinstance(kw.value, ast.Dict):
                        for k, v in zip(kw.value.keys, kw.value.values):
                            key = _extract_string_from_node(k) if k else None
                            if key is not None:
                                expanded_kwargs[key] = v
                    else:
                        has_opaque_kwargs = True

                cmd_kw_values = [
                    v for k, v in expanded_kwargs.items() if k in _CMD_KWARGS
                ]
                all_call_args = list(node.args) + cmd_kw_values
                blocked_in_args = _check_args_for_blocked(all_call_args)

                if has_opaque_kwargs:
                    # Can't inspect dynamic **kwargs -- flag as unsafe
                    shell_escapes.append(
                        {
                            "type": "shell_escape_dynamic",
                            "line": node.lineno,
                            "description": (
                                f"{shell_func}() called with dynamic **kwargs"
                            ),
                        }
                    )
                elif blocked_in_args:
                    shell_escapes.append(
                        {
                            "type": "shell_escape",
                            "line": node.lineno,
                            "description": (
                                f"{shell_func}() invokes blocked command(s): "
                                f"{', '.join(sorted(blocked_in_args))}"
                            ),
                        }
                    )
                else:
                    # Only flag dynamic args for functions that interpret
                    # strings as shell commands, or when shell= might be
                    # enabled.  Treat any non-literal-False shell= value
                    # as potentially True (conservative).
                    _STRING_SHELL_FUNCS = frozenset(
                        {
                            "os.system",
                            "os.popen",
                            "os.popen2",
                            "os.popen3",
                            "os.popen4",
                            "subprocess.getoutput",
                            "subprocess.getstatusoutput",
                        }
                    )
                    shell_node = expanded_kwargs.get("shell")
                    shell_safe = shell_node is None or (
                        isinstance(shell_node, ast.Constant)
                        and shell_node.value is False
                    )
                    if shell_func in _STRING_SHELL_FUNCS or not shell_safe:

                        def _is_safe_literal(n):
                            if _extract_string_from_node(n) is not None:
                                return True
                            if isinstance(n, (ast.List, ast.Tuple)):
                                return all(
                                    _extract_string_from_node(e) is not None
                                    for e in n.elts
                                )
                            return False

                        has_non_literal = any(
                            not _is_safe_literal(a) for a in all_call_args
                        )
                        if has_non_literal:
                            shell_escapes.append(
                                {
                                    "type": "shell_escape_dynamic",
                                    "line": node.lineno,
                                    "description": (
                                        f"{shell_func}() called with non-literal "
                                        f"shell command (potential shell escape)"
                                    ),
                                }
                            )

            self.generic_visit(node)

        def visit_ExceptHandler(self, node):
            if self.loop_depth == 0:
                self.generic_visit(node)
                return
            if node.type is None:
                exception_catching.append(
                    {
                        "type": "bare_except_in_loop",
                        "line": node.lineno,
                        "description": "Bare except in loop catches TimeoutError and continues looping",
                    }
                )
            elif isinstance(node.type, ast.Name):
                # Only flag BaseException and TimeoutError, NOT Exception.
                # except Exception does not catch SystemExit or
                # KeyboardInterrupt, so it cannot suppress timeout
                # enforcement.  Flagging Exception causes false positives
                # on normal error-handling patterns.
                if node.type.id in ("TimeoutError", "BaseException"):
                    exception_catching.append(
                        {
                            "type": f"catches_{node.type.id}_in_loop",
                            "line": node.lineno,
                            "description": f"Catches {node.type.id} in loop - may suppress timeout and continue",
                        }
                    )
            elif isinstance(node.type, ast.Tuple):
                for elt in node.type.elts:
                    if isinstance(elt, ast.Name):
                        if elt.id in ("TimeoutError", "BaseException"):
                            exception_catching.append(
                                {
                                    "type": f"catches_{elt.id}_in_loop",
                                    "line": node.lineno,
                                    "description": f"Catches {elt.id} in loop - may suppress timeout and continue",
                                }
                            )
            self.generic_visit(node)

    visitor = SignalEscapeVisitor()
    visitor.visit(tree)

    if visitor.imports_signal and not signal_tampering:
        warnings.append("Code imports 'signal' module - review manually for safety")

    is_safe = (
        len(signal_tampering) == 0
        and len(exception_catching) == 0
        and len(shell_escapes) == 0
    )
    return is_safe, {
        "signal_tampering": signal_tampering,
        "exception_catching": exception_catching,
        "shell_escapes": shell_escapes,
        "warnings": warnings,
    }


def _check_code_safety(code: str) -> str | None:
    """Validate code safety via static analysis.

    Returns an error message string if the code is unsafe, or None if OK.
    """
    safe, info = _check_signal_escape_patterns(code)
    if not safe:
        # SyntaxError from ast.parse -- let these through so the subprocess
        # produces a normal Python traceback instead of a misleading
        # "unsafe code detected" message.
        if info.get("error"):
            return None

        reasons = [
            item.get("description", "") for item in info.get("signal_tampering", [])
        ]
        shell_reasons = [
            item.get("description", "") for item in info.get("shell_escapes", [])
        ]
        exception_reasons = [
            item.get("description", "") for item in info.get("exception_catching", [])
        ]
        all_reasons = [r for r in reasons + shell_reasons + exception_reasons if r]
        if all_reasons:
            return (
                f"Error: unsafe code detected ({'; '.join(all_reasons)}). "
                f"Please remove unsafe patterns from your code."
            )

    return None


def _cancel_watcher(proc, cancel_event, poll_interval = 0.2):
    """Daemon thread that kills a process when cancel_event is set."""
    while proc.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            proc.kill()
            return
        cancel_event.wait(poll_interval) if cancel_event else None


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n\n... (truncated, {len(text)} chars total)"
    return text


def _python_exec(
    code: str,
    cancel_event = None,
    timeout: int = _EXEC_TIMEOUT,
    session_id: str | None = None,
) -> str:
    """Execute Python code in a subprocess sandbox."""
    if not code or not code.strip():
        return "No code provided."

    # Validate imports and code safety
    error = _check_code_safety(code)
    if error:
        return error

    tmp_path = None
    workdir = _get_workdir(session_id)
    # Snapshot image mtimes so we detect both new and overwritten files.
    _before: dict[str, int] = {}
    if os.path.isdir(workdir):
        for _name in os.listdir(workdir):
            if os.path.splitext(_name)[1].lower() in _IMAGE_EXTS:
                _p = os.path.join(workdir, _name)
                if os.path.isfile(_p):
                    try:
                        _before[_name] = os.stat(_p).st_mtime_ns
                    except OSError:
                        pass
    try:
        fd, tmp_path = tempfile.mkstemp(
            suffix = ".py", prefix = "studio_exec_", dir = workdir
        )
        with os.fdopen(fd, "w") as f:
            f.write(code)

        safe_env = _build_safe_env(workdir)
        popen_kwargs = dict(
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            cwd = workdir,
            env = safe_env,
        )
        if sys.platform != "win32":
            popen_kwargs["preexec_fn"] = _sandbox_preexec
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        proc = subprocess.Popen([sys.executable, tmp_path], **popen_kwargs)

        # Spawn cancel watcher if we have a cancel event
        if cancel_event is not None:
            watcher = threading.Thread(
                target = _cancel_watcher, args = (proc, cancel_event), daemon = True
            )
            watcher.start()

        try:
            output, _ = proc.communicate(timeout = timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return _truncate(f"Execution timed out after {timeout} seconds.")

        if cancel_event is not None and cancel_event.is_set():
            return "Execution cancelled."

        result = output or ""
        if proc.returncode != 0:
            result = f"Exit code {proc.returncode}:\n{result}"
        result = _truncate(result) if result.strip() else "(no output)"

        # Detect new or overwritten image files and append sentinel for frontend
        if session_id and os.path.isdir(workdir):
            new_images = []
            for _name in os.listdir(workdir):
                if os.path.splitext(_name)[1].lower() not in _IMAGE_EXTS:
                    continue
                _p = os.path.join(workdir, _name)
                if not os.path.isfile(_p):
                    continue
                try:
                    _mtime = os.stat(_p).st_mtime_ns
                except OSError:
                    continue
                if _name not in _before or _mtime != _before[_name]:
                    new_images.append(_name)
            if new_images:
                import json as _json

                result += f"\n__IMAGES__:{_json.dumps(sorted(new_images))}"

        return result

    except Exception as e:
        return f"Execution error: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _bash_exec(
    command: str,
    cancel_event = None,
    timeout: int = _EXEC_TIMEOUT,
    session_id: str | None = None,
) -> str:
    """Execute a bash command in a subprocess sandbox."""
    if not command or not command.strip():
        return "No command provided."

    # Block dangerous commands (shlex + regex based)
    blocked = _find_blocked_commands(command)
    if blocked:
        return f"Blocked command(s) for safety: {', '.join(sorted(blocked))}"

    try:
        workdir = _get_workdir(session_id)
        safe_env = _build_safe_env(workdir)
        popen_kwargs = dict(
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            cwd = workdir,
            env = safe_env,
        )
        if sys.platform != "win32":
            popen_kwargs["preexec_fn"] = _sandbox_preexec
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        proc = subprocess.Popen(_get_shell_cmd(command), **popen_kwargs)

        if cancel_event is not None:
            watcher = threading.Thread(
                target = _cancel_watcher, args = (proc, cancel_event), daemon = True
            )
            watcher.start()

        try:
            output, _ = proc.communicate(timeout = timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return _truncate(f"Execution timed out after {timeout} seconds.")

        if cancel_event is not None and cancel_event.is_set():
            return "Execution cancelled."

        result = output or ""
        if proc.returncode != 0:
            result = f"Exit code {proc.returncode}:\n{result}"
        return _truncate(result) if result.strip() else "(no output)"

    except Exception as e:
        return f"Execution error: {e}"
