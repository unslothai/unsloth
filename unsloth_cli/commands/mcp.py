# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth mcp add <agent>` and the checks behind `unsloth doctor --mcp`.

Agent configs only name the environment variable that holds the API key. `mcp add`
never reads the key; doctor sends it to the server to test it and never prints it.
"""

import copy
import difflib
import json
import os
import re
import shlex
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import typer

from unsloth_cli._inference import (
    _USER_AGENT,
    find_studio_server,
    is_loopback_url,
    urlopen_no_redirect,
)
from unsloth_cli.commands.start import (
    _codex_source_home,
    _fail,
    _load_app_config,
    _read_app_file,
    _which_with_install_dirs,
    _write_app_file,
)

mcp_app = typer.Typer(
    help = "Connect coding agents to the Unsloth Studio MCP server.",
    no_args_is_help = True,
    context_settings = {"help_option_names": ["-h", "--help"]},
)

_DEFAULT_BASE = "http://127.0.0.1:8888"
_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_TABLE_LINE_RE = re.compile(r"^[ \t]*\[\[?[^\[\]\n]+\]\]?[ \t]*(#[^\n]*)?\r?\n?$")
# The longest an Unsloth Studio MCP tool waits before answering (the server's wait cap).
# The backend can't be imported from the CLI, so it is repeated here.
_MCP_WAIT_CAP_S = 240
_CODEX_DEFAULT_TIMEOUT_S = 60
_CLAUDE_ADD = ["mcp", "add", "--transport", "http", "--scope", "user"]
_INITIALIZE = (
    b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18",'
    b'"capabilities":{},"clientInfo":{"name":"unsloth-doctor","version":"1"}}}'
)


class Agent(str, Enum):
    claude_code = "claude-code"
    codex = "codex"
    cursor = "cursor"
    generic = "generic"


def mcp_url(base: str) -> str:
    base = base.strip().rstrip("/")
    if base.endswith("/mcp"):
        base = base[: -len("/mcp")]
    return base + "/mcp/"


def _bearer(env_var: str, style: str = "${%s}") -> str:
    return "Bearer " + style % env_var


def claude_entry(url: str, env_var: str) -> dict:
    return {"type": "http", "url": url, "headers": {"Authorization": _bearer(env_var)}}


def _with_server(existing: dict, name: str, entry: dict) -> dict:
    data = copy.deepcopy(existing)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = data["mcpServers"] = {}
    servers[name] = entry
    return data


def cursor_merge(existing: dict, name: str, url: str, env_var: str) -> dict:
    entry = {"url": url, "headers": {"Authorization": _bearer(env_var, "${env:%s}")}}
    return _with_server(existing, name, entry)


def generic_snippet(name: str, url: str, env_var: str) -> dict:
    return {"mcpServers": {name: claude_entry(url, env_var)}}


def _codex_header(name: str):
    n = re.escape(name)
    return re.compile(rf"""^\s*\[\s*mcp_servers\s*\.\s*(?:"{n}"|'{n}'|{n})\s*[\].]""")


def _codex_block(name: str, url: str, env_var: str, timeout: int) -> str:
    return (
        f"[mcp_servers.{name}]\n"
        f"url = {json.dumps(url)}\n"
        f"bearer_token_env_var = {json.dumps(env_var)}\n"
        f"tool_timeout_sec = {int(timeout)}\n"
    )


def _comment_tail(chunk: str) -> str:
    # Blank and comment lines at the end of a table usually describe the next one: keep them.
    lines = chunk.splitlines(True)
    end = len(lines)
    while end and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
        end -= 1
    return "".join(lines[end:])


def codex_merge(existing_text: str, name: str, url: str, env_var: str, timeout: int) -> str:
    """Replace the [mcp_servers.NAME] table (and its subtables) in place, or append it."""
    header = _codex_header(name)
    block = _codex_block(name, url, env_var, timeout)
    out, placed = [], False
    for chunk in re.split(r"(?m)^(?=[ \t]*\[)", existing_text):
        if not header.match(chunk):
            out.append(chunk)
        elif not placed:
            out.append(block + _comment_tail(chunk))
            placed = True
        else:
            tail = _comment_tail(chunk)
            out.append(tail if "#" in tail else "")
    text = "".join(out)
    if not placed:
        if text.strip():
            text = text.rstrip("\n") + "\n\n"
        else:
            text = ""
        text += block
    return text


def _toml_loads():
    try:
        import tomllib  # novermin
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return None
    return tomllib.loads


def codex_check(old_text: str, new_text: str, name: str, expected: dict) -> None:
    """Raise ValueError unless the splice changed exactly mcp_servers.NAME."""
    loads = _toml_loads()
    if loads is None:
        # No TOML parser (Python < 3.11 without tomli): refuse any other mention of NAME, and
        # anything that makes the line-based splice unsafe: a multi-line string can hold a
        # line that looks like a table header, and a line opening with "[" can be array data.
        block = _codex_block(name, *expected.values())
        if name in new_text.replace(block, ""):
            raise ValueError(f"{name} is also defined in another form")
        if '"""' in old_text or "'''" in old_text:
            raise ValueError("it has multi-line strings and no TOML parser is installed")
        for chunk in re.split(r"(?m)^(?=[ \t]*\[)", old_text)[1:]:
            if not _TABLE_LINE_RE.match(chunk.splitlines(True)[0]):
                raise ValueError("a line starting with '[' is not a table header")
        return
    try:
        old, new = loads(old_text), loads(new_text)
    except Exception as e:
        raise ValueError(f"not valid TOML after the edit ({e})")
    if (new.get("mcp_servers") or {}).get(name) != expected:
        raise ValueError(f"mcp_servers.{name} did not come out as expected")

    def without(doc):
        doc = dict(doc)
        servers = doc.get("mcp_servers")
        if isinstance(servers, dict):
            servers = {k: v for k, v in servers.items() if k != name}
            if servers:
                doc["mcp_servers"] = servers
            else:
                doc.pop("mcp_servers")
        return doc

    if without(old) != without(new):
        raise ValueError("other settings would change")


def _redact(text: str) -> str:
    # Last line of defence for text we don't parse, such as claude's own error output.
    text = re.sub(r"(?i)(bearer\s+)(?!\$\{)\S+", r"\1***", text)
    return re.sub(r"sk-unsloth-\S+", "sk-unsloth-***", text)


def _diff(path: Path, old_text: Optional[str], new_text: str) -> str:
    return "".join(
        difflib.unified_diff(
            (old_text or "").splitlines(True),
            new_text.splitlines(True),
            fromfile = str(path) if old_text is not None else "/dev/null",
            tofile = str(path),
        )
    )


def _show_entry(path: Path, name: str, entry_text: str, replaces: bool) -> None:
    # Only the new entry, which names an env var: an old one may hold a key pasted in by hand.
    typer.echo(_diff(path, None, entry_text), nl = False)
    if replaces:
        typer.echo(f"This replaces the current {name} entry; its old values are not shown.")


def write_config(
    path: Path,
    old_text: Optional[str],
    new_text: str,
    *,
    dry_run: bool,
    shown: Tuple[str, str, bool],
) -> str:
    """Write ``new_text``; a dry run prints only ``shown``: (name, new entry text, replaces)."""
    if old_text == new_text:
        return "unchanged"
    if dry_run:
        _show_entry(path, *shown)
        return "dry-run"
    mode = None if old_text is None else path.stat().st_mode & 0o777
    _write_app_file(path, new_text)
    if mode is not None and os.name != "nt":
        os.chmod(path, mode)
    return "written"


def _read_json_config(path: Path) -> Tuple[Optional[str], dict]:
    text = _read_app_file(path)
    try:
        data = _load_app_config(path, text)
    except Exception:
        data = None
    if not isinstance(data, dict):
        _fail(
            f"Couldn't parse {path} as a JSON object, so it was left as is. "
            "Fix it and re-run, or add the entry by hand (see `unsloth mcp add generic`)."
        )
    if not isinstance(data.get("mcpServers", {}), dict):
        _fail(f'"mcpServers" in {path} is not an object, so the file was left as is.')
    return text, data


def _json_text(data: dict) -> str:
    return json.dumps(data, indent = 2, ensure_ascii = False) + "\n"


def _merge_json_file(
    path: Path, name: str, data: dict, text: Optional[str], new: dict, dry_run: bool
) -> str:
    if text is not None and new == data:
        return "unchanged"
    replaces = name in data.get("mcpServers", {})
    shown = (name, _json_text({name: new["mcpServers"][name]}), replaces)
    return write_config(path, text, _json_text(new), dry_run = dry_run, shown = shown)


def claude_config_path() -> Path:
    configured = os.environ.get("CLAUDE_CONFIG_DIR")
    return (
        Path(configured).expanduser() / ".claude.json"
        if configured
        else Path.home() / ".claude.json"
    )


def _add_claude(name, url, env_var, dry_run) -> Tuple[str, Path]:
    path = claude_config_path()
    text, data = _read_json_config(path)
    entry = claude_entry(url, env_var)
    servers = data.get("mcpServers") if isinstance(data.get("mcpServers"), dict) else {}
    current = servers.get(name)
    if current == entry:
        return "unchanged", path
    exe = _which_with_install_dirs("claude")
    if exe is None:
        typer.echo("Claude Code was not found on PATH, so its config file is edited directly.")
        new = _with_server(data, name, entry)
        return _merge_json_file(path, name, data, text, new, dry_run), path
    commands = []
    if current is not None:
        commands.append([exe, "mcp", "remove", "--scope", "user", name])
    commands.append(
        [exe, *_CLAUDE_ADD, name, url, "--header", f"Authorization: {_bearer(env_var)}"]
    )
    if dry_run:
        _show_entry(path, name, _json_text({name: entry}), current is not None)
        for argv in commands:
            typer.echo("Would run: " + shlex.join(argv))
        return "dry-run", path
    for argv in commands:
        removed = current is not None and argv is commands[-1]
        _run_claude(
            argv, removed and f"The old {name} entry was already removed; re-run this command."
        )
    if _read_json_config(path)[1].get("mcpServers", {}).get(name) != entry:
        typer.echo(f"Check the result with `claude mcp get {name}`: it is not what was asked.")
    return "written", path


def _run_claude(argv: List[str], after: Optional[str] = None) -> None:
    run = dict(stdin = subprocess.DEVNULL, capture_output = True, text = True, timeout = 60)
    note = f"\n{after}" if after else ""
    try:
        done = subprocess.run(argv, **run)
    except (OSError, subprocess.TimeoutExpired) as e:
        _fail(f"`claude mcp {argv[2]}` failed: {e}{note}")
    if done.returncode != 0:
        output = _redact((done.stderr or done.stdout).strip())
        _fail(f"`claude mcp {argv[2]}` failed:\n{output}{note}")


def _add_codex(name, url, env_var, timeout, dry_run) -> Tuple[str, Path]:
    path = _codex_source_home() / "config.toml"
    text = _read_app_file(path)
    old = text or ""
    new = codex_merge(old, name, url, env_var, timeout)
    expected = {"url": url, "bearer_token_env_var": env_var, "tool_timeout_sec": int(timeout)}
    try:
        codex_check(old, new, name, expected)
    except ValueError as e:
        _fail(
            f"Couldn't update {path} safely ({e}), so it was left as is. Add this table by hand:\n\n"
            + _codex_block(name, url, env_var, timeout)
        )
    chunks = re.split(r"(?m)^(?=[ \t]*\[)", old)
    replaces = any(_codex_header(name).match(chunk) for chunk in chunks)
    shown = (name, _codex_block(name, url, env_var, timeout), replaces)
    return write_config(path, text, new, dry_run = dry_run, shown = shown), path


def _add_json(path: Path, name: str, merge, dry_run) -> Tuple[str, Path]:
    text, data = _read_json_config(path)
    return _merge_json_file(path, name, data, text, merge(data), dry_run), path


def _resolve_base(url: Optional[str]) -> str:
    return url or os.environ.get("UNSLOTH_STUDIO_URL") or find_studio_server() or _DEFAULT_BASE


@mcp_app.command("add")
def add(
    agent: Agent = typer.Argument(..., help = "The agent to set up."),
    url: Optional[str] = typer.Option(None, "--url", help = "Unsloth Studio address."),
    env_var: str = typer.Option("UNSLOTH_API_KEY", "--env-var", help = "Env var with the key."),
    name: str = typer.Option("unsloth-studio", "--name", help = "Server name in the config."),
    timeout: int = typer.Option(300, "--timeout", min = 240, help = "Codex tool timeout (s)."),
    output: Optional[Path] = typer.Option(None, "--output", help = "generic: merge into this file."),
    dry_run: bool = typer.Option(False, "--dry-run", help = "Show the change; write nothing."),
) -> None:
    """Add the Unsloth Studio MCP server to an agent's config. The API key stays in an env var."""
    if not _ENV_VAR_RE.match(env_var):
        _fail(f"--env-var {env_var!r} is not a valid environment variable name.")
    if not _NAME_RE.match(name):
        _fail("--name may only use letters, digits, '-' and '_'.")
    base = _resolve_base(url)
    parsed = urlparse(base.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        _fail(f"{base!r} is not an http(s) URL.")
    target = mcp_url(base)

    if agent is Agent.generic and output is None:
        typer.echo(_json_text(generic_snippet(name, target, env_var)), nl = False)
        typer.echo(f"Replace ${{{env_var}}} with your agent's syntax for environment variables.")
        state = None
    elif agent is Agent.claude_code:
        state, path = _add_claude(name, target, env_var, dry_run)
    elif agent is Agent.codex:
        state, path = _add_codex(name, target, env_var, timeout, dry_run)
    elif agent is Agent.cursor:
        path = Path.home() / ".cursor" / "mcp.json"
        state, _ = _add_json(path, name, lambda d: cursor_merge(d, name, target, env_var), dry_run)
    else:
        if output.suffix.lower() not in (".json", ".jsonc"):
            _fail("--output must be a .json file; for another format, paste the printed entry.")
        entry = claude_entry(target, env_var)
        state, _ = _add_json(output, name, lambda d: _with_server(d, name, entry), dry_run)
        path = output

    if state == "written":
        typer.echo(f"Unsloth Studio MCP set up for {agent.value} in {path}")
    elif state == "unchanged":
        typer.echo(f"Unsloth Studio MCP for {agent.value} is already set up in {path} (unchanged).")
    elif state == "dry-run":
        typer.echo("Dry run: nothing was written.")
    typer.echo(f"Set {env_var} to an Unsloth Studio API key (Settings → API) where it starts.")
    if not os.environ.get(env_var):
        typer.echo(f"{env_var} is not set in this shell.")
    if agent is Agent.claude_code:
        typer.echo("For long jobs, also set MCP_TOOL_TIMEOUT=300000 where Claude Code starts.")


# --- unsloth doctor --mcp ---


@dataclass
class McpFacts:
    env_var: str = "UNSLOTH_API_KEY"
    base: Optional[str] = None
    key: str = "missing"  # missing | unsent | valid | invalid | error
    key_detail: str = ""
    mcp: str = "unknown"  # on | off | absent | refused | error | unknown
    mcp_detail: str = ""
    timeouts: List[Tuple[str, float, str]] = field(default_factory = list)  # (label, s, fix)


def _probe(url: str, key: Optional[str], timeout: float, data: Optional[bytes]):
    """(status, detail); status is None when nothing answered. POSTs when data is given."""
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        with urlopen_no_redirect(urllib.request.Request(url, data, headers), timeout) as response:
            return response.status, ""
    except urllib.error.HTTPError as e:
        try:
            detail = str(json.loads(e.read(65536) or b"{}").get("detail") or e.reason)
        except Exception:
            detail = str(e.reason)
        return e.code, _scrub(detail, key)
    except Exception as e:
        # http.client quotes a header value it refuses (a key with a newline) in its error.
        return None, _scrub(str(getattr(e, "reason", None) or e), key)


def _scrub(detail: str, key: Optional[str]) -> str:
    for secret in (key, (key or "").strip(), repr(key or "")[1:-1]):
        if secret:
            detail = detail.replace(secret, "***")
    return _redact(detail)


def _ours(name: str, key: str, entry, urls: Tuple[str, ...]) -> bool:
    if not isinstance(entry, dict):
        return False
    url = str(entry.get("url", ""))
    return key == name or (url.startswith(("http://", "https://")) and mcp_url(url) in urls)


def _codex_timeouts(name: str, urls: Tuple[str, ...] = ()) -> List[Tuple[str, float, str]]:
    loads = _toml_loads()
    try:
        text = (_codex_source_home() / "config.toml").read_text(encoding = "utf-8")
        servers = loads(text).get("mcp_servers") or {}
    except (Exception, typer.Exit):
        return []
    fix = "Run `unsloth mcp add codex` to set tool_timeout_sec = 300."
    found = [
        (key, entry.get("tool_timeout_sec", _CODEX_DEFAULT_TIMEOUT_S))
        for key, entry in servers.items()
        if _ours(name, key, entry, urls)
    ]
    return [(f"Codex {key}", s, fix) for key, s in found if isinstance(s, (int, float))]


def _claude_timeouts(name: str, urls: Tuple[str, ...] = ()) -> List[Tuple[str, float, str]]:
    root = os.environ.get("CLAUDE_CONFIG_DIR")
    settings = (Path(root).expanduser() if root else Path.home() / ".claude") / "settings.json"
    try:
        servers = json.loads(claude_config_path().read_text(encoding = "utf-8"))["mcpServers"]
        if not any(_ours(name, k, v, urls) for k, v in servers.items()):
            return []
        value = os.environ.get("MCP_TOOL_TIMEOUT")
        if value is None:
            value = json.loads(settings.read_text(encoding = "utf-8"))["env"]["MCP_TOOL_TIMEOUT"]
        return [("Claude Code", float(value) / 1000, "Set MCP_TOOL_TIMEOUT=300000.")]
    except Exception:
        return []


def gather_mcp_facts(env_var: str = "UNSLOTH_API_KEY", timeout: float = 5.0) -> McpFacts:
    facts = McpFacts(env_var = env_var)
    facts.base = find_studio_server(timeout = min(timeout, 3.0))
    urls = (mcp_url(facts.base),) if facts.base else ()
    facts.timeouts = _codex_timeouts("unsloth-studio", urls) + _claude_timeouts(
        "unsloth-studio", urls
    )
    if facts.base is None:
        return facts
    key = os.environ.get(env_var) or None
    # Like the other auto flows, the key only goes to loopback or an explicit UNSLOTH_STUDIO_URL.
    if key and not (is_loopback_url(facts.base) or os.environ.get("UNSLOTH_STUDIO_URL")):
        facts.key, key = "unsent", None
    elif key:
        status, detail = _probe(f"{facts.base}/api/inference/loaded-models", key, timeout, None)
        if status is not None and 200 <= status < 300:
            facts.key = "valid"
        else:
            facts.key = "invalid" if status in (401, 403) else "error"
            facts.key_detail, key = f"{status or ''} {detail}".strip(), None
    status, detail = _probe(f"{facts.base}/mcp/", key, timeout, _INITIALIZE)
    if status == 404:
        facts.mcp = "off"
    elif status == 405:  # an Unsloth Studio from before MCP: its page catch-all takes GET only
        facts.mcp = "absent"
    elif status == 401:
        facts.mcp, facts.mcp_detail = "refused", detail
    elif status is not None and 200 <= status < 300:
        facts.mcp = "on"
    else:
        facts.mcp, facts.mcp_detail = "error", f"{status or ''} {detail}".strip()
    return facts


def mcp_verdict(facts: McpFacts) -> Tuple[bool, List[str], List[str]]:
    """(healthy, hints, advice). Advice never makes the verdict fail."""
    var, hints = facts.env_var, []
    key_hints = {
        "missing": f"Set {var} to an Unsloth Studio API key (Settings → API).",
        "unsent": f"{facts.base} is not on this machine; set UNSLOTH_STUDIO_URL to test the key.",
        "invalid": f"Unsloth Studio refused the key in {var}. Create one in Settings → API.",
        "error": f"Unsloth Studio failed while checking the key: {facts.key_detail}",
    }
    mcp_hints = {
        "off": "Turn on Settings → API → Allow agent connections, or start Unsloth Studio "
        "with UNSLOTH_STUDIO_ENABLE_MCP=1.",
        "absent": "This Unsloth Studio has no MCP server. Update it: unsloth studio update",
        "refused": f"The MCP server refused the key: {facts.mcp_detail}",
        "error": f"The MCP server did not answer properly: {facts.mcp_detail}",
    }
    if facts.base is None:
        where = os.environ.get("UNSLOTH_STUDIO_URL") or _DEFAULT_BASE
        hints.append(
            f"Unsloth Studio is not reachable at {where}. Start it, or set UNSLOTH_STUDIO_URL."
        )
    else:
        hints += [key_hints[facts.key]] if facts.key in key_hints else []
        # A missing or refused key already explains a refused MCP call.
        if facts.mcp in mcp_hints and not (facts.mcp == "refused" and facts.key != "valid"):
            hints.append(mcp_hints[facts.mcp])
    advice = [
        f"{label} gives a tool {s:g} s; Unsloth Studio tools can wait {_MCP_WAIT_CAP_S} s. {fix}"
        for label, s, fix in facts.timeouts
        if s < _MCP_WAIT_CAP_S
    ]
    return not hints, hints, advice


def mcp_doctor(env_var: str = "UNSLOTH_API_KEY", timeout: float = 5.0) -> int:
    facts = gather_mcp_facts(env_var, timeout)
    key = "set" if os.environ.get(env_var) else "not set"
    checked = {
        "valid": "accepted",
        "invalid": "refused",
        "error": "not checked",
        "unsent": "not sent",
    }
    if facts.base and facts.key in checked:
        key += f", {checked[facts.key]}"
    typer.echo("=== Unsloth Studio MCP ===")
    typer.echo(f"  server: {facts.base or 'not reachable'}")
    typer.echo(f"  API key ({env_var}): {key}")
    if facts.base:
        typer.echo(f"  MCP endpoint: {facts.mcp}")
    for label, seconds, _ in facts.timeouts:
        typer.echo(f"  {label} tool timeout: {seconds:g} s")
    healthy, hints, advice = mcp_verdict(facts)
    typer.echo(f"\n=== Verdict: {'OK' if healthy else 'PROBLEMS FOUND'} ===")
    for line in hints + advice or ["No problems found."]:
        typer.echo(f"- {line}")
    return 0 if healthy else 1
