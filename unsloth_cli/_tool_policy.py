# Copyright 2025-present the Unsloth AI Inc. team. All rights reserved.

"""Pure resolver for `unsloth run --enable-tools/--disable-tools`.

Kept as a standalone module so the truth table can be unit-tested
without spinning up Typer or the studio venv.
"""

from typing import Callable, Optional

import typer

# Claude Code brand orange — used so the security warning stands out
# in a crowded terminal.
_PROMPT_FG = (217, 119, 87)

# Loopback aliases. Any other bind address (0.0.0.0, ::, a specific
# LAN IP, a hostname, ...) is treated as network-reachable.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_external_host(host: str) -> bool:
    """True when `host` is reachable from beyond loopback."""
    return host.lower() not in _LOOPBACK_HOSTS


def _build_prompt_text(host: str) -> str:
    return typer.style(
        (
            f"Tools include arbitrary code execution (Python, terminal). "
            f"You're binding to {host}, which is reachable from your network. "
            f"If your API key leaks, anyone with it can run code on this machine. "
            f"Do not share the API key. Continue?"
        ),
        fg = _PROMPT_FG,
        bold = True,
    )


def resolve_tool_policy(
    host: str,
    flag: Optional[bool],
    yes: bool,
    silent: bool,
    prompt: Callable[[str], bool] = typer.confirm,
) -> bool:
    """Return the resolved server-side tool policy.

    Args:
        host: The bind address (e.g. "127.0.0.1", "0.0.0.0", or a
              specific network IP).
        flag: Tri-state from `--enable-tools/--disable-tools`
              (None when neither was passed).
        yes:  True if the operator passed `--yes/-y`.
        silent: True if the operator passed `--silent/-q`.
        prompt: Callable used for the network-bind + on confirmation
                (injected for testability; defaults to ``typer.confirm``).

    Raises:
        typer.Exit: when the operator declines the confirmation.
    """
    is_external = is_external_host(host)
    default = not is_external  # loopback defaults on, network defaults off

    resolved = default if flag is None else flag

    if is_external and resolved is True and not yes and not silent:
        if not prompt(_build_prompt_text(host)):
            raise typer.Exit(1)

    return resolved
