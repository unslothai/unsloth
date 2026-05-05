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

PROMPT_TEXT = typer.style(
    (
        "Tools include arbitrary code execution (Python, terminal). "
        "You're binding to 0.0.0.0, which is reachable from your network. "
        "If your API key leaks, anyone with it can run code on this machine. "
        "Do not share the API key. Continue?"
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
        host: The bind address (e.g. "127.0.0.1" or "0.0.0.0").
        flag: Tri-state from `--enable-tools/--disable-tools`
              (None when neither was passed).
        yes:  True if the operator passed `--yes/-y`.
        silent: True if the operator passed `--silent/-q`.
        prompt: Callable used for the 0.0.0.0 + on confirmation
                (injected for testability; defaults to ``typer.confirm``).

    Raises:
        typer.Exit: when the operator declines the 0.0.0.0 confirmation.
    """
    is_external = host == "0.0.0.0"
    default = not is_external  # localhost defaults on, 0.0.0.0 defaults off

    resolved = default if flag is None else flag

    if is_external and resolved is True and not yes and not silent:
        if not prompt(PROMPT_TEXT):
            raise typer.Exit(1)

    return resolved
