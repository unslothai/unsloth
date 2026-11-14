"""The seeded browser keeps update offers off, under the keys the frontend actually reads.

An update card in the overlay rail covers the composer; a parity cell that got one timed out clicking
through it. The mute only works while these keys match use-notification-frequency.ts.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.runtime.lifecycle import (  # noqa: E402
    QUIET_NOTIFICATION_CHANNELS,
    QUIET_NOTIFICATIONS,
    StudioAuth,
    seed_init_script,
)

_HOOK = (
    Path(__file__).resolve().parents[5]
    / "studio"
    / "frontend"
    / "src"
    / "hooks"
    / "use-notification-frequency.ts"
)


def _hook() -> str:
    return _HOOK.read_text(encoding = "utf-8")


def _const_list(name: str) -> list[str]:
    body = re.search(rf"export const {name} = \[(.*?)\] as const;", _hook(), re.S)
    assert body, f"{name} is no longer a literal list in {_HOOK.name}"
    return re.findall(r'"([^"]+)"', body.group(1))


def test_every_frontend_channel_is_muted():
    assert sorted(QUIET_NOTIFICATION_CHANNELS) == sorted(_const_list("NOTIFICATION_CHANNELS"))


def test_the_keys_and_value_are_the_ones_the_frontend_reads():
    assert re.search(
        r"const frequencyKey = \(channel: NotificationChannel\) =>\s*`unsloth_\$\{channel\}_notification_frequency`;",
        _hook(),
    )
    assert "off" in _const_list("NOTIFICATION_FREQUENCIES")
    assert set(QUIET_NOTIFICATIONS.values()) == {"off"}


def test_the_seed_script_writes_them():
    script = seed_init_script(
        StudioAuth(
            access_token = "a", refresh_token = "r", base_url = "http://x", username = "u", password = "p"
        ),
        [],
    )
    seed = json.loads(script.split("const seed = ", 1)[1].split("; for (const k", 1)[0])
    for key, value in QUIET_NOTIFICATIONS.items():
        assert seed[key] == value
