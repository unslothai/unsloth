# SPDX-License-Identifier: AGPL-3.0-only
from core.inference.srt_diagnostics import ProbeReason


def test_windows_diagnostics_are_bounded():
    reason = ProbeReason("dependency_missing", "dependency", "node")
    assert reason.fields()["diagnostic"]["dependency"] == "node"
    assert "secret" not in str(ProbeReason("secret", details = {"secret": "secret"}).fields())
