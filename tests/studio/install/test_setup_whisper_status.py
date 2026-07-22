# SPDX-License-Identifier: AGPL-3.0-only
"""Contract checks for actionable whisper prebuilt setup outcomes."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_shell_setup_distinguishes_release_skew_from_install_failure():
    script = (ROOT / "studio" / "setup.sh").read_text(encoding="utf-8")
    assert 'if [ "$_WHISPER_STATUS" -eq 2 ]' in script
    assert "installed llama.cpp ${_WHISPER_INSTALLED_TAG:-unknown}" in script
    assert "whisper requires ${_WHISPER_REQUIRED_TAG:-unknown}" in script
    assert "prebuilt install failed" in script
    assert "retry setup or inspect verbose output" in script
    assert "curated whisper.cpp dictation is unavailable" in script
    assert "browser and Transformers dictation remain available" in script


def test_powershell_setup_distinguishes_release_skew_from_install_failure():
    script = (ROOT / "studio" / "setup.ps1").read_text(encoding="utf-8")
    assert "elseif ($whisperExit -eq 2)" in script
    assert "installed llama.cpp $installedWhisperLlamaTag" in script
    assert "whisper requires $requiredWhisperLlamaTag" in script
    assert "prebuilt install failed" in script
    assert "retry setup or inspect verbose output" in script
    assert "curated whisper.cpp dictation is unavailable" in script
    assert "browser and Transformers dictation remain available" in script
