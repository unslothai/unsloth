"""Width contract for voice settings selects."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
VOICE_TAB = REPO / "studio/frontend/src/features/settings/tabs/voice-tab.tsx"
SELECT = REPO / "studio/frontend/src/components/ui/select.tsx"


def test_voice_selects_grow_without_overflowing_the_dialog():
    source = VOICE_TAB.read_text(encoding = "utf-8")
    assert source.count('className="min-w-56 max-w-72"') == 4

    select_source = SELECT.read_text(encoding = "utf-8")
    assert "*:data-[slot=select-value]:line-clamp-1" in select_source
