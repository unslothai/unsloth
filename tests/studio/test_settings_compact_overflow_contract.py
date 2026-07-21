"""Responsive overflow contracts for the settings dialog."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SETTINGS_DIALOG = REPO / "studio/frontend/src/features/settings/settings-dialog.tsx"
API_MONITOR = (
    REPO / "studio/frontend/src/features/settings/components/api-monitor-console.tsx"
)
GENERAL_TAB = REPO / "studio/frontend/src/features/settings/tabs/general-tab.tsx"


def test_dialog_content_can_shrink_inside_the_dialog_grid():
    source = SETTINGS_DIALOG.read_text(encoding = "utf-8")
    assert "flex h-full min-h-0 min-w-0 w-full max-sm:flex-col" in source
    assert "relative flex min-h-0 min-w-0 flex-1 flex-col" in source


def test_api_monitor_entries_and_expanded_text_can_shrink():
    source = API_MONITOR.read_text(encoding = "utf-8")
    assert (
        '<article className="min-w-0 rounded-lg border border-border/70 bg-background">'
        in source
    )
    assert (
        '<section className="flex min-w-0 flex-col rounded-lg border border-border/70 bg-background">'
        in source
    )
    assert (
        source.count(
            'className="max-h-44 overflow-auto whitespace-pre-wrap break-words'
        )
        == 2
    )


def test_embedding_model_controls_stack_on_the_narrowest_viewports():
    source = GENERAL_TAB.read_text(encoding = "utf-8")
    assert (
        'className="max-[360px]:flex-col max-[360px]:items-stretch max-[360px]:gap-3"'
    ) in source
    assert 'className="w-[220px] max-[360px]:min-w-0 max-[360px]:flex-1"' in source
