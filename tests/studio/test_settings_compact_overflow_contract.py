"""Responsive overflow contracts for the settings dialog."""

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SETTINGS_DIALOG = REPO / "studio/frontend/src/features/settings/settings-dialog.tsx"
API_MONITOR_PAGE = REPO / "studio/frontend/src/features/api-monitor/api-monitor-page.tsx"
MONITOR_LINK = REPO / "studio/frontend/src/features/settings/components/monitor-link.tsx"
REMOTE_ACCESS = REPO / "studio/frontend/src/features/settings/components/remote-access-section.tsx"
GENERAL_TAB = REPO / "studio/frontend/src/features/settings/tabs/general-tab.tsx"
SETTINGS = REPO / "studio/frontend/src/features/settings"


def test_dialog_content_can_shrink_inside_the_dialog_grid():
    source = SETTINGS_DIALOG.read_text(encoding = "utf-8")
    assert "flex h-full min-h-0 min-w-0 w-full" in source
    # Stacks on the dialog's measured width, not a viewport breakpoint, since the UI scale
    # does not change the viewport. The data variant reads its own element.
    at = source.index("min-w-0 w-full data-stacked:flex-col")
    tag = source[source.rindex("<div", 0, at) : source.index(">", at)]
    assert (
        "data-stacked={stacked || undefined}" in tag
    ), "the stacking attribute left the flex container"
    assert "group/settings" in tag, "the stacked children read group/settings off this container"
    assert "relative flex min-h-0 min-w-0 flex-1 flex-col" in source


def test_api_monitor_entries_and_expanded_text_can_shrink():
    source = API_MONITOR_PAGE.read_text(encoding = "utf-8")
    # Flex parents need min-w-0, or a long model id widens the layout past the viewport.
    assert '"flex w-full min-w-0 flex-col gap-1 border-b border-border/50' in source
    assert '<section className="flex min-w-0 flex-col gap-1.5">' in source
    # Read as tokens.
    text_boxes = [
        set(literal.split())
        for literal in re.findall(r'"([^"\n]*)"', source)
        if {"whitespace-pre-wrap", "rounded-lg"} <= set(literal.split())
    ]
    assert text_boxes, "no wrapped, rounded text box in the API monitor"
    for tokens in text_boxes:
        assert {"max-h-72", "overflow-auto", "break-words"} <= tokens, sorted(tokens)
    assert 'className="min-w-0 break-all font-mono' in source


def test_settings_monitor_link_can_shrink():
    source = MONITOR_LINK.read_text(encoding = "utf-8")
    assert "flex w-full min-w-0 items-center gap-3" in source
    assert '<span className="truncate text-xs text-muted-foreground">' in source


def test_remote_access_card_can_shrink():
    source = REMOTE_ACCESS.read_text(encoding = "utf-8")
    assert '<div className="flex min-w-0 items-start gap-3">' in source
    assert '<div className="flex min-w-0 flex-col gap-0.5">' in source
    assert "block w-full break-all rounded-md" in source
    assert "<RemoteUrlPanel url={status?.url ?? null} />" in source


def test_embedding_model_controls_stack_on_the_narrowest_viewports():
    # Follow the component, not a filename: the picker has moved tabs before.
    owners = [
        path
        for path in sorted(SETTINGS.rglob("*.tsx"))
        if "<EmbeddingModelPicker" in path.read_text(encoding = "utf-8")
    ]
    assert owners, "no settings surface renders EmbeddingModelPicker"

    missing = [
        str(path.relative_to(REPO))
        for path in owners
        if not (
            # The quoted class string, whether it is the whole className or one argument of
            # cn(...) beside a conditional class, which is how #12875 writes it.
            '"max-[360px]:flex-col max-[360px]:items-stretch max-[360px]:gap-3"'
            in (source := path.read_text(encoding = "utf-8"))
            and ("max-[360px]:w-full" in source or "max-[360px]:flex-1" in source)
        )
    ]
    assert missing == []
