"""Responsive overflow contracts for the settings dialog."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SETTINGS_DIALOG = REPO / "studio/frontend/src/features/settings/settings-dialog.tsx"
# The monitor has its own page and Settings links to it; the shrink contract covers both.
API_MONITOR_PAGE = REPO / "studio/frontend/src/features/api-monitor/api-monitor-page.tsx"
MONITOR_LINK = REPO / "studio/frontend/src/features/settings/components/monitor-link.tsx"
REMOTE_ACCESS = REPO / "studio/frontend/src/features/settings/components/remote-access-section.tsx"
GENERAL_TAB = REPO / "studio/frontend/src/features/settings/tabs/general-tab.tsx"
SETTINGS = REPO / "studio/frontend/src/features/settings"


def test_dialog_content_can_shrink_inside_the_dialog_grid():
    source = SETTINGS_DIALOG.read_text(encoding = "utf-8")
    assert "flex h-full min-h-0 min-w-0 w-full max-sm:flex-col" in source
    assert "relative flex min-h-0 min-w-0 flex-1 flex-col" in source


def test_api_monitor_entries_and_expanded_text_can_shrink():
    source = API_MONITOR_PAGE.read_text(encoding = "utf-8")
    # Flex parents need min-w-0, or a long model id widens the layout past the viewport.
    assert '"flex w-full min-w-0 flex-col gap-1 border-b border-border/50' in source
    assert '<section className="flex min-w-0 flex-col gap-1.5">' in source
    # Prompt and reply are unbounded user text: height-capped, scrollable, wrapped.
    assert "max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-muted/50" in source
    # A model id or path has no spaces to wrap on, so it needs break-all.
    assert 'className="min-w-0 break-all font-mono' in source


def test_settings_monitor_link_can_shrink():
    source = MONITOR_LINK.read_text(encoding = "utf-8")
    assert "flex w-full min-w-0 items-center gap-3" in source
    # The summary line carries a model id, so it truncates instead of widening.
    assert '<span className="truncate text-xs text-muted-foreground">' in source


def test_remote_access_card_can_shrink():
    source = REMOTE_ACCESS.read_text(encoding = "utf-8")
    # The heading and its status sit beside a button, so they need to shrink.
    assert '<div className="flex min-w-0 items-start gap-3">' in source
    assert '<div className="flex min-w-0 flex-col gap-0.5">' in source
    # A tunnel URL has no spaces to wrap on, so it needs break-all.
    assert "block w-full break-all rounded-md" in source
    assert "<RemoteUrlPanel url={status?.url ?? null} />" in source


def test_embedding_model_controls_stack_on_the_narrowest_viewports():
    # The picker has already moved once, from the General tab to Documents & RAG, and pinning the filename turned that
    # move into a red build even though both responsive classes came along untouched.
    # follow the component, since the contract is that the control stacks and fills the row under 360px.
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
            'className="max-[360px]:flex-col max-[360px]:items-stretch max-[360px]:gap-3"'
            in (source := path.read_text(encoding = "utf-8"))
            # The fixed width has to give way at the breakpoint: flex-1 for the
            # combobox, a full row for the picker trigger.
            and ("max-[360px]:w-full" in source or "max-[360px]:flex-1" in source)
        )
    ]
    assert missing == []
