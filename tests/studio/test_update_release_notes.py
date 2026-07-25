# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Contracts for the update popup's release-notes preview.

The popup renders CHANGELOG.md notes for the exact version it is offering. The
risk this file guards is showing notes from a different release: a near-miss
lookup must return nothing rather than the newest section it can find."""

from __future__ import annotations

import http.server
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "studio/backend"
FRONTEND = REPO / "studio/frontend/src"
CHANGELOG = REPO / "CHANGELOG.md"
PANEL = FRONTEND / "components/update/release-notes-panel.tsx"
NOTES_HOOK = FRONTEND / "hooks/use-release-notes.ts"
PREVIEW = FRONTEND / "lib/release-notes-preview.ts"
WEB_BANNER = FRONTEND / "components/web/update-banner.tsx"
TAURI_BANNER = FRONTEND / "components/tauri/update-banner.tsx"

SAMPLE = """# Changelog

Intro prose that belongs to no release.

## Format

```md
## 9999.9.9 - fenced sample, not a real section
```

## Unreleased

- staged note

## 2026.7.6 - 2026-07-22

### What's Changed

- newer thing

## 2026.7.5

### What's Changed

- older thing
"""


@pytest.fixture(scope = "module")
def changelog_module():
    sys.path.insert(0, str(BACKEND))
    try:
        from utils import changelog
    finally:
        sys.path.pop(0)
    changelog.reset_changelog_cache()
    yield changelog
    changelog.reset_changelog_cache()


@pytest.fixture
def isolated_changelog(changelog_module, tmp_path, monkeypatch):
    """Point the module at a temp file and away from the network."""
    monkeypatch.setenv(changelog_module.DISABLE_ENV_VAR, "1")
    path = tmp_path / "CHANGELOG.md"
    path.write_text(SAMPLE, encoding = "utf-8")
    monkeypatch.setenv(changelog_module.CHANGELOG_PATH_ENV_VAR, str(path))
    changelog_module.reset_changelog_cache()
    yield changelog_module
    changelog_module.reset_changelog_cache()


def test_only_real_release_headings_become_sections(changelog_module):
    versions = [entry.version for entry in changelog_module.parse_changelog(SAMPLE)]
    # "Format"/"Unreleased" are not versions, and the fenced 9999.9.9 is sample
    # markdown inside a code block.
    assert versions == ["2026.7.6", "2026.7.5"]


def test_section_body_stops_at_the_next_release(changelog_module):
    entry = changelog_module.find_release_notes(SAMPLE, "2026.7.6")
    assert entry is not None
    assert "newer thing" in entry.body
    assert "older thing" not in entry.body


def test_unknown_version_returns_no_notes_instead_of_a_nearby_release(changelog_module):
    assert changelog_module.find_release_notes(SAMPLE, "2026.7.7") is None
    assert changelog_module.find_release_notes(SAMPLE, "2026.7") is None


def test_version_equality_is_normalized_not_fuzzy(changelog_module):
    entry = changelog_module.find_release_notes(SAMPLE, "2026.07.6")
    assert entry is not None and entry.version == "2026.7.6"


def test_response_reports_no_match_without_markdown(isolated_changelog):
    payload = isolated_changelog.get_release_notes("2026.7.7")
    assert payload["matched"] is False
    assert payload["markdown"] is None
    assert payload["version"] == "2026.7.7"
    # The UI still needs somewhere to send the user.
    assert payload["release_notes_url"]


def test_response_matches_local_changelog_when_offline(isolated_changelog):
    payload = isolated_changelog.get_release_notes("2026.7.6")
    assert payload["matched"] is True
    assert payload["source"] == "local"
    assert "newer thing" in payload["markdown"]


def test_unsupported_version_query_is_rejected(isolated_changelog):
    assert isolated_changelog.is_supported_version_query("2026.7.6") is True
    for bad in ("../etc/passwd", "2026.7.6 OR 1", "", "a" * 80):
        assert isolated_changelog.is_supported_version_query(bad) is False
    assert isolated_changelog.get_release_notes("../etc/passwd")["matched"] is False


def test_remote_changelog_wins_over_bundled_copy(changelog_module, tmp_path, monkeypatch):
    """The offered version is newer than the installed checkout, so the repo
    copy has to be able to describe versions the local file has never heard of."""
    monkeypatch.delenv(changelog_module.DISABLE_ENV_VAR, raising = False)
    local = tmp_path / "CHANGELOG.md"
    local.write_text(SAMPLE, encoding = "utf-8")
    monkeypatch.setenv(changelog_module.CHANGELOG_PATH_ENV_VAR, str(local))

    remote_body = "# Changelog\n\n## 2026.8.0\n\n- shipped after this install\n"

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib naming
            payload = remote_body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    try:
        monkeypatch.setenv(
            changelog_module.CHANGELOG_URL_ENV_VAR,
            f"http://127.0.0.1:{server.server_port}/CHANGELOG.md",
        )
        changelog_module.reset_changelog_cache()
        payload = changelog_module.get_release_notes("2026.8.0")
        assert payload["matched"] is True
        assert payload["source"] == "remote"
        assert "shipped after this install" in payload["markdown"]
    finally:
        server.shutdown()
        server.server_close()
        changelog_module.reset_changelog_cache()


def test_repo_changelog_exists_and_parses(changelog_module):
    assert CHANGELOG.is_file(), "CHANGELOG.md is the editable source of release notes"
    entries = changelog_module.parse_changelog(CHANGELOG.read_text(encoding = "utf-8"))
    assert entries, "CHANGELOG.md needs at least one `## <version>` section"


def test_longer_outer_fence_does_not_leak_a_fake_section(changelog_module):
    """A ``` sample inside a ```` block must not close the block and let the
    sample's heading be indexed as a real release."""
    text = "## 1.0\n\n````md\n```\n## 9.9.9\n```\n````\n\n- real note\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0"]
    assert changelog_module.find_release_notes(text, "9.9.9") is None


def test_tilde_fence_is_not_closed_by_backticks(changelog_module):
    text = "## 1.0\n\n~~~\n```\n## 9.9.9\n~~~\n\n- real\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0"]


def test_utf8_bom_does_not_hide_the_first_section(changelog_module):
    """Editors on Windows can leave a BOM on the first line."""
    assert [e.version for e in changelog_module.parse_changelog("\ufeff## 1.0\n\n- x\n")] == ["1.0"]


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_non_unix_line_endings(changelog_module, newline):
    text = f"## 1.0{newline}{newline}- windows note{newline}"
    entry = changelog_module.find_release_notes(text, "1.0")
    assert entry is not None and "windows note" in entry.body
    assert "\r" not in entry.body


def test_closing_fence_must_carry_nothing_after_it(changelog_module):
    """CommonMark: a closer is the delimiter plus whitespace only. A ```` line
    with trailing text inside a ```` block is content, not the end."""
    text = "## 1.0\n\n````md\n```` not a closer\n## 9.9.9\n````\n\n- real\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0"]
    # An opening fence may still carry an info string.
    info = "## 1.0\n\n```python\n## 9.9.9\n```\n\n- real\n"
    assert [e.version for e in changelog_module.parse_changelog(info)] == ["1.0"]


@pytest.mark.parametrize(
    "text",
    [
        "## 1.0\n\n- real\n\n<!--\n## 9.9.9\n\n- unpublished\n-->\n",
        "## 1.0\n\n- real\n\n<!-- ## 9.9.9 -->\n",
    ],
)
def test_commented_out_sections_are_not_releases(changelog_module, text):
    """Markdown does not render them, so they are not published notes."""
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0"]
    assert changelog_module.find_release_notes(text, "9.9.9") is None


def test_repo_root_changelog_is_preferred_over_the_build_snapshot(changelog_module):
    """build.sh writes studio/CHANGELOG.md; the edited root file must win."""
    paths = [str(p) for p in changelog_module._local_changelog_candidates()]
    root = next(
        i
        for i, p in enumerate(paths)
        if p.endswith(f"/unsloth/{changelog_module.CHANGELOG_FILENAME}")
    )
    packaged = next(
        i
        for i, p in enumerate(paths)
        if p.endswith(f"/studio/{changelog_module.CHANGELOG_FILENAME}")
    )
    assert root < packaged
    build = (REPO / "build.sh").read_text(encoding = "utf-8")
    assert "rm -f studio/CHANGELOG.md" in build, "snapshot must not linger after a build"


def test_preview_keeps_identifier_underscores():
    """UNSLOTH_DISABLE_UPDATE_CHECK must not render as UNSLOTHDISABLEUPDATECHECK."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "BOLD_UNDERSCORE" in src and "ITALIC_UNDERSCORE" in src
    assert "CODE_SPAN" in src, "code spans are parked so their underscores survive"
    assert "const EMPHASIS" not in src, "the blanket emphasis strip is gone"


def test_panel_prefers_the_callers_release_url():
    """The API only returns the generic changelog; the desktop banner passes
    the exact release page for the version being offered."""
    src = PANEL.read_text(encoding = "utf-8")
    assert "releaseNotesUrl ?? notes?.releaseNotesUrl" in src


def test_remote_failure_is_reported_so_the_ui_can_retry(changelog_module, tmp_path, monkeypatch):
    """A bundled changelog cannot know a version newer than the install, so a
    failed remote lookup must not read as "no notes were published"."""
    monkeypatch.delenv(changelog_module.DISABLE_ENV_VAR, raising = False)
    local = tmp_path / "CHANGELOG.md"
    local.write_text("## 1.0\n\n- old release\n", encoding = "utf-8")
    monkeypatch.setenv(changelog_module.CHANGELOG_PATH_ENV_VAR, str(local))
    # Port 9 (discard) refuses fast, standing in for an unreachable host.
    monkeypatch.setenv(changelog_module.CHANGELOG_URL_ENV_VAR, "http://127.0.0.1:9/CHANGELOG.md")
    changelog_module.reset_changelog_cache()
    try:
        payload = changelog_module.get_release_notes("2.0")
        assert payload["matched"] is False
        assert payload["error"], "remote failure must reach the UI"
    finally:
        changelog_module.reset_changelog_cache()


def test_preview_keeps_comparison_operators():
    """ "Support Python <3.15 and >3.9" must not lose its operators to the tag
    strip, which would turn it into "Support Python 3.9"."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "/<\\/?[a-zA-Z][^>]*>/g" in src, "tag strip must require a name character"


def test_preview_hides_commented_out_notes():
    """Unpublished notes inside <!-- --> are not rendered, so not previewed."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "stripCommentSpans" in src and "COMMENT_OPEN" in src


def test_hook_treats_a_reported_failure_as_retryable():
    src = NOTES_HOOK.read_text(encoding = "utf-8")
    assert "next.error !== null" in src


def test_comment_delimiter_in_inline_code_is_literal(changelog_module):
    """A note documenting `<!--` used to put the parser into comment state,
    swallowing every release below it."""
    text = "## 2.0\n\n- Type `<!--` to begin a comment\n\n## 1.0\n\n- older\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["2.0", "1.0"]
    assert changelog_module.find_release_notes(text, "1.0") is not None
    assert "older" not in changelog_module.find_release_notes(text, "2.0").body


def test_refresh_retries_a_cached_remote_failure(changelog_module, tmp_path, monkeypatch):
    """Retry must reach the network again once connectivity returns, rather
    than replaying the cached failure until its TTL expires."""
    monkeypatch.delenv(changelog_module.DISABLE_ENV_VAR, raising = False)
    local = tmp_path / "CHANGELOG.md"
    local.write_text("## 1.0\n\n- old\n", encoding = "utf-8")
    monkeypatch.setenv(changelog_module.CHANGELOG_PATH_ENV_VAR, str(local))

    hits = {"count": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib naming
            hits["count"] += 1
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    try:
        monkeypatch.setenv(
            changelog_module.CHANGELOG_URL_ENV_VAR,
            f"http://127.0.0.1:{server.server_port}/CHANGELOG.md",
        )
        changelog_module.reset_changelog_cache()
        changelog_module.get_release_notes("2.0")
        changelog_module.get_release_notes("2.0")
        assert hits["count"] == 1, "the failure should be cached"
        changelog_module.get_release_notes("2.0", refresh = True)
        assert hits["count"] == 2, "refresh must bypass the cached failure"
    finally:
        server.shutdown()
        server.server_close()
        changelog_module.reset_changelog_cache()


def test_hook_never_returns_another_versions_notes():
    """On the render where the offered version changes, state still describes
    the previous one until the effect runs."""
    src = NOTES_HOOK.read_text(encoding = "utf-8")
    assert "notes.version === version" in src
    assert "refresh" in src, "retry must ask the backend to bypass its cache"


@pytest.mark.parametrize("indent", ["", " ", "  ", "   "])
def test_headings_and_fences_allow_commonmark_indentation(changelog_module, indent):
    """Markdown renders up to three leading spaces, so the parser must agree
    or an indented release is unreachable and its notes join the one above."""
    text = f"## 1.0\n\n- one\n\n{indent}## 2.0\n\n- two\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0", "2.0"]
    fenced = f"## 1.0\n\n{indent}```\n{indent}## 9.9.9\n{indent}```\n\n- real\n"
    assert [e.version for e in changelog_module.parse_changelog(fenced)] == ["1.0"]


def test_four_space_indentation_is_code_not_structure(changelog_module):
    """At four spaces Markdown switches to indented code, for both forms."""
    assert [e.version for e in changelog_module.parse_changelog(
        "    ## 9.9.9\n\n## 1.0\n\n- real\n"
    )] == ["1.0"]
    assert [e.version for e in changelog_module.parse_changelog(
        "## 1.0\n\n    ```\n    sample\n\n## 2.0\n\n- two\n"
    )] == ["1.0", "2.0"]


def test_desktop_notes_link_to_the_release_page_on_every_platform():
    """manualReleaseUrl is Linux-package only, so in-app updates on macOS,
    Windows and AppImage would otherwise link to the generic changelog."""
    hook = (FRONTEND / "hooks/use-tauri-update.ts").read_text(encoding = "utf-8")
    assert "const releasePageUrl = info ?" in hook
    banner = TAURI_BANNER.read_text(encoding = "utf-8")
    assert "releaseNotesUrl={releasePageUrl ?? manualReleaseUrl}" in banner
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    assert "releasePageUrl={update.releasePageUrl}" in provider


def test_desktop_updater_metadata_maps_published_field_names():
    """latest.json publishes Tauri's `notes`/`pub_date`; the manual Linux path
    must read those, not `body`/`date`, or its release notes are always empty."""
    rust = (REPO / "studio/src-tauri/src/desktop_update_policy.rs").read_text(encoding = "utf-8")
    assert 'alias = "body"' in rust and "notes: Option<String>" in rust
    assert 'alias = "date"' in rust and "pub_date: Option<String>" in rust
    assert "body: metadata.notes" in rust and "date: metadata.pub_date" in rust
    workflow = (REPO / ".github/workflows/release-desktop.yml").read_text(encoding = "utf-8")
    assert "'notes': notes," in workflow, "workflow no longer publishes `notes`"


def test_backend_exposes_release_notes_route():
    src = (BACKEND / "main.py").read_text(encoding = "utf-8")
    assert '@app.get("/api/studio/release-notes")' in src
    assert "is_supported_version_query" in src


def test_panel_is_scrollable_and_version_scoped():
    src = PANEL.read_text(encoding = "utf-8")
    assert "overflow-y-auto" in src, "release notes must scroll inside the popup"
    assert "max-h-" in src, "the scroller needs a bounded height"
    # Falls back to the payload's own body only, never to another version.
    assert "fallbackMarkdown" in src


def test_notes_surface_is_borderless_and_lifts_in_dark_mode():
    src = PANEL.read_text(encoding = "utf-8")
    assert "border border-border" not in src, "the notes box is a fill, not a bordered box"
    # Lighter than the card behind it, rather than a darker inset.
    assert "dark:bg-white/[0.06]" in src
    # Streamdown's mt-6 pins the first heading to the scroller edge and clips
    # its ascenders.
    assert "[&>*>*:first-child]:mt-0" in src
    # Shared utility: thumb hidden until the notes are hovered.
    assert "hover-scrollbar" in src
    # Streamdown renders code at text-sm, twice this panel's body size.
    assert "[&_code]:text-[0.92em]" in src


def test_hook_discards_notes_for_a_different_version():
    src = NOTES_HOOK.read_text(encoding = "utf-8")
    assert "notesVersion !== version" in src


def test_collapsed_panel_previews_the_top_bullets():
    """Collapsed popups show the headline changes without an extra click."""
    preview = PREVIEW.read_text(encoding = "utf-8")
    assert "RELEASE_NOTES_PREVIEW_ITEMS = 4" in preview
    # Wrapped bullets join into one item, or a preview ends mid-sentence.
    assert "collectBullets" in preview and "flush" in preview
    # Nested list items are detail, not headline changes.
    assert "NESTED_INDENT_TOLERANCE" in preview
    # Tag stripping repeats: one pass turns `<<b>b>` back into a live tag.
    assert "while (out !== previous)" in preview

    panel = PANEL.read_text(encoding = "utf-8")
    assert "releaseNotesPreview" in panel
    assert 'data-testid="update-release-notes-summary"' in panel
    # Notes are fetched when the popup appears, since the collapsed preview
    # needs them too.
    assert "enabled: true" in panel


def test_preview_highlights_the_leading_sentence():
    """Each bullet leads with its headline sentence, emphasised over the rest."""
    preview = PREVIEW.read_text(encoding = "utf-8")
    assert "splitLeadSentence" in preview
    # A period inside "CHANGELOG.md" or "e.g." must not read as a break.
    assert "SENTENCE_BREAK" in preview and "(?=" in preview

    panel = PANEL.read_text(encoding = "utf-8")
    assert '<span className="font-medium text-foreground">{item.lead}</span>' in panel
    assert "item.rest" in panel


@pytest.mark.parametrize("banner", [WEB_BANNER, TAURI_BANNER])
def test_update_popup_is_wider_than_the_other_overlays(banner):
    """The card is sized for three same-size buttons on one row.

    Width moved from the shared overlay stack onto each overlay, so widening
    the update popup does not widen the llama.cpp banner or download panel."""
    assert "max-w-[448px]" in banner.read_text(encoding = "utf-8")
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    assert "max-w-[400px]" not in provider, "stack must not cap overlay width"
    llama = (FRONTEND / "components/llama-update-banner.tsx").read_text(encoding = "utf-8")
    assert "max-w-[400px]" in llama, "unrelated overlays keep their width"


@pytest.mark.parametrize("banner", [WEB_BANNER, TAURI_BANNER])
def test_banners_toggle_inline_release_notes(banner):
    src = banner.read_text(encoding = "utf-8")
    assert "ReleaseNotesPanel" in src
    assert "Show release notes" in src and "Hide release notes" in src
    # Expansion state is keyed by version, so a new offer cannot leave the
    # previous release's notes on screen.
    assert "notesVersion" in src


@pytest.mark.parametrize(
    "banner,toggle,action",
    [
        (WEB_BANNER, "web-update-release-notes-toggle", "web-update-snooze-button"),
        (TAURI_BANNER, "tauri-update-release-notes-toggle", "Remind me later"),
    ],
)
def test_notes_toggle_shares_the_action_row(banner, toggle, action):
    """The toggle sits in the same row as the actions, not on its own line."""
    src = banner.read_text(encoding = "utf-8")
    row = src.index("mt-4 flex")
    assert row < src.index(toggle) < src.index(action)
    # Same type size as the actions it sits beside; nowrap stops a label
    # breaking across lines inside the row.
    toggle_line = next(line for line in src.splitlines() if toggle in line)
    toggle_block = src[src.index("Button", row) : src.index(toggle_line)]
    assert "text-ui-13" in toggle_block and "whitespace-nowrap" in toggle_block
