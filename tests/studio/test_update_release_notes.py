# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Contracts for the update popup's release-notes preview.

The popup renders CHANGELOG.md notes for the exact version it is offering. The
risk this file guards is showing notes from a different release: a near-miss
lookup must return nothing rather than the newest section it can find."""

from __future__ import annotations

import http.server
import os
import shutil
import subprocess
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
CODE_SPANS = FRONTEND / "lib/markdown-code-spans.ts"
LINKS = FRONTEND / "lib/changelog-links.ts"
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
    """The build backend writes studio/CHANGELOG.md; the root file must win."""
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
    assert "parkCodeSpans" in src, "code spans are parked so their underscores survive"
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
    assert [
        e.version for e in changelog_module.parse_changelog("    ## 9.9.9\n\n## 1.0\n\n- real\n")
    ] == ["1.0"]
    assert [
        e.version
        for e in changelog_module.parse_changelog(
            "## 1.0\n\n    ```\n    sample\n\n## 2.0\n\n- two\n"
        )
    ] == ["1.0", "2.0"]


def test_desktop_notes_link_to_the_release_page_on_every_platform():
    """manualReleaseUrl is Linux-package only, so in-app updates on macOS,
    Windows and AppImage would otherwise link to the generic changelog."""
    hook = (FRONTEND / "hooks/use-tauri-update.ts").read_text(encoding = "utf-8")
    assert "const releasePageUrl = info ?" in hook
    banner = TAURI_BANNER.read_text(encoding = "utf-8")
    assert "releaseNotesUrl={releasePageUrl ?? manualReleaseUrl}" in banner
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    assert "releasePageUrl={update.releasePageUrl}" in provider


def test_preview_matches_how_markdown_renders_prose_and_links():
    """Three rendering mismatches the preview must not reintroduce: wrapped
    paragraphs split into fragments, autolinks eaten as tags, and a lead cut
    at an abbreviation."""
    src = PREVIEW.read_text(encoding = "utf-8")
    # Contiguous prose lines accumulate and flush at a paragraph boundary.
    assert "collector.paragraph = collector.paragraph" in src
    # <https://x> renders as link text, so it is not a tag.
    assert "AUTOLINK" in src
    # "e.g. GGUF" is not a sentence boundary.
    assert "ABBREVIATIONS" in src and "INITIAL" in src


def test_preview_treats_code_as_literal():
    """Inside a code span, and inside an indented code block, Markdown renders
    the text literally, so the preview must not transform or promote it."""
    src = PREVIEW.read_text(encoding = "utf-8")
    # Code spans are parked before any other inline transformation.
    park = src.index("parkCodeSpans(markdown")
    assert park < src.index("stripHtmlTags(\n    parked")
    # A "- cmd" line inside an indented code block is not a headline bullet.
    assert "INDENTED_CODE_INDENT" in src


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


def test_headings_inside_a_raw_html_block_are_not_releases(changelog_module):
    """<pre> content is literal, so a sample heading in it must not become a
    section and must not cut the real section's body short."""
    text = "## 1.0\n\n<pre>\n## 9.9.9\n</pre>\n\n- real note\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0"]
    assert "real note" in changelog_module.find_release_notes(text, "1.0").body
    assert changelog_module.find_release_notes(text, "9.9.9") is None


def test_details_blocks_still_contain_markdown(changelog_module):
    """<details> is a CommonMark type 6 block: headings inside it still count,
    so collapsible sections keep working."""
    text = "## 2.0\n\n<details>\n<summary>More</summary>\n\n- note\n\n</details>\n\n## 1.0\n\n- older\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["2.0", "1.0"]


def test_inline_raw_html_tag_does_not_open_a_block(changelog_module):
    """A block opens only at the start of a line. A tag named mid-sentence is
    inline HTML and must not swallow the releases below it."""
    text = "## 2.0\n\n- Warn when a <script> tag is pasted\n\n## 1.0\n\n- older\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["2.0", "1.0"]


def test_preview_skips_raw_html_blocks():
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "stripRawHtml" in src
    # Anchored: only a line-leading tag opens a block, matching the parser.
    assert "/^ {0,3}<(pre|script|style|textarea)" in src


def test_fence_inside_a_raw_html_block_is_literal(changelog_module):
    """Raw HTML contents are literal, so a stray ``` in a <pre> sample is not a
    fence. Treating it as one left a block open and hid every later release."""
    text = "## 2.0\n\n<pre>\n```\n</pre>\n\n## 1.0\n\n- older\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["2.0", "1.0"]


def test_raw_html_block_closes_on_any_of_the_four_tags(changelog_module):
    """CommonMark ends a type 1 block at the first `</pre>`, `</script>`,
    `</style>` or `</textarea>`: the closer need not match the opener."""
    text = '## 1.0\n\n<script>\nconst sample = "</pre>";\n## 9.9.9\n</script>\n'
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0", "9.9.9"]


@pytest.mark.parametrize("tag", ["details", "div", "table"])
def test_type_6_blocks_run_until_a_blank_line(changelog_module, tag):
    """`<details>` holds Markdown only after a blank line closes the block, so
    a heading pressed against the opening tag is not a release."""
    packed = f"## 1.0\n\n<{tag}>\n## 9.9.9\n</{tag}>\n\n- note\n"
    assert [e.version for e in changelog_module.parse_changelog(packed)] == ["1.0"]
    spaced = f"## 1.0\n\n<{tag}>\n\n## 2.0\n\n- note\n"
    assert [e.version for e in changelog_module.parse_changelog(spaced)] == ["1.0", "2.0"]


def test_a_tag_only_line_cannot_interrupt_a_paragraph(changelog_module):
    """Type 7 blocks do not interrupt a paragraph, so prose followed by a bare
    tag keeps the releases below it reachable."""
    text = "## 2.0\n\nSome prose.\n<span>\n\n## 1.0\n\n- older\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["2.0", "1.0"]


def test_preview_joins_an_indented_continuation_line():
    """Four spaces only start code outside a paragraph. Inside one the line is
    a wrapped continuation, so it must not be dropped from the preview."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "!insideBlock && line.indent >= INDENTED_CODE_INDENT" in src
    # A fence indented into a list item is a block, not a wrapped line.
    assert "opensDeepFence" in src


def test_every_packaging_path_snapshots_the_changelog():
    """`python -m build` and `pip install .` must ship the offline copy too,
    so the snapshot is made by the build backend rather than by build.sh."""
    pyproject = (REPO / "pyproject.toml").read_text(encoding = "utf-8")
    assert 'build_py = "_changelog_build.build_py"' in pyproject
    hook = (REPO / "_changelog_build.py").read_text(encoding = "utf-8")
    assert "studio" in hook and "CHANGELOG.md" in hook
    # The hook has to reach the sdist, or building from one loses the snapshot.
    manifest = (REPO / "MANIFEST.in").read_text(encoding = "utf-8")
    assert "include _changelog_build.py" in manifest
    assert "include CHANGELOG.md" in manifest


def test_preview_code_spans_need_a_matching_closer():
    """A closer is a run of the same length, so ``Use `` `x` `` `` keeps the
    inner backticks the expanded notes show."""
    src = CODE_SPANS.read_text(encoding = "utf-8")
    assert "candidate === ticks" in src, "a closer is a run of the same length"
    assert "stripPadding" in src, "one space of padding is dropped, as in Markdown"


def test_preview_skips_thematic_breaks():
    """`- - -` renders as a rule, so it must not take a preview slot."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "THEMATIC_BREAK" in src
    assert "THEMATIC_BREAK.test(visible)" in src


def test_preview_keeps_quoted_examples_out_of_the_headlines():
    """A quoted list is example output, not a change, so it never competes
    with the release's own bullets."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "quoted: boolean" in src
    assert "if (!line.quoted)" in src, "quoted bullets never become headlines"


def test_notes_panel_keeps_the_link_when_the_lookup_fails():
    """Retry is not the only route: the changelog page can be reachable even
    when the backend lookup is not."""
    src = PANEL.read_text(encoding = "utf-8")
    error_branch = src[src.index('if (state === "error")') :]
    retry = error_branch.index("update-release-notes-retry")
    assert error_branch.index("{link}") > retry, "link sits beside retry"


def test_hook_waits_for_the_desktop_auth_token():
    """The desktop popup can render before auto-auth installs its token, so a
    missing token must not be recorded as a failed lookup."""
    src = NOTES_HOOK.read_text(encoding = "utf-8")
    assert "hasAuthToken()" in src and "AUTH_POLL_LIMIT" in src


def test_installed_layout_prefers_the_bundled_changelog(tmp_path):
    """Installed, the levels above studio/ are site-packages. A stray
    CHANGELOG.md left there by another package must not outrank the bundled
    snapshot, so those levels are only searched in a source checkout."""
    site_packages = tmp_path / "site-packages"
    package = site_packages / "studio/backend/utils"
    package.mkdir(parents = True)
    for name in ("changelog.py", "update_status.py"):
        shutil.copy(BACKEND / "utils" / name, package / name)
    for parent in (site_packages / "studio", package.parent, package):
        (parent / "__init__.py").write_text("", encoding = "utf-8")
    (site_packages / CHANGELOG.name).write_text("## 2.0\n\n- stray\n", encoding = "utf-8")
    bundled = site_packages / "studio" / CHANGELOG.name
    bundled.write_text("## 2.0\n\n- bundled\n", encoding = "utf-8")

    env = {**os.environ, "PYTHONPATH": str(site_packages)}
    env.pop("UNSLOTH_CHANGELOG_PATH", None)

    def served() -> str:
        # cwd is outside the checkout, so this imports the installed copy.
        return subprocess.run(
            [
                sys.executable,
                "-c",
                "from studio.backend.utils import changelog\n"
                "print(changelog._read_local_changelog().text)",
            ],
            capture_output = True,
            text = True,
            env = env,
            cwd = tmp_path,
            check = True,
        ).stdout

    assert "bundled" in served() and "stray" not in served()

    # A checkout marker there means it really is a repo root, so it wins again.
    (site_packages / "pyproject.toml").write_text("", encoding = "utf-8")
    assert "stray" in served()


def test_a_section_staged_as_a_comment_reads_as_unpublished(
    changelog_module, tmp_path, monkeypatch
):
    """Notes staged inside <!-- --> render as nothing, so the popup must say
    no notes were published rather than show an empty surface."""
    monkeypatch.setenv(changelog_module.DISABLE_ENV_VAR, "1")
    local = tmp_path / "CHANGELOG.md"
    local.write_text("## 2.0\n\n<!-- not ready -->\n\n## 1.0\n\n- shipped\n", encoding = "utf-8")
    monkeypatch.setenv(changelog_module.CHANGELOG_PATH_ENV_VAR, str(local))
    changelog_module.reset_changelog_cache()
    try:
        staged = changelog_module.get_release_notes("2.0")
        assert staged["matched"] is False and staged["markdown"] is None
        assert changelog_module.get_release_notes("1.0")["matched"] is True
    finally:
        changelog_module.reset_changelog_cache()


@pytest.mark.parametrize(
    "body,visible",
    [
        ("- note", True),
        ("<!-- staged -->", False),
        ("```\n```", True),
        ("<pre>\n</pre>", True),
        ("  ", False),
    ],
)
def test_visibility_check_only_hides_comments(changelog_module, body, visible):
    assert changelog_module._renders_visibly(body) is visible


@pytest.mark.parametrize(
    "block",
    [
        "<?php\n## 9.9.9\n?>",
        "<![CDATA[\n## 9.9.9\n]]>",
        "<!DOCTYPE\n## 9.9.9\n>",
    ],
)
def test_processing_instructions_and_declarations_are_literal(changelog_module, block):
    """Raw block types 3 to 5 render literally, like <pre>, so a heading inside
    one is a sample and not a release."""
    text = f"## 1.0\n\n{block}\n\n- real note\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0"]
    assert "real note" in changelog_module.find_release_notes(text, "1.0").body


def test_headings_need_a_space_or_tab_after_the_hashes(changelog_module):
    """A non-breaking space pasted from rich text renders as ordinary text, so
    the line must not end the release above it."""
    text = "## 1.0\n\n- real note\n\n## 9.9.9\n\n- not a release\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0"]
    assert changelog_module.find_release_notes(text, "9.9.9") is None
    # A tab is valid and still opens a heading.
    tabbed = "## 1.0\n\n- one\n\n##\t2.0\n\n- two\n"
    assert [e.version for e in changelog_module.parse_changelog(tabbed)] == ["1.0", "2.0"]


def test_preview_skips_every_raw_block_form():
    """The extractor tracks the same block forms as the parser, so a sample
    bullet inside one cannot become the collapsed headline."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "RAW_BLOCKS" in src
    assert "CDATA" in src and "[A-Za-z]" in src


@pytest.mark.parametrize("banner", [WEB_BANNER, TAURI_BANNER])
def test_expanded_popup_fits_a_short_viewport(banner):
    """A window under roughly 430px high used to push the card's title and
    dismiss control above the top of the screen."""
    panel = PANEL.read_text(encoding = "utf-8")
    # The notes region shrinks inside a card capped to the viewport, so the
    # header and the actions stay on screen at any height.
    assert "min-h-0 flex-1" in panel, "notes height must follow the viewport"
    src = banner.read_text(encoding = "utf-8")
    assert "max-h-[calc(100dvh_-_2rem)]" in src, "card is the backstop on tiny viewports"


def test_relative_changelog_links_point_at_the_repository():
    """CHANGELOG.md links are repository-relative. Rendered as-is they resolve
    against Studio's origin, so the renderer blocks them."""
    src = LINKS.read_text(encoding = "utf-8")
    assert "https://github.com/unslothai/unsloth/blob/main/" in src
    assert "https://raw.githubusercontent.com/unslothai/unsloth/main/" in src
    # Absolute targets, fragments, fenced code and code spans stay untouched.
    assert "ABSOLUTE" in src and "codeSpans" in src and "FENCE" in src
    panel = PANEL.read_text(encoding = "utf-8")
    assert "resolveChangelogLinks" in panel


@pytest.mark.parametrize("query", ["latest", "main", "not-a-version", "abc"])
def test_unparseable_versions_are_rejected(changelog_module, query):
    """Sections are indexed only when their version parses, so a query that
    cannot parse can never match and is a bad request, not an empty result."""
    assert changelog_module.is_supported_version_query(query) is False


@pytest.mark.parametrize("query", ["2026.7.5", "v2026.7.5", "2026.07.5", "1.0.0rc1"])
def test_real_versions_are_still_accepted(changelog_module, query):
    assert changelog_module.is_supported_version_query(query) is True


def test_reference_style_images_resolve_to_the_raw_host():
    """`![alt][arch]` with `[arch]: docs/arch.png` needs the raw file: the blob
    URL is an HTML page, so the image would not load."""
    src = LINKS.read_text(encoding = "utf-8")
    assert "IMAGE_REFERENCE" in src
    assert "imageLabels" in src


def test_collapsed_notes_surface_is_hidden_when_nothing_previews():
    """Notes that are only a fenced command block preview as nothing, and an
    empty muted strip is worse than no strip."""
    src = PANEL.read_text(encoding = "utf-8")
    assert "preview?.items.length === 0" in src


def test_a_fence_closer_accepts_only_spaces_and_tabs(changelog_module):
    """A delimiter followed by a non-breaking space is code content, so it must
    not close the block and let a sample heading through."""
    text = "## 1.0\n\n```\n```\u00a0\n## 9.9.9\n```\n\n- real note\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0"]
    plain = "## 1.0\n\n```\nx\n```\t\n\n## 2.0\n\n- two\n"
    assert [e.version for e in changelog_module.parse_changelog(plain)] == ["1.0", "2.0"]
    # The same rule in both frontend scanners.
    for source in (PREVIEW, LINKS):
        assert "/[^ \\t]/" in source.read_text(encoding = "utf-8")


def test_code_spans_close_on_a_run_of_equal_length():
    """`a``b [x](y.md)` is one code span, so the link inside it is literal."""
    src = CODE_SPANS.read_text(encoding = "utf-8")
    assert "candidate === ticks" in src, "closer length must match the opener"
    # Shared, so the preview and the link resolver cannot drift apart.
    assert "markdown-code-spans" in PREVIEW.read_text(encoding = "utf-8")
    assert "markdown-code-spans" in LINKS.read_text(encoding = "utf-8")


def test_preview_decodes_entities_like_the_renderer():
    """Streamdown renders `AT&amp;T` as AT&T, so the collapsed preview must
    not show the raw entity."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "NAMED_ENTITIES" in src and "decodeEntity" in src
    # Decoded before code spans are restored, so code keeps the literal text.
    assert src.index(".replace(ENTITY, decodeEntity)") < src.index(".replace(PARKED")


def test_release_notes_request_refreshes_an_expired_token():
    """A direct fetch cannot recover from a 401; authFetch refreshes first."""
    src = NOTES_HOOK.read_text(encoding = "utf-8")
    assert "authFetch(" in src
    assert "getAuthToken" not in src


def test_preview_handles_the_desktop_updater_line_endings():
    """The updater body arrives with CRLF, which used to hide fences from the
    extractor and promote a code sample to a headline."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "LINE_ENDINGS" in src
    assert "LINE_ENDINGS" in LINKS.read_text(encoding = "utf-8")


def test_preview_renders_reference_links_as_text():
    """`[text][label]` and `![alt][label]` render as a link and an image, so
    the preview must not show their raw markup."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "LINK_REFERENCE" in src and "IMAGE_REFERENCE" in src
    # A definition line renders as nothing, so it is not a preview item.
    assert "DEFINITION" in src


def test_preview_treats_escaped_punctuation_as_literal():
    """`\\*not italic\\*` keeps its stars and an escaped backtick does not open
    a code span."""
    assert "ESCAPE" in PREVIEW.read_text(encoding = "utf-8")
    assert "escaped(" in CODE_SPANS.read_text(encoding = "utf-8")


def test_link_resolver_skips_every_code_form():
    """Indented code and code spans crossing a line render as code, so their
    contents must not be rewritten."""
    src = LINKS.read_text(encoding = "utf-8")
    assert "INDENTED_CODE" in src
    # Spans are scanned over the whole document, not line by line.
    assert "codeSpans(masked)" in src
    # A definition cannot interrupt a paragraph.
    assert "definition.has(index)" in src


def test_badge_links_resolve_both_targets():
    """`[![alt](img)](link)` is the badge idiom: the outer link used to stay
    relative because the label was not allowed to nest."""
    assert "NESTED_LABEL" in LINKS.read_text(encoding = "utf-8")


def test_in_flight_requests_are_identified_not_just_versioned():
    """Two requests for the same version could resolve out of order and leave
    the panel showing the older result."""
    assert "requestIdRef" in NOTES_HOOK.read_text(encoding = "utf-8")


def test_notes_repair_the_shared_previews_width_reset():
    """MarkdownPreview clears max-width on every descendant, so a wide image
    and the renderer's own link dialog escape the card."""
    src = PANEL.read_text(encoding = "utf-8")
    assert "[&_img]:max-w-full" in src
    assert "[&_[data-streamdown=link-safety-modal]>*]:max-w-md" in src


@pytest.mark.parametrize("banner", [WEB_BANNER, TAURI_BANNER])
def test_only_the_notes_region_scrolls(banner):
    """The dismiss control sits inside the card, so scrolling the card itself
    carried it off screen on a short viewport."""
    src = banner.read_text(encoding = "utf-8")
    assert "flex max-h-[calc(100dvh_-_2rem)] flex-col overflow-hidden" in src
    assert 'className="min-h-0 flex-1"' in src
    panel = PANEL.read_text(encoding = "utf-8")
    assert "max-h-64 min-h-0 flex-1 overflow-y-auto" in panel


def test_a_comment_marker_in_prose_cannot_swallow_later_releases(changelog_module):
    """A note that mentions `<!--` used to put the parser into comment state
    for the rest of the file: the releases below it disappeared and their
    notes were served under the newer version's heading."""
    text = (
        "## 2026.8.0\n\n- Studio strips <!-- markers from pasted prompts.\n\n"
        "## 2026.7.5\n\n- SECRET: an older release\n"
    )
    assert [e.version for e in changelog_module.parse_changelog(text)] == [
        "2026.8.0",
        "2026.7.5",
    ]
    assert "SECRET" not in changelog_module.find_release_notes(text, "2026.8.0").body
    assert changelog_module.find_release_notes(text, "2026.7.5") is not None
    # A comment that starts a line is still a block and still hides its body.
    hidden = "## 2.0\n\n<!--\n## 9.9.9\n-->\n\n- note\n"
    assert [e.version for e in changelog_module.parse_changelog(hidden)] == ["2.0"]


def test_a_closing_delimiter_takes_its_whole_line(changelog_module):
    """CommonMark keeps the closing line inside the block, so a heading glued
    after `-->` or `</pre>` is not a release."""
    for text in (
        "## 1.0\n\n<!-- hidden -->## 9.9.9\n\n- note\n",
        "## 1.0\n\n<pre>\nx\n</pre>## 9.9.9\n\n- note\n",
    ):
        assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0"]


def test_an_exact_heading_is_never_shadowed(changelog_module):
    """PEP 440 says 1.0 == 1.0.0, so the normalised match used to win even
    when the file had a section spelled exactly as asked."""
    text = "## 1.0.0\n\n- padded\n\n## 1.0\n\n- exact\n"
    assert changelog_module.find_release_notes(text, "1.0").body == "- exact"
    assert changelog_module.find_release_notes(text, "1.0.0").body == "- padded"
    # Normalised matching still applies when there is no exact heading.
    assert changelog_module.find_release_notes("## 2026.7.6\n\n- x\n", "2026.07.6") is not None


def test_setext_headings_are_release_boundaries(changelog_module):
    """A version over a line of dashes is the same heading in setext form."""
    text = "2.0\n---\n\n- new\n\n1.0\n---\n\n- old\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["2.0", "1.0"]
    assert changelog_module.find_release_notes(text, "2.0").body == "- new"
    # A rule between sections is still a rule, and a setext h1 is not a release.
    assert [
        e.version
        for e in changelog_module.parse_changelog("## 2.0\n\n- a\n\n---\n\n## 1.0\n\n- b\n")
    ] == ["2.0", "1.0"]


def test_a_long_backtick_run_does_not_stall_the_parser(changelog_module):
    """The code-span guard used to backtrack: 20k backticks took over a minute
    and every request re-parsed the file."""
    import time

    text = "## 1.0\n\n- " + "`" * 20_000 + " <!--\n"
    started = time.perf_counter()
    changelog_module.parse_changelog(text)
    assert time.perf_counter() - started < 1.0


def test_the_remote_fetch_has_a_total_deadline(changelog_module):
    """The socket timeout resets on every read, so a trickling server could
    hold a worker for minutes and still be treated as a success."""
    source = (BACKEND / "utils/changelog.py").read_text(encoding = "utf-8")
    assert "deadline = time.monotonic() + CHANGELOG_TIMEOUT_SECONDS" in source
    # read1 returns after one socket read, so the deadline is actually checked.
    assert "response.read1(" in source
    # Waiters give up rather than queue behind a stalled fetch.
    assert "Release notes are still loading." in source


def test_truncated_notes_close_their_fence(changelog_module):
    """A blind slice could end inside a code block and break the rendering."""
    body = "```\n" + "x\n" * 20_000 + "```\n"
    payload = changelog_module._notes_response(version = "1.0", markdown = body, source = "local")
    assert payload["truncated"] is True
    assert payload["markdown"].rstrip().endswith("```")


def test_the_opt_out_beats_the_developer_override():
    """UNSLOTH_STUDIO_FAKE_UPDATE is a dev switch; the documented kill switch
    still wins, and the value has to parse as a version."""
    source = (BACKEND / "utils/update_status.py").read_text(encoding = "utf-8")
    assert "forced_version and not disabled and _is_version(forced_version)" in source


def test_a_list_item_over_dashes_is_not_a_setext_heading(changelog_module):
    """`- first` followed by `---` is a list and a rule. Reading it as a
    heading discarded the bullet and the rest of the section with it."""
    text = "## 1.0\n\n- first\n---\n\n- second\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0"]
    body = changelog_module.find_release_notes(text, "1.0").body
    assert "first" in body and "second" in body
    # Real setext headings still work.
    setext = "2.0\n---\n\n- new\n\n1.0\n---\n\n- old\n"
    assert [e.version for e in changelog_module.parse_changelog(setext)] == ["2.0", "1.0"]


def test_a_backtick_in_a_fence_info_string_is_not_a_fence(changelog_module):
    """CommonMark forbids backticks in a backtick fence's info string, so such
    a line is prose and must not swallow the releases below it."""
    text = "## 2.0\n\n```bad`info\n\n## 1.0\n\n- old\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["2.0", "1.0"]
    # A tilde fence may hold backticks, and a normal fence still hides samples.
    assert [
        e.version
        for e in changelog_module.parse_changelog(
            "## 2.0\n\n```md\n## 9.9.9\n```\n\n## 1.0\n\n- old\n"
        )
    ] == ["2.0", "1.0"]
    for source in (PREVIEW, LINKS):
        assert "info string" in source.read_text(encoding = "utf-8")


def test_preview_follows_commonmark_paragraph_rules():
    """Only an ordered list starting at 1 may interrupt a paragraph, and an
    unresolved reference keeps its brackets."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "const interrupts = collector.current === null" in src
    assert "definedLabel" in src, "a reference only renders as text when defined"
    # A comment written mid-sentence hides its own line at most.
    assert "COMMENT_BLOCK_OPEN" in src


def test_link_resolver_leaves_raw_blocks_and_escapes_alone():
    src = LINKS.read_text(encoding = "utf-8")
    assert "RAW_HTML_OPEN" in src and "inRawHtml" in src
    assert "isEscaped(line, opener)" in src
    # A heading ends a paragraph, so a definition under one is a definition.
    assert "BLOCK_LINE.test(line)" in src


def test_code_span_closers_ignore_backslashes():
    """Escapes are not processed inside a code span, so a run after a
    backslash still closes it."""
    src = CODE_SPANS.read_text(encoding = "utf-8")
    body = src[src.index("export function codeSpans") :]
    assert body.count("escaped(text") == 1, "only an opener can be escaped"


def test_the_overlay_stack_fits_the_viewport():
    """The update card's own cap does not account for a long download list
    stacked beneath it."""
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    assert "max-h-[calc(100dvh_-_2rem)]" in provider
    panel = (FRONTEND / "features/hub/download-manager/download-manager-panel.tsx").read_text(
        encoding = "utf-8"
    )
    # Both overlays scroll internally, so they can give up height.
    assert "flex min-h-0" in panel
    assert "flex min-h-0" in WEB_BANNER.read_text(encoding = "utf-8")
