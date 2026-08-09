# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Contracts for the update popup's release-notes preview.

The popup renders CHANGELOG.md notes for the exact version it is offering. The
risk this file guards is showing notes from a different release: a near-miss
lookup must return nothing rather than the newest section it can find."""

from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
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
LIST_COLUMNS = FRONTEND / "lib/markdown-list-columns.ts"
INLINE_COMMENTS = FRONTEND / "lib/markdown-inline-comments.ts"
WEB_BANNER = FRONTEND / "components/web/update-banner.tsx"
TAURI_BANNER = FRONTEND / "components/tauri/update-banner.tsx"

# The scanners are the frontend half of the contract the parser implements, so they are
# run rather than read. Node strips the types and nothing imports a package: no install.
_TS_ALIAS = re.compile(r'"@/lib/([a-z-]+)"')
_TS_RUNNER = """
import { resolveChangelogLinks } from "./changelog-links.ts";
import { releaseNotesPreview } from "./release-notes-preview.ts";

const chunks: Buffer[] = [];
process.stdin.on("data", (chunk: Buffer) => chunks.push(chunk));
process.stdin.on("end", () => {
  const markdown = Buffer.concat(chunks).toString("utf8");
  const result =
    process.argv[2] === "links"
      ? resolveChangelogLinks(markdown)
      : releaseNotesPreview(markdown);
  process.stdout.write(JSON.stringify(result));
});
"""

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
    # "Format"/"Unreleased" are not versions, and 9999.9.9 is fenced sample.
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
    # Resolved paths, not name suffixes: a checkout may be renamed and Windows uses "\".
    paths = [Path(p).resolve() for p in changelog_module._local_changelog_candidates()]
    root = paths.index((REPO / changelog_module.CHANGELOG_FILENAME).resolve())
    packaged = paths.index((REPO / "studio" / changelog_module.CHANGELOG_FILENAME).resolve())
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
    text = f"## 1.0\n\nOne.\n\n{indent}## 2.0\n\nTwo.\n"
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
    # Streamdown's mt-6 clips the first heading against the scroller edge.
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
    # Fetched when the popup appears: the collapsed preview needs them too.
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
    # Keyed by version, so a new offer cannot leave old notes on screen.
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
    # Same type size as the actions beside it; nowrap keeps labels on one line.
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
    # Measured from the line's container, so an item's own indent does not count.
    assert "!insideBlock && line.indent - line.column >= INDENTED_CODE_INDENT" in src
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
    # The notes region shrinks inside the capped card, so header and actions stay on screen.
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


def test_unmatched_backtick_runs_stay_linear(changelog_module):
    """Rescanning the suffix for every opener was quadratic: a line of runs of
    1, 2, 3 ... backticks, none of which ever closes, took 7.7s at 321 KB and
    is reparsed on every popup request, so one malformed remote changelog could
    tie up backend workers."""
    line = "".join("`" * (i + 1) + "x" for i in range(800))
    assert len(line) > 300_000
    started = time.monotonic()
    assert changelog_module._code_span_ranges(line) == []
    assert time.monotonic() - started < 2.0


def test_a_base_exception_releases_the_single_flight_flag(changelog_module, monkeypatch):
    """The flag was cleared only after `except Exception`, so a BaseException
    (KeyboardInterrupt, SystemExit, CancelledError) stranded it and every later
    caller then waited out the full deadline for the life of the process."""
    changelog_module.reset_changelog_cache()

    def explode():
        raise KeyboardInterrupt

    monkeypatch.setattr(changelog_module, "_fetch_remote_changelog", explode)
    with pytest.raises(KeyboardInterrupt):
        changelog_module.get_remote_changelog()
    assert changelog_module._remote_fetching is False
    changelog_module.reset_changelog_cache()


@pytest.mark.parametrize("marker", ["<!-->", "<!--->"])
def test_an_empty_comment_does_not_swallow_later_releases(changelog_module, marker):
    """`<!-->` and `<!--->` are complete comments in CommonMark: the closer
    overlaps the opener. Searching for `-->` past the opener missed them, so an
    empty comment used as a section marker hid every release below it."""
    text = f"## 2.0\n\n- new stuff\n\n{marker}\n\n## 1.0\n\n- old stuff\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["2.0", "1.0"]
    assert changelog_module.find_release_notes(text, "1.0") is not None
    assert "old stuff" not in changelog_module.find_release_notes(text, "2.0").body
    # The frontend scanner has to agree, or the preview and the body disagree.
    assert "!line.includes(COMMENT_CLOSE)" in PREVIEW.read_text(encoding = "utf-8")


def test_an_unterminated_comment_still_hides_the_rest(changelog_module):
    """The fix must not turn every `<!--` line into a no-op block."""
    text = "## 2.0\n\n<!-- never closed\n\n## 1.0\n\n- old stuff\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["2.0"]


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
    unresolved reference keeps its brackets. A quote owns the paragraph its own
    lines hold, so a marker written outside the quote interrupts nothing."""
    src = " ".join(PREVIEW.read_text(encoding = "utf-8").split())
    assert "const interrupts = collector.current === null" in src
    assert "!collector.quotedParagraph;" in src
    assert "definedLabel" in src, "a reference only renders as text when defined"
    # A comment written mid-sentence hides its own line at most.
    assert "COMMENT_BLOCK_OPEN" in src


def test_link_resolver_leaves_raw_blocks_and_escapes_alone():
    src = LINKS.read_text(encoding = "utf-8")
    assert "RAW_HTML_OPEN" in src and "inRawHtml" in src
    assert "isEscaped(line, opener)" in src
    # A heading ends a paragraph, so a definition under one is a definition.
    assert "BLOCK_LINE.test(structure)" in src


def test_code_span_closers_ignore_backslashes():
    """Escapes are not processed inside a code span, so a run after a
    backslash still closes it."""
    src = CODE_SPANS.read_text(encoding = "utf-8")
    body = src[src.index("export function codeSpans") :]
    assert body.count("escaped(text") == 1, "only an opener can be escaped"


def test_the_overlay_stack_fits_the_viewport():
    """The update card's own cap does not account for a long download list stacked
    beneath it. The cap is no longer the literal class `max-h-[calc(100dvh_-_2rem)]`
    but `stackGeometry`, whose arithmetic is checked numerically in
    studio/frontend/tests/monitor-stack-inset.test.ts; what has to hold here is that
    the stack reads it rather than growing unbounded."""
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    stacks = provider.count("z-[9998] flex flex-col items-end gap-2")
    assert stacks, "the bottom-right overlay stack is gone"
    # Counted, not merely present: capping only one of the stacks is the bug here.
    assert provider.count("maxHeight: stack.maxHeight") == stacks, "every stack is capped"
    panel = (FRONTEND / "features/hub/download-manager/download-manager-panel.tsx").read_text(
        encoding = "utf-8"
    )
    # Both overlays scroll internally, so they can give up height.
    assert "flex min-h-0" in panel
    assert "flex min-h-0" in WEB_BANNER.read_text(encoding = "utf-8")


def test_the_desktop_stack_is_capped_like_the_browser_one():
    """The download panel shares the desktop stack, so the update card's own cap is
    not enough there either, and the desktop branch has been left uncapped before."""
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    assert provider.count("useStackGeometry()") == 2, "both stacks measure themselves"
    assert provider.count("maxHeight: stack.maxHeight") == 2, "both stacks are capped"
    assert "flex min-h-0" in TAURI_BANNER.read_text(encoding = "utf-8")


def test_the_stack_geometry_is_checked_numerically():
    """The cap is arithmetic now, not a class name, so the node test owns it. Named
    here so deleting it does not quietly leave the cap unchecked."""
    geometry = REPO / "studio/frontend/tests/monitor-stack-inset.test.ts"
    src = geometry.read_text(encoding = "utf-8")
    assert (
        "stackGeometry(null, W, H).maxHeight, H - 32" in src
    ), "nothing pins the no-obstacle cap to the 2rem the class used to spell"


def test_desktop_notes_are_looked_up_by_the_backend_version():
    """latest.json's `version` is the app SemVer while CHANGELOG.md is keyed by
    the backend release, so the desktop popup used to find no section at all
    and fall back to the updater's generic text."""
    workflow = (REPO / ".github/workflows/release-desktop.yml").read_text(encoding = "utf-8")
    assert "'pypi_version': os.environ['PYPI_VERSION']" in workflow
    assert "PYPI_VERSION: ${{ needs.prepare-version.outputs.pypi_version }}" in workflow
    rust = (REPO / "studio/src-tauri/src/desktop_update_policy.rs").read_text(encoding = "utf-8")
    assert "pypi_version: Option<String>" in rust
    hook = NOTES_HOOK.parent.joinpath("use-tauri-update.ts").read_text(encoding = "utf-8")
    # Both desktop paths carry it: the plugin exposes the raw metadata.
    assert "rawPypiVersion(update.rawJson)" in hook
    assert "manualUpdate.pypiVersion" in hook
    banner = TAURI_BANNER.read_text(encoding = "utf-8")
    assert "info?.pypiVersion ?? info?.version" in banner


def test_one_slow_read_cannot_outlast_the_fetch_budget(changelog_module):
    """The socket timeout is per operation, so slow headers followed by a slow
    body could hold a worker for twice the advertised deadline."""
    source = (BACKEND / "utils/changelog.py").read_text(encoding = "utf-8")
    assert "_limit_read(response, remaining)" in source
    assert "sock.settimeout(max(remaining, _CHANGELOG_MIN_READ_SECONDS))" in source


def test_a_heading_indented_into_a_list_item_is_not_a_release(changelog_module):
    """CommonMark keeps a heading at the item's content column inside the item.
    Treating it as a boundary truncated the real release and indexed a version
    that does not exist. Checked against markdown-it (commonmark preset)."""
    text = "## 1.0\n\n- Example:\n  ## 9.9.9\n\n- after\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0"]
    body = changelog_module.find_release_notes(text, "1.0").body
    assert "9.9.9" in body and "after" in body
    # One space short of the content column, the list ends and it is a release.
    left = "## 1.0\n\n- Example:\n ## 2.0\n"
    assert [e.version for e in changelog_module.parse_changelog(left)] == ["1.0", "2.0"]


def test_a_closed_list_stops_holding_headings(changelog_module):
    """Only an open item nests a heading, so a dedented paragraph, heading,
    break or fence hands the following indentation back to the document."""

    def versions(text):
        return [e.version for e in changelog_module.parse_changelog(text)]

    assert versions("## 1.0\n\n- Example:\n\nText.\n\n  ## 2.0\n") == ["1.0", "2.0"]
    assert versions("## 1.0\n\n- Example:\n## 2.0\n  ## 3.0\n") == ["1.0", "2.0", "3.0"]
    assert versions("## 1.0\n\n- Example:\n  Text.\n---\n  ## 2.0\n") == ["1.0", "2.0"]
    assert versions("## 1.0\n\n- Example:\n```\n```\n  ## 2.0\n") == ["1.0", "2.0"]
    # An item may begin with one blank line; content after that is outside it.
    assert versions("## 1.0\n\n-\n\n  ## 2.0\n") == ["1.0", "2.0"]


def test_a_version_line_is_not_an_ordered_list_marker(changelog_module):
    """`2.` needs whitespace after it to be a marker, or list tracking would
    read every setext version as a list item and lose the heading."""
    text = "2.0\n---\n\n- new\n\n1.0\n---\n\n- old\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["2.0", "1.0"]
    # An ordered item interrupts a paragraph only when it starts at 1.
    assert [
        e.version for e in changelog_module.parse_changelog("## 1.0\n\nText.\n9) one\n   ## 2.0\n")
    ] == ["1.0", "2.0"]


def test_a_wrapped_setext_heading_is_still_a_release(changelog_module):
    """CommonMark promotes the whole paragraph, so a heading that wraps keeps
    the version in its first token. Reading only the last line left the release
    unindexed and its notes unreachable."""
    text = "2026.7.5 - Release\nJuly 25\n---\n\n- note\n"
    entries = changelog_module.parse_changelog(text)
    assert [e.version for e in entries] == ["2026.7.5"]
    # The heading lines are the heading, not the body.
    assert entries[0].body == "- note"
    assert "July 25" not in entries[0].body


def test_a_lowercase_declaration_is_not_a_raw_block(changelog_module):
    """Only `<!` plus an uppercase letter opens one, so prose that mentions
    `<!note` must not hide every release under it."""
    assert [
        e.version for e in changelog_module.parse_changelog("<!note\n\n## 1.0\n\n- real\n")
    ] == ["1.0"]
    # A real declaration still hides its own block.
    assert [
        e.version for e in changelog_module.parse_changelog("<!DOCTYPE\n## 9.9.9\n>\n\n## 1.0\n")
    ] == ["1.0"]
    # The collapsed preview needs the same rule or it drops visible bullets.
    assert "<![A-Z]" in PREVIEW.read_text(encoding = "utf-8")


def test_link_resolver_reads_html_containers_the_way_the_others_do():
    """A `<details>` or `<div>` with no blank line inside is a type 6 block, so
    its contents render literally. Rewriting a link there mutates text the
    reader sees verbatim, and a fence inside such a block was being taken for a
    real fence, which stopped every link below it from resolving at all. The
    backend parser and the collapsed preview already apply the type 6 and 7
    rules, so the resolver has to share them or the three disagree on the same
    notes."""
    links = LINKS.read_text(encoding = "utf-8")
    for source in (PREVIEW, LINKS):
        text = source.read_text(encoding = "utf-8")
        assert "HTML_BLOCK_TAGS" in text and "HTML_TAG_ONLY_LINE" in text
    # A blank line ends the block, not the closing tag, and a bare quote marker counts as blank.
    assert "inHtmlBlock = !!container.trim()" in links
    # Type 7 cannot interrupt a paragraph, so prose above it keeps its links.
    assert "return !afterParagraph && HTML_TAG_ONLY_LINE.test(line);" in links


def test_an_escaped_mark_makes_an_image_a_link():
    """`\\![alt](path)` renders as a link, so it resolves to the file's page on
    GitHub rather than to the raw-content host."""
    links = LINKS.read_text(encoding = "utf-8")
    assert 'const image = bang === "!" && !isEscaped(line, offset);' in links
    # The reference pre-scan has to skip it too, or the definition flips host.
    assert "isEscaped(line, match.index)" in links


def test_only_markdown_line_endings_split_the_changelog(changelog_module):
    """str.splitlines also breaks on U+2028, U+2029, NEL, vertical tab and form
    feed, none of which end a line in CommonMark. A separator sitting in prose
    ahead of "## 9.9.9" made the parser index a release the renderer never shows
    and truncate the notes above it."""
    text = "## 2.0\n\nnote with a separator  ## 9.9.9\n\n## 1.0\n\n- old\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["2.0", "1.0"]
    # The prose stays whole rather than being cut at the separator.
    entry = changelog_module.find_release_notes(text, "2.0")
    assert entry is not None and "9.9.9" in entry.body
    for separator in (" ", "\x85", "\x0b", "\x0c"):
        broken = f"## 2.0\n\nnote{separator}## 9.9.9\n\n## 1.0\n\n- old\n"
        assert [e.version for e in changelog_module.parse_changelog(broken)] == ["2.0", "1.0"]
    # The three real line endings still split.
    for ending in ("\n", "\r\n", "\r"):
        real = f"## 2.0{ending}{ending}- new{ending}{ending}## 1.0{ending}{ending}- old{ending}"
        assert [e.version for e in changelog_module.parse_changelog(real)] == ["2.0", "1.0"]


def test_the_build_does_not_require_a_writable_source_tree():
    """A PEP 517 build may run against an immutable checkout (Nix, Bazel, a
    read-only container mount). Writing the snapshot beside the sources raised
    PermissionError before build_py started, so no wheel could be built at all.
    """
    src = (REPO / "_changelog_build.py").read_text(encoding="utf-8")
    # The source-tree copy is best effort.
    assert "except OSError:" in src
    # The wheel gets its copy from the staging directory either way.
    assert 'Path(self.build_lib) / "studio" / "CHANGELOG.md"' in src


def test_link_resolver_reads_comments_before_fences():
    """A fence delimiter hidden inside an HTML comment is not a fence. Reading
    it as one left the fence open, so every visible line below was classified as
    code and none of its links were resolved, which is far worse than the
    mutated-text case: the whole rest of the notes silently stops working. The
    order matters both ways, so a comment opener inside a real fence is not a
    comment either."""
    links = LINKS.read_text(encoding="utf-8")
    # Fence state is read before comments are masked, the order the collapsed preview uses.
    assert "const fenceSource = inComment\n      ? null\n      : FENCE.exec(" in links
    # Masking happens only after the in-fence early return.
    fence_return = links.index("// Fenced content is literal")
    assert links.index("const [line, stillInComment, stillRunOn] = maskComments(") > fence_return
    # Commented ranges join the code spans, so a hidden link is left alone.
    assert "const spans = [...codeSpans(masked), ...comments].sort(" in links


def test_preview_heading_and_quote_markers_follow_the_backend_rule():
    """An ATX heading needs an ASCII space, a tab or the end of the line after
    the marker, which is what _HEADING_PATTERN requires; `\\s` also matches a
    non-breaking space, so prose beginning "## Important change" with one was
    read as a heading and dropped, leaving a prose-only release with no
    collapsed preview at all. A blockquote marker takes at most three leading
    spaces for the same reason every other marker here does: accepting any run
    let an indented code sample containing "> - sample output" shed its
    indentation and be shown as the summary."""
    src = PREVIEW.read_text(encoding="utf-8")
    assert "const HEADING = /^#{1,6}(?:[ \\t]|$)/;" in src
    assert "const HEADING_LINE = /^ {0,3}#{1,6}(?:[ \\t]|$)/;" in src
    assert "const BLOCKQUOTE = /^ {0,3}>[ \\t]?/;" in src
    # The backend rule this mirrors.
    backend = (BACKEND / "utils" / "changelog.py").read_text(encoding="utf-8")
    assert "^ {0,3}##(?:[ \\t]+(?P<title>.*?))?[ \\t]*$" in backend


def test_preview_collects_labels_only_from_real_definitions():
    """A definition-shaped line inside an indented code block or a deep fence is
    literal text, so CommonMark leaves a later "[Beta] support" unresolved with
    its brackets showing. Recording the label anyway made toPlainText strip them
    in the collapsed preview, so it disagreed with the expanded view. The
    pre-scan skips the same code the collector pass skips; a real definition
    takes at most three spaces of indentation, so the indent test cannot reject
    one."""
    src = PREVIEW.read_text(encoding="utf-8")
    scan = src.index("const labels = new Set<string>();")
    collect = src.index("let deepFence: string | null = null;")
    prescan = " ".join(src[scan:collect].split())
    assert "let labelFence: string | null = null;" in prescan
    assert "if (line.indent - line.column >= INDENTED_CODE_INDENT) { continue; }" in prescan
    assert "endsDeepFence(labelFence, labelColumn, line)" in prescan


def test_an_html_block_to_the_left_of_a_list_item_closes_it(changelog_module):
    """Types 1 to 6 interrupt a paragraph, so an unindented <div> after "- item"
    closes the item and a following one-to-three-space-indented "## 2.0" is a
    real document heading. It was read as a lazy paragraph continuation, so the
    item stayed open and the release below the block was swallowed."""
    text = "## 3.0\n\n- item\n<div>\nhidden\n</div>\n\n  ## 2.0\n\n- two\n\n## 1.0\n\n- one\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["3.0", "2.0", "1.0"]
    # Without the block the heading really is nested, so it stays suppressed.
    nested = "## 3.0\n\n- item\n\n  ## 2.0\n\n- two\n\n## 1.0\n\n- one\n"
    assert [e.version for e in changelog_module.parse_changelog(nested)] == ["3.0", "1.0"]
    # Ordinary lazy continuation is untouched.
    lazy = "## 3.0\n\n- item\ncontinued\n\n  ## 2.0\n\n## 1.0\n\n- one\n"
    assert [e.version for e in changelog_module.parse_changelog(lazy)] == ["3.0", "1.0"]


def test_the_download_panel_can_shrink_inside_the_capped_stack():
    """The bottom-right stack is capped to the viewport, and a flex item defaults
    to min-height:auto, so this wrapper could not shrink below its own content.
    On a short viewport the cap was then absorbed by the update card, whose
    header and actions are fixed, rather than by the download list, which
    scrolls. Only the shared-stack branch needs it; standalone is positioned
    fixed and is not a flex item at all."""
    panel = (FRONTEND / "features/hub/download-manager/download-manager-panel.tsx").read_text(
        encoding="utf-8"
    )
    assert 'positioned ? "fixed bottom-4 right-4 z-50" : "flex min-h-0 justify-end"' in panel
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding="utf-8")
    stacks = provider.count("z-[9998] flex flex-col items-end gap-2")
    assert provider.count("maxHeight: stack.maxHeight") == stacks, "the cap this has to absorb"


@pytest.fixture(scope="module")
def run_scanner(tmp_path_factory):
    """Run the frontend's markdown scanners under node.

    Their job is to classify a line the way a CommonMark renderer would, which
    only a real run can show. The sources are copied with their "@/lib" aliases
    rewritten, because that alias resolves through Vite and not through node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is needed to run the TypeScript scanners")
    work = tmp_path_factory.mktemp("release-notes-scanners")
    for source in (PREVIEW, CODE_SPANS, LINKS, LIST_COLUMNS, INLINE_COMMENTS):
        rewritten = _TS_ALIAS.sub(r'"./\1.ts"', source.read_text(encoding="utf-8"))
        (work / source.name).write_text(rewritten, encoding="utf-8")
    (work / "run.ts").write_text(_TS_RUNNER, encoding="utf-8")

    def run(kind: str, markdown: str):
        result = subprocess.run(
            [node, "--experimental-strip-types", "--no-warnings", str(work / "run.ts"), kind],
            input=markdown,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"node could not run the scanners: {result.stderr.strip()[:200]}")
        return json.loads(result.stdout)

    return run


def preview_leads(preview) -> list[str]:
    return [item["lead"] for item in preview["items"]]


def test_a_link_indented_under_a_bullet_still_resolves(run_scanner):
    """CommonMark measures indentation from the container, not the margin
    (spec 0.31.2 section 5.2, list items). Under "- Details:" the content column
    is 2, so a four-space line is only two columns in: a paragraph holding a
    link, which GitHub renders and follows. The scanner measured from the margin
    instead, called it an indented code block (section 4.4) and left the
    destination relative, so the link resolved against Studio's own origin."""
    resolved = run_scanner("links", "- Details:\n\n    [guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in resolved
    # The same prose one column further in really is code, and stays untouched.
    code = run_scanner("links", "- Added.\n\n      [guide](docs/a.md)\n")
    assert "[guide](docs/a.md)" in code and "github.com" not in code
    # At document level four spaces is code, so that link is still left alone.
    top = run_scanner("links", "Intro.\n\n    [guide](docs/a.md)\n")
    assert "[guide](docs/a.md)" in top and "github.com" not in top


def test_an_indented_fence_does_not_swallow_the_bullets_below_it(run_scanner):
    """A four-space line at document level is an indented code block, and a
    top-level bullet is not indented enough to continue it, so the block ends
    and the list renders. Promoting the line to a list-contained fence left a
    block open with no closer, so every bullet after it was skipped and the
    collapsed popup lost its summary."""
    swallowed = "Example:\n\n    ```\n\n- Added the exporter\n- Fixed the crash\n"
    assert preview_leads(run_scanner("preview", swallowed)) == [
        "Added the exporter",
        "Fixed the crash",
    ]
    # With nothing else to fall back on the summary disappeared entirely.
    assert preview_leads(run_scanner("preview", "    ```\n\n- Added the exporter\n")) == [
        "Added the exporter"
    ]
    # A fence that really is inside an item still hides that item's code.
    nested = "- a\n  - b\n    ```\n    - not a bullet\n    ```\n\n- Added tests\n"
    assert preview_leads(run_scanner("preview", nested)) == ["a", "Added tests"]


def test_a_table_only_release_previews_as_nothing(run_scanner):
    """A release written as a GFM table renders as a grid, and the panel treats
    notes that preview as nothing by staying collapsed rather than showing an
    empty strip. Falling through to the prose collector put the raw
    "| Change | Detail | | --- | --- |" delimiters in the popup instead."""
    table = "| Change | Detail |\n| --- | --- |\n| Exporter | Added GGUF |\n"
    assert run_scanner("preview", table)["items"] == []
    # A table after prose is dropped too, rather than joined onto it.
    assert preview_leads(run_scanner("preview", f"Some prose.\n\n{table}")) == ["Some prose."]
    # A bullet right after the rows ends the table, so it still previews.
    assert preview_leads(run_scanner("preview", f"{table}- Added tests\n")) == ["Added tests"]
    # Mismatched header and delimiter widths are no table, as on GitHub, so both lines are prose.
    assert preview_leads(run_scanner("preview", "| a | b |\n| --- |\n")) == ["| a | b | | --- |"]


def test_a_fence_inside_a_list_item_ends_with_the_item(changelog_module):
    """A fence is scoped to its container: with no closer it runs to the end of
    the containing block, not the document (spec 0.31.2 section 4.5). A
    dedented "## 2.0" closes the list item, so it is a real release heading.
    Document-wide fence state kept the block open and hid every release below
    it, so one missing closing line emptied the rest of the changelog."""
    text = "## 1.0\n\n- item\n  ```\n\n## 2.0\n\n- two\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0", "2.0"]
    # A fence at document level still runs to the end of the file.
    top = "## 1.0\n\n```\n\n## 2.0\n\n- two\n"
    assert [e.version for e in changelog_module.parse_changelog(top)] == ["1.0"]
    # A closed fence inside an item is unaffected, and its sample stays hidden.
    closed = "## 1.0\n\n- Run:\n  ```bash\n  ## 9.9.9\n  ```\n\n## 2.0\n\n- two\n"
    assert [e.version for e in changelog_module.parse_changelog(closed)] == ["1.0", "2.0"]
    # Content dedented out of the item ends the item and the fence with it.
    assert changelog_module.find_release_notes(text, "2.0").body == "- two"


def test_stripping_comments_stays_linear_in_the_code_spans(changelog_module):
    """The comment scanner restarted its code-span search at the first span for
    every opener, so a line of N spans and N openers cost N squared. A 203 KiB
    line is well inside the 2 MiB the fetcher accepts, and notes are reparsed on
    every request, so one such line held a worker for over ten seconds."""
    line = "`a` <!--x--> " * 16_000
    assert len(line) < changelog_module.CHANGELOG_MAX_BYTES
    started = time.monotonic()
    visible, in_comment = changelog_module._strip_comments(line, False, False)
    elapsed = time.monotonic() - started
    # Roughly 40ms scanning forward against roughly 11s restarting each time.
    assert elapsed < 2.0, f"comment stripping took {elapsed:.1f}s"
    # Same result as before: the spans survive and the comments are gone.
    assert in_comment is False
    assert "<!--" not in visible and visible.count("`a`") == 16_000


def test_the_three_scanners_share_one_list_column_rule():
    """The parser and both frontend scanners have to classify a line the same
    way, and drifting apart on indentation is what put a paragraph link inside a
    code block. The frontend pair reads its list columns from one module, ported
    from the backend's own tracker."""
    shared = LIST_COLUMNS.read_text(encoding="utf-8")
    assert "export function openLists(" in shared
    assert "_open_lists" in shared, "the backend function this mirrors"
    for source in (PREVIEW, LINKS):
        src = source.read_text(encoding="utf-8")
        assert 'from "@/lib/markdown-list-columns"' in src
        assert "openLists(" in src
    # Both sides measure indented code from the container, not from the margin.
    backend = (BACKEND / "utils" / "changelog.py").read_text(encoding="utf-8")
    assert "_indent_width(visible) - column >= 4" in backend
    assert "indentWidth(structure) - column >= INDENTED_CODE_INDENT" in LINKS.read_text(
        encoding="utf-8"
    )


def test_a_failed_fetch_keeps_retry_reachable():
    """The fallback stands in for "no section for this version", which the hook
    reports as ready. A failed fetch is reported as error and is retryable, and on
    desktop the fallback is the updater's static install blurb, so taking it there
    replaced the Retry button with generic text until the cache expired."""
    src = " ".join(PANEL.read_text(encoding="utf-8").split())
    assert 'notes?.matched ? notes.markdown : state === "error" ? null' in src
    # Only NotesStatus renders retry, in the else of the markdown branch: an error has no markdown.
    assert "{markdown ? (" in src
    assert "retry={retry}" in src

    hook = " ".join(
        (FRONTEND / "hooks" / "use-release-notes.ts").read_text(encoding="utf-8").split()
    )
    assert (
        "const failed = !next || (!next.matched && next.error !== null);" in hook
    ), "the distinction this relies on"


def test_an_unclosed_comment_in_prose_cannot_hide_later_links(run_scanner):
    """CommonMark opens an HTML block (spec 0.31.2 section 4.6, type 2) only
    when the line itself begins with `<!--`; one written mid-sentence is inline
    raw HTML and cannot outlive the block it sits in. The link resolver carried
    the unclosed state to every following line instead, so a note that merely
    mentions the delimiter masked the relative links under it and they resolved
    against Studio's own origin."""
    repo = "https://github.com/unslothai/unsloth/blob/main/docs/a.md"
    # A separate list item is a separate block, so the link below still renders.
    item = run_scanner("links", "- Type <!-- to begin a comment\n- See [docs](docs/a.md)\n")
    assert repo in item
    # So does a paragraph the blank line already ended.
    paragraph = run_scanner("links", "Type <!-- to begin\n\nSee [docs](docs/a.md)\n")
    assert repo in paragraph
    # A delimiter inside inline code is literal, as it is for the parser.
    spanned = run_scanner("links", "- Wrap in `<!--` and `-->`\n- See [docs](docs/a.md)\n")
    assert repo in spanned
    # A comment starting a line is a block: it hides down to the closer's line, that line included.
    block = run_scanner("links", "<!-- staged\n- See [docs](docs/a.md)\n-->\n")
    assert repo not in block
    closer = run_scanner("links", "<!-- staged\n--> See [docs](docs/a.md)\n")
    assert repo not in closer


def test_a_bare_level_two_marker_ends_the_release(changelog_module, run_scanner):
    """An ATX heading's opening sequence may be followed by the end of the line
    (spec 0.31.2 section 4.2), so a bare `##` is an empty level-two heading. The
    scanners required whitespace after the hashes, so everything below such a
    line stayed inside the release above it and the popup showed unrelated notes
    under that version."""
    text = "## 2.0\n\n- new thing\n\n##\n\n- SECRET: not part of 2.0\n"
    entry = changelog_module.find_release_notes(text, "2.0")
    assert "new thing" in entry.body
    assert "SECRET" not in entry.body
    # An empty heading has no version, so it ends a release without indexing one.
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["2.0"]
    # Prose still needs a space or a tab: `##x` is a paragraph, not a heading.
    prose = "## 2.0\n\n- new thing\n\n##x\n\n- still 2.0\n"
    assert "still 2.0" in changelog_module.find_release_notes(prose, "2.0").body
    # The preview agrees: an empty heading renders as nothing, so it ends the bullet.
    preview = run_scanner("preview", "- new thing\n##\nUnrelated scratch notes\n")
    assert preview_leads(preview) == ["new thing"]


def test_a_comment_between_bullets_closes_the_list(changelog_module, run_scanner):
    """A comment is an HTML block (spec 0.31.2 section 4.6, type 2), so one
    written at the margin under a bullet is not indented enough to continue that
    item and closes the list. The scanners blanked the line before list tracking
    saw it, which reads as a blank line and leaves the item open, so the release
    heading below it looked like nested item content and the new release was
    merged into the one above."""
    text = "## 1.0\n\n- old item\n<!-- separator -->\n  ## 2.0\n\n- new item\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0", "2.0"]
    assert "new item" not in changelog_module.find_release_notes(text, "1.0").body
    assert "new item" in changelog_module.find_release_notes(text, "2.0").body
    # At the item's content column the comment stays inside it, so the heading under it is nested.
    nested = "## 1.0\n\n- old item\n  <!-- separator -->\n  ## 2.0\n\n- new item\n"
    assert [e.version for e in changelog_module.parse_changelog(nested)] == ["1.0"]
    # The link resolver reads the same column: list closed, four spaces is code, left untouched.
    code = run_scanner("links", "- old item\n<!-- separator -->\n    [guide](docs/a.md)\n")
    assert "[guide](docs/a.md)" in code and "github.com" not in code
    # Inside the item those four spaces are two columns in, so it is prose and the link resolves.
    prose = run_scanner("links", "- old item\n  <!-- separator -->\n    [guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in prose
    # The preview agrees: the fence is indented code, not a fence swallowing the bullet below.
    preview = run_scanner(
        "preview",
        "- Details:\n<!-- separator -->\n    ```\n    - hidden sample\n- Real second item\n",
    )
    assert preview_leads(preview) == ["Details:", "Real second item"]


def test_a_parenthesised_link_destination_still_resolves(run_scanner):
    """A destination may hold parentheses while they balance (spec 0.31.2
    section 6.3), so `[x]((draft).md)` points at `(draft).md`. The resolver's
    destination expression stopped at the first paren, matched an empty
    destination and left the markdown alone, so the link resolved against
    Studio's own origin instead of the repository."""
    leading = run_scanner("links", "[details]((draft).md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/(draft).md" in leading
    # An image resolves against the raw host the same way.
    image = run_scanner("links", "![shield]((badge).png)\n")
    assert "https://raw.githubusercontent.com/unslothai/unsloth/main/(badge).png" in image
    # A pair in the middle of a path balances too.
    middle = run_scanner("links", "[api](docs/(v2)/api.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/(v2)/api.md" in middle
    # An unbalanced paren makes the destination invalid, so `[x](a(b.md)` is plain text, not a link.
    unbalanced = run_scanner("links", "[x](a(b.md)\n")
    assert unbalanced == "[x](a(b.md)\n"
    # One more closer balances the pair, and then it is a link again.
    closed = run_scanner("links", "[x](a(b.md))\n")
    assert "https://github.com/unslothai/unsloth/blob/main/a(b.md)" in closed
    # Pairs nest, and one level was all the expression allowed, so a path with two stayed relative.
    nested = run_scanner("links", "[x](((draft)).md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/((draft)).md" in nested
    deep = run_scanner("links", "![shot](((((v2))))).png)\n")
    assert "https://raw.githubusercontent.com/unslothai/unsloth/main/((((v2))))" in deep
    # The closer must still be there: an unbalanced run below a nested pair is not a link.
    across = run_scanner("links", "[x](((a).md\n[y](docs/y.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/y.md" in across
    assert "[x](((a).md" in across


def test_a_fence_inside_a_container_still_hides_its_sample(run_scanner):
    """A fence is measured from its container and not from the margin (spec
    0.31.2 section 4.5), so `> ~~~` and a fence three columns under a nested
    bullet open one. Reading the margin instead never saw them, so the sample
    inside was treated as prose and a relative link written in a code block was
    rewritten into the text the reader sees verbatim."""
    quoted = run_scanner("links", "> ~~~\n> [guide](docs/a.md)\n> ~~~\n")
    assert "[guide](docs/a.md)" in quoted and "github.com" not in quoted
    nested = run_scanner("links", "- a\n  - b\n    ~~~\n    [x](docs/x.md)\n    ~~~\n")
    assert "[x](docs/x.md)" in nested and "github.com" not in nested
    # A longer closer is still a closer, so the pair is not something a code span hid.
    uneven = run_scanner("links", "> ```\n> [guide](docs/a.md)\n> ````\n")
    assert "[guide](docs/a.md)" in uneven and "github.com" not in uneven
    # The fence ends with its container: a line outside the quote, or left of the item, is Markdown.
    left = run_scanner("links", "> ~~~\n[guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in left
    dedented = run_scanner("links", "- a\n  ~~~\n[guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in dedented
    # A document-level fence owns the quoted lines below, so the marker does not undo it.
    document = run_scanner("links", "~~~\n> [guide](docs/a.md)\n~~~\n")
    assert "[guide](docs/a.md)" in document and "github.com" not in document
    # Four columns past the item's content column it is indented code, not a fence: still literal.
    code = run_scanner("links", "- Details:\n\n      ~~~\n      [guide](docs/a.md)\n")
    assert "[guide](docs/a.md)" in code and "github.com" not in code


def test_an_html_block_inside_a_container_is_literal_too(run_scanner):
    """Type 1 and type 6 blocks are measured from their container the same way,
    so a `<details>` under a nested bullet and a `<pre>` inside a quote both
    show their contents verbatim. Missing the opener treated the body as
    Markdown and rewrote the literal examples in it."""
    nested = run_scanner("links", "- a\n  - b\n    <details>\n    [x](docs/x.md)\n    </details>\n")
    assert "[x](docs/x.md)" in nested and "github.com" not in nested
    quoted = run_scanner("links", "> <pre>\n> [x](docs/x.md)\n> </pre>\n")
    assert "[x](docs/x.md)" in quoted and "github.com" not in quoted
    # The block ends with its container, so a line dedented out of the item is Markdown again.
    dedented = run_scanner("links", "- a\n  - b\n    <details>\n[x](docs/x.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/x.md" in dedented
    # Inside a quote a bare marker holds nothing, the blank line that ends a type 6 block.
    blank = run_scanner("links", "> <details>\n>\n> [x](docs/x.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/x.md" in blank


def test_an_underline_left_of_an_item_is_lazy_text_of_it(changelog_module, run_scanner):
    """A setext underline may never be a lazy continuation line (spec 0.31.2
    section 4.3), so `===` written left of an open list item is read as more of
    the item's paragraph rather than as a block that closes it. Rejecting every
    underline-shaped line ended the list there, which promoted the nested
    "## 2.0" below it to a document-level heading and indexed a release the
    renderer never shows."""
    nested = "## 1.0\n- old note\n===\n  ## 2.0\n- new\n"
    assert [e.version for e in changelog_module.parse_changelog(nested)] == ["1.0"]
    # A row of dashes is a thematic break, closing the item, so the heading is the next release.
    broken = "## 1.0\n- old note\n---\n  ## 2.0\n"
    assert [e.version for e in changelog_module.parse_changelog(broken)] == ["1.0", "2.0"]
    # With no paragraph above it the underline opens one, so the blank line closes the item.
    apart = "## 1.0\n- old note\n\n===\n  ## 2.0\n"
    assert [e.version for e in changelog_module.parse_changelog(apart)] == ["1.0", "2.0"]
    # The link scanner keeps the item open, so the four-space line is a paragraph and resolves.
    resolved = run_scanner("links", "- Details:\n===\n\n    [guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in resolved


def test_a_quote_keeps_its_paragraph_to_itself(changelog_module, run_scanner):
    """Lazy continuation runs the other way too: a marker written outside a
    blockquote is not text of the quote's paragraph, so `2. item` under
    `> quote` opens a list even though an ordered marker past 1 may not
    interrupt a paragraph (spec 0.31.2 section 5.2). Lending the quote's
    paragraph to the document left the list closed, so the heading indented to
    the item's content column read as a release of its own."""
    quoted = "## 1.0\n> quote\n2. item\n   ## 2.0\n- new\n"
    assert [e.version for e in changelog_module.parse_changelog(quoted)] == ["1.0"]
    # A quote holding a heading leaves no paragraph, nor does an empty one, so the list opens.
    heading = "## 1.0\n> # inner\n2. item\n   ## 2.0\n"
    assert [e.version for e in changelog_module.parse_changelog(heading)] == ["1.0"]
    # An unquoted line the quote's paragraph swallows keeps it open, the marker still outside.
    lazy = "## 1.0\n> quote\ntext\n2. item\n   ## 2.0\n"
    assert [e.version for e in changelog_module.parse_changelog(lazy)] == ["1.0"]
    # Under an ordinary paragraph the marker is its text, so no list opens and the heading is real.
    prose = "## 1.0\nprose\n2. item\n   ## 2.0\n"
    assert [e.version for e in changelog_module.parse_changelog(prose)] == ["1.0", "2.0"]
    # The preview reads the marker as a bullet for the same reason.
    assert preview_leads(run_scanner("preview", "> quote\n2. item\n")) == ["item"]


def test_indented_code_before_an_ordered_marker_still_opens_a_list(changelog_module):
    """An indented code block ends at the first line that is not indented enough
    to continue it, and no paragraph is open for the marker below to continue,
    so `2. item` opens a list whatever its start number. Reading it as text of
    the code block instead would leave the list closed and index the heading at
    the item's content column as a release."""
    joined = "## 1.0\n\n    code\n2. item\n   ## 2.0\n- new\n"
    assert [e.version for e in changelog_module.parse_changelog(joined)] == ["1.0"]
    # A blank line between the two changes nothing: the list opens either way.
    apart = "## 1.0\n\n    code\n\n2. item\n   ## 2.0\n- new\n"
    assert [e.version for e in changelog_module.parse_changelog(apart)] == ["1.0"]
    # Four columns past its container the marker is code, so no list opens and the heading stands.
    inside = "## 1.0\n\n    code\n    - item\n  ## 2.0\n"
    assert [e.version for e in changelog_module.parse_changelog(inside)] == ["1.0", "2.0"]


def test_a_fence_written_as_an_item_first_content_opens_in_that_item(run_scanner):
    """A block written straight after a list marker is the item's own first
    content, measured from the column that content starts (spec 0.31.2 section
    5.2), so "- ```md" opens a fence. Reading the whole line instead never saw
    one, so the code sample below it was treated as prose: the resolver rewrote
    a destination the reader sees verbatim, and the preview offered the info
    string as a headline bullet."""
    sample = run_scanner("links", "- ```md\n  [example](docs/a.md)\n  ```\n")
    assert "[example](docs/a.md)" in sample and "github.com" not in sample
    ordered = run_scanner("links", "1. ~~~\n   [example](docs/a.md)\n   ~~~\n")
    assert "[example](docs/a.md)" in ordered and "github.com" not in ordered
    # The preview agrees: an item of only a code block previews as nothing; the next is a bullet.
    preview = run_scanner("preview", "- ```md\n  sample text\n  ```\n- Added tests\n")
    assert preview_leads(preview) == ["Added tests"]
    # One column further in it is indented code inside the item, so the link is prose and resolves.
    padded = run_scanner("links", "-     ```\n  [example](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in padded
    # A marker the paragraph above swallows opens no item, so no fence: ordered items open at 1.
    lazy = run_scanner("links", "Intro.\n2. ```\n[guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in lazy


def test_an_html_block_ends_with_the_item_it_was_written_in(changelog_module, run_scanner):
    """An HTML block holds no lazy continuation line, so one opened on a list
    item's continuation line ends where the item does, exactly as a fence there
    does. Ending it only on a blank line let it run past the item and swallow
    the next release heading, so those notes could never be found, and the
    collapsed preview lost every bullet below it."""
    text = "## 1.0\n\n- item\n\n  <div>\n## 2.0\n\n- new thing\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0", "2.0"]
    assert "new thing" in changelog_module.find_release_notes(text, "2.0").body
    # A raw block such as <pre> is scoped the same way.
    raw = "## 1.0\n\n- item\n\n  <pre>\n## 2.0\n\n- new thing\n"
    assert [e.version for e in changelog_module.parse_changelog(raw)] == ["1.0", "2.0"]
    # At the item's content column the block holds the heading, which is nested and indexes nothing.
    nested = "## 1.0\n\n- item\n\n  <div>\n  ## 2.0\n"
    assert [e.version for e in changelog_module.parse_changelog(nested)] == ["1.0"]
    # The preview reads it the same way: the bullet below the block is a bullet.
    preview = run_scanner("preview", "- item\n\n  <div>\n- Added tests\n")
    assert preview_leads(preview) == ["item", "Added tests"]
    # An opener straight after a marker opens in that item, so the dedented heading is a release.
    marked = "## 1.0\n\n- <div>\n## 2.0\n\n- new thing\n"
    assert [e.version for e in changelog_module.parse_changelog(marked)] == ["1.0", "2.0"]


def test_a_comment_may_close_on_a_later_line_of_its_paragraph(run_scanner):
    """A comment written mid-sentence is inline raw HTML belonging to the
    paragraph around it, so its `-->` may arrive on a later line of that same
    paragraph and everything between renders as nothing. Ending the comment at
    its own line left a backtick inside it pairing with a real one below, which
    hid a following link from the resolver, and left the collapsed preview
    quoting text the popup body does not show."""
    carried = run_scanner("links", "Note <!-- ` open\nstill --> see [d](docs/a.md) and `x`\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in carried
    # Text inside the comment renders as nothing, so it is left alone.
    inside = run_scanner("links", "Note <!-- see [c](docs/c.md)\nmore --> end\n")
    assert "[c](docs/c.md)" in inside and "github.com" not in inside
    # The preview hides it too, rather than quoting the comment at the reader.
    preview = run_scanner(
        "preview", "- Added X <!-- TODO: rewrite\n  this properly -->\n- Second\n"
    )
    assert preview_leads(preview) == ["Added X", "Second"]
    # An opener cannot outlive its paragraph: with it closed the `<!--` is text and hides nothing.
    broken = run_scanner("links", "Note <!-- open\n\nsecret --> end [d](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in broken
    # A heading breaks into the paragraph, so it ends the comment's reach too.
    headed = run_scanner("links", "Note <!-- open\n## 2.0 --> end [d](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in headed
    assert preview_leads(run_scanner("preview", "Note <!-- open\n\n- Second\n")) == ["Second"]


def test_only_punctuation_is_escapable_in_a_link_destination(run_scanner):
    """CommonMark escapes ASCII punctuation and nothing else (spec 0.31.2
    section 2.4), so the backslash in `docs\\alpha.md` is a character of the
    path. Dropping every backslash rewrote it to a path that does not exist,
    and a URL parser reads what is left as a separator, so a Windows or
    namespaced path pointed at the wrong file either way."""
    kept = run_scanner("links", "[guide](docs\\alpha.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs%5Calpha.md" in kept
    # An escaped backslash is one literal backslash, which survives the same.
    escaped = run_scanner("links", "[guide](docs\\\\alpha.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs%5Calpha.md" in escaped
    # A real escape is still an escape: `\\(` is a paren of the path.
    paren = run_scanner("links", "[guide](a\\(b.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/a(b.md" in paren
    # A space still ends the destination, escaped or not, so there is no link.
    spaced = run_scanner("links", "[guide](a\\ b.md)\n")
    assert spaced == "[guide](a\\ b.md)\n"


def test_one_definition_does_not_hide_the_next(run_scanner):
    """Definitions may run consecutively (spec 0.31.2 section 4.7): a block of
    them is how a changelog collects its link targets. A definition is not
    paragraph text, so it opens no paragraph for the next one to be unable to
    interrupt. The resolver counted one as prose, which left every definition
    after the first outside the set of lines a definition may start on, so only
    the first was rewritten and the rest resolved against Studio's own origin.
    The backend already reads the line this way."""
    text = (
        "- AMD support is here, see [the AMD guide][amd] and the\n"
        "  [Intel notes][xpu].\n\n"
        "[amd]: docs/basics/amd.md\n"
        "[xpu]: docs/basics/xpu.md\n"
    )
    resolved = run_scanner("links", text)
    base = "https://github.com/unslothai/unsloth/blob/main/docs/basics/"
    assert f"[amd]: {base}amd.md" in resolved
    assert f"[xpu]: {base}xpu.md" in resolved
    # A run of them stays a run however long it is.
    run = run_scanner("links", "[a]: docs/a.md\n[b]: docs/b.md\n[c]: docs/c.md\n")
    assert run.count("https://github.com/unslothai/unsloth/blob/main/docs/") == 3
    # Prose between them opens a paragraph the next line may not interrupt, so it is not one.
    prose = run_scanner("links", "[a]: docs/a.md\nintro\n[b]: docs/b.md\n")
    assert "[b]: docs/b.md" in prose


def test_a_comment_closed_on_its_own_line_still_closes(run_scanner):
    """A multiline comment is ordinarily closed by a `-->` written on a line of
    its own, and a wrapped line may open with emphasis. The guard asking whether
    the closer is reachable read any line whose first character was punctuation
    as the start of a new block, so neither shape counted as more of the
    paragraph carrying the comment. The comment then never closed, and the
    collapsed popup showed the author's internal note to the user."""
    closer = run_scanner(
        "preview",
        "- DoRA training is available in Studio. <!-- TODO confirm the exact\n"
        "  flag name before release\n-->\n",
    )
    assert preview_leads(closer) == ["DoRA training is available in Studio."]
    # A continuation may open with emphasis, which is text and not a block.
    starred = run_scanner(
        "preview",
        "- DoRA training is available. <!-- TODO confirm the\n  *before* release -->\n",
    )
    assert preview_leads(starred) == ["DoRA training is available."]
    underscored = run_scanner(
        "preview",
        "- DoRA training is available. <!-- TODO confirm the\n  _draft_ note -->\n",
    )
    assert preview_leads(underscored) == ["DoRA training is available."]
    # A real block still ends the paragraph, so the opener below one is text and hides nothing.
    broken = run_scanner("links", "Note <!-- open\n## 2.0\nsecret --> [d](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in broken
    # So does a list item with content, which may interrupt a paragraph.
    item = run_scanner("links", "Note <!-- open\n- bullet\nsecret --> [d](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in item


def test_a_comment_written_as_an_item_first_content_is_a_block(changelog_module, run_scanner):
    """A comment is an HTML block (spec 0.31.2 section 4.6, type 2), so one
    written as a list item's first content opens inside that item, exactly as a
    fence written there does. The scanners looked for the opener at the margin
    of the line as written, so a marker in front of it hid the block: the
    resolver rewrote a destination inside raw HTML, which Streamdown then shows
    the reader as a literal URL, and the preview quoted the hidden note back at
    them as though the bullet were Markdown."""
    item = run_scanner("links", "- <!-- new --> AMD support, see [the guide](docs/amd.md)\n")
    assert item == "- <!-- new --> AMD support, see [the guide](docs/amd.md)\n"
    # Every marker opens an item, and a nested one is still an item.
    for text in (
        "* <!-- new --> see [the guide](docs/amd.md)\n",
        "1. <!-- new --> see [the guide](docs/amd.md)\n",
        "- outer\n  - <!-- new --> see [the guide](docs/amd.md)\n",
    ):
        assert "github.com" not in run_scanner("links", text)
    # The multiline form hides lines to the closer, as a comment at the item's content column did.
    multiline = run_scanner("links", "- <!-- hidden\n  [a](docs/x.md)\n  -->\n")
    assert "[a](docs/x.md)" in multiline and "github.com" not in multiline
    # Still scoped to the item it was written in, so a line dedented out of it ends the block.
    dedented = run_scanner("links", "- <!-- hidden\n[a](docs/x.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/x.md" in dedented
    # The preview agrees: an item of only the block previews as nothing; the next is a bullet.
    preview = run_scanner("preview", "- <!-- new --> hidden note\n- Real bullet\n")
    assert preview_leads(preview) == ["Real bullet"]
    # The parser agrees too: the item keeps its column, so a heading inside is nested, not indexed.
    text = "## 1.0\n\n- <!-- hidden\n\n  ## 2.0\n"
    assert [e.version for e in changelog_module.parse_changelog(text)] == ["1.0"]
