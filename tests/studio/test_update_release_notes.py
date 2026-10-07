# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Contracts for the update popup's release-notes preview.

The popup shows the newest GitHub release's announcement. Two risks are guarded:
showing a release the maintainers have not published, and showing a body's
generated sections as though they were the announcement."""

from __future__ import annotations

import http.server
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from growth import assert_linear  # tests/_shared, on sys.path via tests/conftest.py

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "studio/backend"
FRONTEND = REPO / "studio/frontend/src"
MODULE = BACKEND / "utils/release_notes.py"
BODIES = Path(__file__).parent / "fixtures/release_bodies"
PANEL = FRONTEND / "components/update/release-notes-panel.tsx"
NOTES_LAYOUT = FRONTEND / "components/update/update-notes-layout.ts"
NOTES_HOOK = FRONTEND / "hooks/use-release-notes.ts"
PREVIEW = FRONTEND / "lib/release-notes-preview.ts"
CODE_SPANS = FRONTEND / "lib/markdown-code-spans.ts"
LINKS = FRONTEND / "lib/release-body-links.ts"
LIST_COLUMNS = FRONTEND / "lib/markdown-list-columns.ts"
INLINE_COMMENTS = FRONTEND / "lib/markdown-inline-comments.ts"
WEB_BANNER = FRONTEND / "components/web/update-banner.tsx"
TAURI_BANNER = FRONTEND / "components/tauri/update-banner.tsx"


CARD_CAP = {
    WEB_BANNER: "max-h-[calc(100dvh_-_2rem)]",
    TAURI_BANNER: "max-h-[calc(100dvh_-_2rem_-_var(--studio-window-chrome-top,0px))]",
}


# An apostrophe in JSX text is prose, not a string delimiter; the frontend uses double quotes.
# A set, not a string: `"" in '"`'` is true for a substring.
_QUOTES = frozenset('"`')

_CALL = re.compile(r"[A-Za-z_$][\w$]*\(")

_IMPORTANT = re.compile(r"^!|!$")


def _split_variants(token: str) -> tuple[tuple[str, ...], str]:
    """A Tailwind class token as (variants, utility), split on top-level colons.

    Depth-aware, because an arbitrary value may carry its own brackets and its
    own colon: `has-[[data-slot=update-release-notes]]:min-h-[calc(...)]`.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in token:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        if char == ":" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    # `!min-h-0` and `min-h-0!` are `min-h-0` at a weight that beats the floor.
    return tuple(parts[:-1]), _IMPORTANT.sub("", parts[-1])


def _tokens(source: str) -> list[tuple[tuple[str, ...], str]]:
    """Every class token in `source`, as (variants, utility)."""
    return [_split_variants(token) for token in re.findall(r"""[^\s"'`]+""", source)]


def _applies(source: str, utility: str, *variants: str) -> bool:
    """Is `utility` written anywhere in `source` under at least `variants`?

    Token-wise, not as a substring. These assertions used to name a run of
    classes verbatim, so gating a rule (`max-[383px]:has-[...]:min-h-[calc]`)
    or inserting an unrelated one beside it read as the rule being gone, and
    #10229 went red on four of them while every rule it named was still there.
    A utility is the last top-level segment, so a longer utility that merely
    ends with this one does not count, and variant order is Tailwind's business
    rather than this test's.

    Name no variants and this asks whether the utility is there under any gate
    or none, which is what a check for a rule's *absence* wants. A check that a
    rule is in force needs `_only_under`.
    """
    for token_variants, token_utility in _tokens(source):
        if token_utility == utility and set(variants) <= set(token_variants):
            return True
    return False


def _only_under(source: str, utility: str, *variants: str) -> bool:
    """Is `utility` written at least once, and every time under exactly these?

    What a positive layout guarantee needs, and neither half of it is a subset
    test. An extra gate narrows when the rule is in force, so a floor written
    `md:has-[...]:min-h-[...]` leaves every width from 384px to the `md`
    breakpoint with none, the narrow override having stopped at 383px. A
    second, ungated copy widens it the other way, back to the reserved empty
    height the gate was added to stop. Both leave the utility present, so both
    pass an existence check.

    Named with no variants, this is "written, and never gated": an ungated
    `shrink-0` is the whole guarantee when there are no notes, and
    `max-[383px]:shrink-0` would satisfy an existence check while leaving the
    card squeezable at every width but one.
    """
    found = False
    for token_variants, token_utility in _tokens(source):
        if token_utility != utility:
            continue
        if set(token_variants) != set(variants):
            return False
        found = True
    return found


def _class_const(source: str, name: str) -> str:
    """The class string of an exported `const NAME = "..."`.

    Comments blanked and anchored on the `export`, like the banner extractors:
    an old declaration left commented out above the live one would otherwise
    answer for it, and a stale copy that still reads correctly is exactly how a
    regression in the live one goes unnoticed.
    """
    match = re.search(
        rf'export const {re.escape(name)}\s*=\s*\n?\s*"([^"]*)"', _without_comments(source)
    )
    assert match, f"{name} is not an exported class constant"
    return match.group(1)


_COMMENT_SPAN = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


# Shared with tests/studio/test_overlay_layering.py so both use the literal-aware scanner.
from jsx_tags import (  # noqa: E402
    opening_tag as _opening_tag,
    skip_literal as _skip_literal,
    without_comments as _without_comments,
)


def _class_on_testid(source: str, testid: str) -> str:
    """The literal class string of the element carrying `data-testid=testid`.

    Anchored on the attribute that names the element rather than on a run of
    its classes. An anchor built from classes cannot survive one of them being
    inserted or reordered, which is the failure this file is being fixed for,
    and it fails by raising rather than by reporting a missing rule.
    """
    clean = _without_comments(source)
    start, end = _opening_tag(clean, clean.index(f'data-testid="{testid}"'))
    return _class_value(clean[start:end], testid)


def _class_value(tag: str, what: str) -> str:
    """Every class named by the `className` of `tag`, joined.

    A literal today. Wrapping one in the `cn()` this file already uses renders
    the same DOM, so it has to read the same rather than being skipped, which
    would have taken the next element's classes instead and reported every rule
    on this one as missing.
    """
    key = "className="
    assert key in tag, f"{what} carries no className"
    at = tag.index(key) + len(key)
    if tag[at] == '"':
        return tag[at + 1 : _skip_literal(tag, at) - 1]
    assert tag[at] == "{", f"{what}'s className is neither a literal nor an expression"
    return _always_rendered(tag[at + 1 : _balanced(tag, at) - 1])


def _arguments(call: str) -> list[str]:
    """The arguments of a call, split on its top-level commas."""
    at = call.index("(")
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    index = at
    while index < len(call):
        char = call[index]
        if char in _QUOTES:
            end = _skip_literal(call, index)
            current.append(call[index:end])
            index = end
            continue
        if char in "([{":
            depth += 1
            if depth == 1:
                index += 1
                continue
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                break
        elif char == "," and depth == 1:
            parts.append("".join(current))
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    parts.append("".join(current))
    return parts


def _always_rendered(expression: str) -> str:
    """The classes `expression` renders in every state, joined.

    Only an argument that is a bare literal. `cn(open && "x")` renders `x`
    sometimes and a rule the card must always carry is not satisfied by
    sometimes; written against a constant, `cn(false && "x")` renders it never,
    while the text of the class sits there in the file either way. Reading the
    literals out of the whole expression would call all three the same.
    """
    text = expression.strip()
    if text[:1] in _QUOTES and _skip_literal(text, 0) == len(text):
        return text[1:-1]
    if "(" not in text:
        return ""
    literals = []
    for argument in _arguments(text):
        part = argument.strip()
        if part[:1] in _QUOTES and _skip_literal(part, 0) == len(part):
            literals.append(part[1:-1])
        elif _CALL.match(part) and _balanced(part, part.index("(")) == len(part):
            # A nested `cn()` renders the same classes, so it must read the same.
            literals.append(_always_rendered(part))
    return " ".join(literals)


def _balanced(source: str, at: int) -> int:
    """The index just past the bracket group opening at `at`."""
    depth = 0
    index = at
    while index < len(source):
        char = source[index]
        if char in _QUOTES:
            index = _skip_literal(source, index)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise AssertionError("unbalanced brackets")


def _assert_classes(class_string: str, *rules: str) -> None:
    """Every one of `rules` is its own class in `class_string`."""
    present = set(class_string.split())
    missing = [rule for rule in rules if rule not in present]
    assert not missing, f"{missing} missing from {class_string!r}"


# The scanners are run, not read: Node strips the types and nothing needs installing.
_TS_ALIAS = re.compile(r'"@/lib/([a-z-]+)"')
_TS_RUNNER = """
import { resolveReleaseBodyLinks } from "./release-body-links.ts";
import { releaseNotesPreview } from "./release-notes-preview.ts";

const chunks: Buffer[] = [];
process.stdin.on("data", (chunk: Buffer) => chunks.push(chunk));
process.stdin.on("end", () => {
  const markdown = Buffer.concat(chunks).toString("utf8");
  const result =
    process.argv[2] === "links"
      ? resolveReleaseBodyLinks(markdown)
      : releaseNotesPreview(markdown);
  process.stdout.write(JSON.stringify(result));
});
"""

SAMPLE = """Intro prose, the announcement itself.

## Kimi K3

- a real section

## Updating / installing Unsloth

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

## What's Changed

* Something by @someone in https://github.com/unslothai/unsloth/pull/1

**Full Changelog**: https://github.com/unslothai/unsloth/compare/v1...v2
"""


@dataclass(frozen = True)
class Section:
    """A heading and the lines under it, to the next heading of any level."""

    version: str
    heading: str
    body: str


def sections(module, text: str) -> list[Section]:
    """Every document-level heading in `text`, with the lines beneath it.

    The shipped scanner decides what a heading is; this only groups its events.
    """
    found: list[Section] = []
    bodies: list[list[str]] = []
    for event in module.scan_blocks(text):
        if isinstance(event, module.Heading):
            if bodies and event.retract:
                del bodies[-1][len(bodies[-1]) - event.retract :]
            title = event.title.strip()
            found.append(Section(version = title.split()[0] if title else "", heading = title, body = ""))
            bodies.append([])
            continue
        if bodies:
            bodies[-1].append(event.line)
    return [
        Section(version = entry.version, heading = entry.heading, body = "\n".join(body).strip())
        for entry, body in zip(found, bodies)
    ]


def parse_sections(module, text: str) -> list[Section]:
    """`sections`, keeping only those whose heading starts with a version."""
    return [entry for entry in sections(module, text) if module._parse_version(entry.version)]


def find_section(module, text: str, version: str) -> Section | None:
    for entry in parse_sections(module, text):
        if entry.version == version:
            return entry
    wanted = module._parse_version(version)
    for entry in parse_sections(module, text):
        if wanted is not None and module._parse_version(entry.version) == wanted:
            return entry
    return None


def releases_payload(*entries: dict) -> str:
    """A GitHub releases response, with the fields the selector reads."""
    defaults = {
        "draft": False,
        "prerelease": False,
        "name": "",
        "body": "",
        "html_url": "",
    }
    return json.dumps([{**defaults, **entry} for entry in entries])


@pytest.fixture(scope = "module")
def notes_module():
    sys.path.insert(0, str(BACKEND))
    try:
        from utils import release_notes
    finally:
        sys.path.pop(0)
    release_notes.reset_release_notes_cache()
    yield release_notes
    release_notes.reset_release_notes_cache()


@pytest.fixture
def serve_releases(notes_module, monkeypatch):
    """Serve a releases payload locally, and point the module at it."""
    monkeypatch.delenv(notes_module.DISABLE_ENV_VAR, raising = False)
    servers: list[http.server.HTTPServer] = []

    def serve(
        body: str,
        status: int = 200,
        headers: dict[str, str] | None = None,
        reset: bool = True,
    ):
        hits = {"count": 0}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib naming
                hits["count"] += 1
                payload = body.encode("utf-8")
                self.send_response(status)
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        servers.append(server)
        threading.Thread(target = server.serve_forever, daemon = True).start()
        monkeypatch.setenv(
            notes_module.RELEASES_URL_ENV_VAR,
            f"http://127.0.0.1:{server.server_port}/releases",
        )
        if reset:
            notes_module.reset_release_notes_cache()
        return hits

    yield serve
    for server in servers:
        server.shutdown()
        server.server_close()
    notes_module.reset_release_notes_cache()


@pytest.fixture
def isolated_releases(notes_module, monkeypatch):
    """Keep the module away from the network entirely."""
    monkeypatch.setenv(notes_module.DISABLE_ENV_VAR, "1")
    notes_module.reset_release_notes_cache()
    yield notes_module
    notes_module.reset_release_notes_cache()


def test_only_real_headings_become_sections(notes_module):
    headings = [entry.heading for entry in sections(notes_module, SAMPLE)]
    assert headings == ["Kimi K3", "Updating / installing Unsloth", "What's Changed"]


def test_section_body_stops_at_the_next_heading(notes_module):
    entry = sections(notes_module, SAMPLE)[0]
    assert "a real section" in entry.body
    assert "install.sh" not in entry.body


def test_the_announcement_survives_and_the_generated_sections_do_not(notes_module):
    stripped = notes_module.strip_release_body(SAMPLE)
    assert "Intro prose" in stripped
    assert "## Kimi K3" in stripped and "a real section" in stripped
    for gone in (
        "Updating / installing Unsloth",
        "install.sh",
        "What's Changed",
        "pull/1",
        "Full Changelog",
    ):
        assert gone not in stripped


def test_response_reports_no_notes_without_markdown(isolated_releases):
    """Update checks are off, so there is no release and nothing to show."""
    payload = isolated_releases.get_release_notes("2026.7.7")
    assert payload["matched"] is False
    assert payload["markdown"] is None
    assert payload["version"] == "2026.7.7"
    assert payload["release_notes_url"]


def test_the_offered_version_is_echoed_not_looked_up(notes_module, serve_releases):
    """No tag could match the PyPI version the pip popup offers, so it comes
    back untouched for the UI to match a stale answer against."""
    serve_releases(
        releases_payload(
            {
                "tag_name": "v0.1.60-beta",
                "name": "Meta Muse Glimmer",
                "body": "The announcement.\n",
                "html_url": "https://github.com/unslothai/unsloth/releases/tag/v0.1.60-beta",
                "published_at": "2026-08-10T11:59:46Z",
            }
        )
    )
    payload = notes_module.get_release_notes("2026.8.11")
    assert payload["version"] == "2026.8.11"
    assert payload["tag"] == "v0.1.60-beta"
    assert payload["heading"] == "Meta Muse Glimmer"
    assert payload["html_url"].endswith("/releases/tag/v0.1.60-beta")
    assert payload["matched"] is True and payload["source"] == "github"
    assert "The announcement." in payload["markdown"]


def test_unsupported_version_query_is_rejected(isolated_releases):
    assert isolated_releases.is_supported_version_query("2026.7.6") is True
    for bad in ("../etc/passwd", "2026.7.6 OR 1", "", "a" * 80):
        assert isolated_releases.is_supported_version_query(bad) is False
    assert isolated_releases.get_release_notes("../etc/passwd")["matched"] is False


def test_the_newest_published_release_wins(notes_module, serve_releases):
    """Ordered by publication, never by tag: v0.1.60-beta was published after
    v0.1.527-beta, so any numeric or SemVer sort picks the wrong one."""
    serve_releases(
        releases_payload(
            {
                "tag_name": "v0.1.527-beta",
                "body": "older",
                "published_at": "2026-08-09T17:14:42Z",
            },
            {
                "tag_name": "v0.1.60-beta",
                "body": "newer",
                "published_at": "2026-08-10T11:59:46Z",
            },
        )
    )
    payload = notes_module.get_release_notes("2026.8.11")
    assert payload["tag"] == "v0.1.60-beta"
    assert "newer" in payload["markdown"]


@pytest.mark.parametrize(
    "entry",
    [
        {"tag_name": "v9.9.9", "draft": True},
        {"tag_name": "desktop-v0.1.60-beta"},
        {"tag_name": "b8475"},
        {"tag_name": "February-2026"},
    ],
)
def test_only_a_published_studio_release_is_shown(notes_module, serve_releases, entry):
    serve_releases(
        releases_payload(
            {"body": "not an announcement", "published_at": "2026-08-11T00:00:00Z", **entry},
            {
                "tag_name": "v0.1.60-beta",
                "body": "the announcement",
                "published_at": "2026-08-10T11:59:46Z",
            },
        )
    )
    payload = notes_module.get_release_notes("2026.8.11")
    assert payload["tag"] == "v0.1.60-beta"
    assert "not an announcement" not in (payload["markdown"] or "")


@pytest.mark.parametrize("body", ['{"message": "Not Found"}', "[]", "not json at all"])
def test_a_payload_without_releases_is_an_error_not_a_crash(notes_module, serve_releases, body):
    serve_releases(body)
    payload = notes_module.get_release_notes("2026.8.11")
    assert payload["matched"] is False
    assert payload["error"], "the UI needs to know it can retry"


def test_a_release_with_no_announcement_shows_none(notes_module, serve_releases):
    """v0.1.527-beta's body is only the generated list, so nothing survives
    stripping: the popup links out rather than showing an older release."""
    serve_releases(
        releases_payload(
            {
                "tag_name": "v0.1.527-beta",
                "name": "Unsloth v0.1.527-beta",
                "body": (BODIES / "v0.1.527-beta.md").read_text(encoding = "utf-8"),
                "html_url": "https://github.com/unslothai/unsloth/releases/tag/v0.1.527-beta",
                "published_at": "2026-08-09T17:14:42Z",
            },
            {
                "tag_name": "v0.1.526-beta",
                "body": "An earlier announcement.\n",
                "published_at": "2026-08-04T16:06:43Z",
            },
        )
    )
    payload = notes_module.get_release_notes("2026.8.10")
    assert payload["matched"] is False and payload["markdown"] is None
    assert payload["tag"] == "v0.1.527-beta"
    assert payload["html_url"].endswith("/releases/tag/v0.1.527-beta")
    assert payload["error"] is None, "no notes is not a failure"


def test_longer_outer_fence_does_not_leak_a_fake_section(notes_module):
    """A ``` sample inside a ```` block must not close the block and let the
    sample's heading be indexed as a real release."""
    text = "## 1.0\n\n````md\n```\n## 9.9.9\n```\n````\n\n- real note\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0"]
    assert find_section(notes_module, text, "9.9.9") is None


def test_tilde_fence_is_not_closed_by_backticks(notes_module):
    text = "## 1.0\n\n~~~\n```\n## 9.9.9\n~~~\n\n- real\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0"]


def test_utf8_bom_does_not_hide_the_first_section(notes_module):
    """Editors on Windows can leave a BOM on the first line."""
    assert [e.version for e in parse_sections(notes_module, "\ufeff## 1.0\n\n- x\n")] == ["1.0"]


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_non_unix_line_endings(notes_module, newline):
    text = f"## 1.0{newline}{newline}- windows note{newline}"
    entry = find_section(notes_module, text, "1.0")
    assert entry is not None and "windows note" in entry.body
    assert "\r" not in entry.body


def test_closing_fence_must_carry_nothing_after_it(notes_module):
    """CommonMark: a closer is the delimiter plus whitespace only. A ```` line
    with trailing text inside a ```` block is content, not the end."""
    text = "## 1.0\n\n````md\n```` not a closer\n## 9.9.9\n````\n\n- real\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0"]
    info = "## 1.0\n\n```python\n## 9.9.9\n```\n\n- real\n"
    assert [e.version for e in parse_sections(notes_module, info)] == ["1.0"]


@pytest.mark.parametrize(
    "text",
    [
        "## 1.0\n\n- real\n\n<!--\n## 9.9.9\n\n- unpublished\n-->\n",
        "## 1.0\n\n- real\n\n<!-- ## 9.9.9 -->\n",
    ],
)
def test_commented_out_sections_are_not_releases(notes_module, text):
    """Markdown does not render them, so they are not published notes."""
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0"]
    assert find_section(notes_module, text, "9.9.9") is None


def test_no_changelog_file_is_packaged_or_read():
    """The releases are the only source now. A file left in the packaging would
    be a second one, editable in a checkout and stale in a wheel."""
    assert not (REPO / "CHANGELOG.md").exists()
    assert not (REPO / "_changelog_build.py").exists()
    for name in ("pyproject.toml", "build.sh", ".gitignore"):
        assert "CHANGELOG.md" not in (REPO / name).read_text(encoding = "utf-8")
    source = MODULE.read_text(encoding = "utf-8")
    assert "CHANGELOG.md" not in source and "changelog.py" not in source


def test_preview_keeps_identifier_underscores():
    """UNSLOTH_DISABLE_UPDATE_CHECK must not render as UNSLOTHDISABLEUPDATECHECK."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "BOLD_UNDERSCORE" in src and "ITALIC_UNDERSCORE" in src
    assert "parkCodeSpans" in src, "code spans are parked so their underscores survive"
    assert "const EMPHASIS" not in src, "the blanket emphasis strip is gone"


def test_panel_prefers_the_page_the_notes_came_from():
    """The page the notes came from wins over the caller's URL and the changelog."""
    src = PANEL.read_text(encoding = "utf-8")
    assert "notes?.htmlUrl ?? releaseNotesUrl ?? notes?.releaseNotesUrl" in src


def test_remote_failure_is_reported_so_the_ui_can_retry(notes_module, monkeypatch):
    """An unreachable GitHub is retryable; "no notes published" is not."""
    monkeypatch.delenv(notes_module.DISABLE_ENV_VAR, raising = False)
    # Port 9 (discard) refuses fast, standing in for an unreachable host.
    monkeypatch.setenv(notes_module.RELEASES_URL_ENV_VAR, "http://127.0.0.1:9/releases")
    notes_module.reset_release_notes_cache()
    try:
        payload = notes_module.get_release_notes("2.0")
        assert payload["matched"] is False
        assert payload["error"], "remote failure must reach the UI"
    finally:
        notes_module.reset_release_notes_cache()


def test_preview_keeps_comparison_operators():
    """The tag strip must keep the operators in "Support Python <3.15 and >3.9"."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "/<\\/?[a-zA-Z][^>]*>/g" in src, "tag strip must require a name character"


def test_preview_hides_commented_out_notes():
    """Unpublished notes inside <!-- --> are not rendered, so not previewed."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "stripCommentSpans" in src and "COMMENT_OPEN" in src


def test_hook_treats_a_reported_failure_as_retryable():
    src = NOTES_HOOK.read_text(encoding = "utf-8")
    assert "next.error !== null" in src


def test_comment_delimiter_in_inline_code_is_literal(notes_module):
    """A note documenting `<!--` used to swallow every release below it."""
    text = "## 2.0\n\n- Type `<!--` to begin a comment\n\n## 1.0\n\n- older\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["2.0", "1.0"]
    assert find_section(notes_module, text, "1.0") is not None
    assert "older" not in find_section(notes_module, text, "2.0").body


def test_refresh_retries_a_cached_remote_failure(notes_module, serve_releases):
    """Retry must reach the network again rather than replay the cached failure."""
    hits = serve_releases("", status = 500)
    notes_module.get_release_notes("2.0")
    notes_module.get_release_notes("2.0")
    assert hits["count"] == 1, "the failure should be cached"
    notes_module.get_release_notes("2.0", refresh = True)
    assert hits["count"] == 2, "refresh must bypass the cached failure"


def test_a_rate_limit_is_not_retried_until_it_resets(notes_module, serve_releases):
    """A shared IP that has spent its 60 requests an hour gains nothing by
    retrying: the request is refused and only pushes the reset further away."""
    reset = str(int(time.time()) + 900)
    hits = serve_releases(
        '{"message": "API rate limit exceeded"}',
        status = 403,
        headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset},
    )
    payload = notes_module.get_release_notes("2.0")
    assert payload["matched"] is False and "rate limit" in payload["error"].lower()
    notes_module.get_release_notes("2.0", refresh = True)
    assert hits["count"] == 1, "refresh must not bypass a rate-limit lockout"


def test_a_rate_limit_deadline_is_bounded_not_just_its_first_wait(notes_module):
    """GitHub says not to request again before X-RateLimit-Reset, so the reset
    wins over the back-off. Only the first wait used to be bounded, so the fetch
    after it answered from the raw header and a skewed value parked the popup."""
    import email.message
    import urllib.error

    def refused(seconds_out: float):
        headers = email.message.Message()
        headers["X-RateLimit-Remaining"] = "0"
        headers["X-RateLimit-Reset"] = str(int(time.time() + seconds_out))
        return urllib.error.HTTPError("url", 403, "rate limited", headers, None)

    ceiling = notes_module.RELEASES_RATE_LIMIT_MAX_SECONDS
    for seconds_out in (ceiling, 365 * 24 * 60 * 60):
        notes_module.reset_release_notes_cache()
        _, first = notes_module._http_error_source(refused(seconds_out))
        _, next_wait = notes_module._fetch_latest_release()
        assert first <= ceiling and next_wait <= ceiling
    notes_module.reset_release_notes_cache()
    _, short = notes_module._http_error_source(refused(120))
    assert 60 <= short <= 180


def test_every_refusal_records_a_deadline_retry_has_to_wait_out(notes_module):
    """A secondary limit answers 403 or 429 with no `X-RateLimit-Remaining: 0`,
    and at most a `Retry-After`. Only the primary path recorded a deadline, so
    Retry dropped the cached failure and requested straight back into it."""
    import email.message
    import urllib.error

    def refused(code: int, **headers: str):
        message = email.message.Message()
        for name, value in headers.items():
            message[name] = value
        return urllib.error.HTTPError("url", code, "refused", message, None)

    ceiling = notes_module.RELEASES_RATE_LIMIT_MAX_SECONDS
    cases = [
        (refused(429), notes_module.RELEASES_RATE_LIMITED_TTL_SECONDS),
        # Retry-After wins, being how a secondary limit states its wait.
        (refused(403, **{"Retry-After": "120"}), 120),
        (
            refused(403, **{"Retry-After": "45", "X-RateLimit-Remaining": "0"}),
            45,
        ),
    ]
    for error, expected in cases:
        notes_module.reset_release_notes_cache()
        _, ttl = notes_module._http_error_source(error)
        assert notes_module._rate_limited_until > time.time(), "no lockout recorded"
        assert abs(ttl - expected) <= 2 and ttl <= ceiling


def test_the_page_asked_for_fits_under_the_read_cap(notes_module, serve_releases):
    """A release entry carries its whole body, so the page size and the byte cap
    are one decision: the endpoint's maximum of 100 is near 4 MiB against a
    2 MiB cap, and the fetch then fails outright."""
    import urllib.parse

    query = urllib.parse.urlparse(notes_module.RELEASES_API_URL).query
    per_page = int(urllib.parse.parse_qs(query)["per_page"][0])
    largest = max(len(path.read_text(encoding = "utf-8")) for path in BODIES.glob("*.md"))
    full_page = [
        {
            "tag_name": f"v0.1.{index}-beta",
            "body": "x" * largest,
            "published_at": f"2026-08-{index % 28 + 1:02d}T00:00:00Z",
        }
        for index in range(per_page)
    ]
    payload = releases_payload(*full_page)
    assert len(payload) < notes_module.RELEASES_MAX_BYTES, (
        f"a full page of {per_page} is {len(payload) / 1024 / 1024:.1f} MiB "
        f"against a {notes_module.RELEASES_MAX_BYTES / 1024 / 1024:.0f} MiB cap"
    )
    serve_releases(payload)
    assert notes_module.get_release_notes("2026.8.11")["error"] is None


@pytest.mark.parametrize(
    "heading",
    ["### macOS, Linux, WSL:", "### macOS / Linux / WSL", "### macOS/Linux/WSL"],
)
def test_platform_headings_split_on_a_slash_as_well_as_a_comma(notes_module, heading):
    """The install block splits its commands across per-platform headings, whose
    separator is written either way."""
    body = (
        "The announcement.\n\n"
        "### To update Unsloth or install a new Unsloth Studio, you must use:\n\n"
        f"{heading}\n\n```\ncurl -fsSL https://unsloth.ai/install.sh | sh\n```\n\n"
        "### Kimi K3\n\n- a real change\n"
    )
    stripped = notes_module.strip_release_body(body)
    assert "install.sh" not in stripped and "macOS" not in stripped
    assert "### Kimi K3" in stripped and "a real change" in stripped


def test_a_cached_release_answers_an_unchanged_response(notes_module, serve_releases):
    """GitHub answers a conditional request with 304 and no body, which is the
    release already held rather than a failure."""
    serve_releases(
        releases_payload(
            {
                "tag_name": "v0.1.60-beta",
                "body": "The announcement.\n",
                "published_at": "2026-08-10T11:59:46Z",
            }
        ),
        headers = {"ETag": '"abc"'},
    )
    assert notes_module.get_release_notes("2.0")["matched"] is True
    serve_releases("", status = 304, reset = False)
    # Expired, not reset: the ETag and last good release must carry over.
    notes_module._remote_cache.expires_at = 0
    payload = notes_module.get_release_notes("2.0")
    assert payload["matched"] is True and payload["tag"] == "v0.1.60-beta"


def test_hook_never_returns_another_versions_notes():
    """State still describes the previous version until the effect runs."""
    src = NOTES_HOOK.read_text(encoding = "utf-8")
    assert "notes.version === version" in src
    assert "load(version, true)" in src, "retry must ask the backend to bypass its cache"


@pytest.mark.parametrize("indent", ["", " ", "  ", "   "])
def test_headings_and_fences_allow_commonmark_indentation(notes_module, indent):
    """Markdown renders up to three leading spaces, so the parser must agree."""
    text = f"## 1.0\n\nOne.\n\n{indent}## 2.0\n\nTwo.\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0", "2.0"]
    fenced = f"## 1.0\n\n{indent}```\n{indent}## 9.9.9\n{indent}```\n\n- real\n"
    assert [e.version for e in parse_sections(notes_module, fenced)] == ["1.0"]


def test_four_space_indentation_is_code_not_structure(notes_module):
    """At four spaces Markdown switches to indented code, for both forms."""
    assert [
        e.version for e in parse_sections(notes_module, "    ## 9.9.9\n\n## 1.0\n\n- real\n")
    ] == ["1.0"]
    assert [
        e.version
        for e in parse_sections(notes_module, "## 1.0\n\n    ```\n    sample\n\n## 2.0\n\n- two\n")
    ] == ["1.0", "2.0"]


def test_desktop_notes_link_to_the_release_page_on_every_platform():
    """manualReleaseUrl is Linux-package only; the rest need the release page."""
    hook = (FRONTEND / "hooks/use-tauri-update.ts").read_text(encoding = "utf-8")
    assert "const releasePageUrl = info ?" in hook
    banner = TAURI_BANNER.read_text(encoding = "utf-8")
    assert "releaseNotesUrl={releasePageUrl ?? manualReleaseUrl}" in banner
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    assert "releasePageUrl={update.releasePageUrl}" in provider


def test_preview_matches_how_markdown_renders_prose_and_links():
    """Three rendering mismatches the preview must not reintroduce: wrapped
    paragraphs split into fragments, autolinks eaten as tags, a lead cut short."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "collector.paragraph = collector.paragraph" in src
    assert "AUTOLINK" in src
    assert "ABBREVIATIONS" in src and "INITIAL" in src


def test_preview_treats_code_as_literal():
    """Inside a code span, and inside an indented code block, Markdown renders
    the text literally, so the preview must not transform or promote it."""
    src = PREVIEW.read_text(encoding = "utf-8")
    park = src.index("parkCodeSpans(markdown")
    assert park < src.index("stripHtmlTags(\n    parked")
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


def test_panel_is_scrollable_and_shows_only_the_stripped_notes():
    src = PANEL.read_text(encoding = "utf-8")
    assert "overflow-y-auto" in src, "release notes must scroll inside the popup"
    assert "max-h-" in src, "the scroller needs a bounded height"
    # latest.json's `notes` is install boilerplate, the same every release.
    assert "fallbackMarkdown" not in src
    assert "notes?.matched ? notes.markdown : null" in src


def test_notes_surface_is_borderless_and_lifts_in_dark_mode():
    src = PANEL.read_text(encoding = "utf-8")
    layout = NOTES_LAYOUT.read_text(encoding = "utf-8")
    assert "border border-border" not in src, "the notes box is a fill, not a bordered box"
    # Lighter than the card behind it: 0.06 white scaled by --contrast-wash-gain. The class must
    # match whole, since a suffix, variant or `hover:` prefix makes Tailwind read another candidate.
    assert re.search(
        r"(?:(?<=[\s\"'`])|^)"
        r"dark:bg-(?:white/\[0\.06\]"
        r"|\[rgb\(255_255_255_/_calc\(0\.06\*var\(--contrast-wash-gain,\s*1\)\)\)\])"
        r"(?=[\s\"'`]|$)",
        layout,
    ), "the dark notes surface is no longer a 0.06 white lift"
    # Streamdown's mt-6 clips the first heading against the scroller edge.
    assert "[&>*>*:first-child]:mt-0" in src
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
    assert "collectBullets" in preview and "flush" in preview
    assert "NESTED_INDENT_TOLERANCE" in preview
    # Tag stripping repeats: one pass turns `<<b>b>` back into a live tag.
    assert "while (out !== previous)" in preview

    panel = PANEL.read_text(encoding = "utf-8")
    assert "releaseNotesPreview" in panel
    assert 'data-testid="update-release-notes-summary"' in panel
    assert "enabled: true" in panel


def test_preview_highlights_the_leading_sentence():
    """Each bullet leads with its headline sentence, emphasised over the rest."""
    preview = PREVIEW.read_text(encoding = "utf-8")
    assert "splitLeadSentence" in preview
    assert "SENTENCE_BREAK" in preview and "(?=" in preview

    panel = PANEL.read_text(encoding = "utf-8")
    layout = NOTES_LAYOUT.read_text(encoding = "utf-8")
    assert "UPDATE_NOTES_LEAD_CLASS" in panel
    assert '"font-medium text-foreground"' in layout
    assert "item.rest" in panel


def _max_widths(source: str) -> set[str]:
    """Every `max-w-[...]` in *source*, read at the default UI scale.

    #11648 wrapped these lengths in `calc(Npx*var(--ui-space-scale,1))` so they follow the interface size, which is
    Npx at the default scale. Read back that way, the contract stays about the width rather than its spelling.
    """
    return {_SCALED_PX.sub(r"\1", width) for width in re.findall(r"max-w-\[([^\]\s\"]+)\]", source)}


_SCALED_PX = re.compile(r"calc\((\d+(?:\.\d+)?px)\*var\(--ui-space-scale,1\)\)")
# `_max_widths` reads a bare 448px identically, so the scaled form is pinned raw.
_NOTES_WIDTH_SCALED = "max-w-[calc(448px*var(--ui-space-scale,1))]"


@pytest.mark.parametrize("banner", [WEB_BANNER, TAURI_BANNER])
def test_update_popups_share_the_notes_width(banner):
    """Every update popup uses the same width for its notes and action rows."""
    source = banner.read_text(encoding = "utf-8")
    assert "448px" in _max_widths(source)
    assert (
        _NOTES_WIDTH_SCALED in source and "max-w-[448px]" not in source
    ), "the notes width stopped scaling"
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    assert "400px" not in _max_widths(provider), "stack must not cap overlay width"
    llama = (FRONTEND / "components/llama-update-banner.tsx").read_text(encoding = "utf-8")
    assert "448px" in _max_widths(llama)
    assert (
        _NOTES_WIDTH_SCALED in llama and "max-w-[448px]" not in llama
    ), "the notes width stopped scaling"
    assert "400px" not in _max_widths(llama)


@pytest.mark.parametrize("banner", [WEB_BANNER, TAURI_BANNER])
def test_banners_toggle_inline_release_notes(banner):
    src = banner.read_text(encoding = "utf-8")
    assert "ReleaseNotesPanel" in src
    assert "Show release notes" in src and "Hide release notes" in src
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
    toggle_line = next(line for line in src.splitlines() if toggle in line)
    toggle_block = src[src.index("Button", row) : src.index(toggle_line)]
    assert "text-ui-13" in toggle_block and "whitespace-nowrap" in toggle_block


def test_headings_inside_a_raw_html_block_are_not_releases(notes_module):
    """<pre> content is literal, so a sample heading in it is not a section."""
    text = "## 1.0\n\n<pre>\n## 9.9.9\n</pre>\n\n- real note\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0"]
    assert "real note" in find_section(notes_module, text, "1.0").body
    assert find_section(notes_module, text, "9.9.9") is None


def test_details_blocks_still_contain_markdown(notes_module):
    """<details> is a type 6 block, so headings inside it still count."""
    text = "## 2.0\n\n<details>\n<summary>More</summary>\n\n- note\n\n</details>\n\n## 1.0\n\n- older\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["2.0", "1.0"]


def test_inline_raw_html_tag_does_not_open_a_block(notes_module):
    """A block opens only at the start of a line; a tag mid-sentence is inline."""
    text = "## 2.0\n\n- Warn when a <script> tag is pasted\n\n## 1.0\n\n- older\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["2.0", "1.0"]


def test_preview_skips_raw_html_blocks():
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "stripRawHtml" in src
    assert "/^ {0,3}<(pre|script|style|textarea)" in src


def test_fence_inside_a_raw_html_block_is_literal(notes_module):
    """A stray ``` in a <pre> sample is literal; as a fence it hid later releases."""
    text = "## 2.0\n\n<pre>\n```\n</pre>\n\n## 1.0\n\n- older\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["2.0", "1.0"]


def test_raw_html_block_closes_on_any_of_the_four_tags(notes_module):
    """A type 1 block ends at the first of the four closers, matched or not."""
    text = '## 1.0\n\n<script>\nconst sample = "</pre>";\n## 9.9.9\n</script>\n'
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0", "9.9.9"]


@pytest.mark.parametrize("tag", ["details", "div", "table"])
def test_type_6_blocks_run_until_a_blank_line(notes_module, tag):
    """`<details>` holds Markdown only after a blank line closes the block."""
    packed = f"## 1.0\n\n<{tag}>\n## 9.9.9\n</{tag}>\n\n- note\n"
    assert [e.version for e in parse_sections(notes_module, packed)] == ["1.0"]
    spaced = f"## 1.0\n\n<{tag}>\n\n## 2.0\n\n- note\n"
    assert [e.version for e in parse_sections(notes_module, spaced)] == ["1.0", "2.0"]


def test_a_tag_only_line_cannot_interrupt_a_paragraph(notes_module):
    """Type 7 blocks do not interrupt a paragraph."""
    text = "## 2.0\n\nSome prose.\n<span>\n\n## 1.0\n\n- older\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["2.0", "1.0"]


def test_preview_joins_an_indented_continuation_line():
    """Four spaces only start code outside a paragraph; inside one it is a wrap."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "!insideBlock && line.indent - line.column >= INDENTED_CODE_INDENT" in src
    assert "opensDeepFence" in src


def test_preview_code_spans_need_a_matching_closer():
    """A closer is a run of the same length, so the inner backticks survive."""
    src = CODE_SPANS.read_text(encoding = "utf-8")
    assert "candidate === ticks" in src, "a closer is a run of the same length"
    assert "stripPadding" in src, "one space of padding is dropped, as in Markdown"


def test_preview_skips_thematic_breaks():
    """`- - -` renders as a rule, so it must not take a preview slot."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "THEMATIC_BREAK" in src
    assert "THEMATIC_BREAK.test(visible)" in src


def test_preview_keeps_quoted_examples_out_of_the_headlines():
    """A quoted list is example output, not a change."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "quoted: boolean" in src
    assert "if (!line.quoted)" in src, "quoted bullets never become headlines"


def test_notes_panel_keeps_the_link_when_the_lookup_fails():
    """The release page can be reachable when the backend lookup is not."""
    src = PANEL.read_text(encoding = "utf-8")
    error_branch = src[src.index('if (state === "error")') :]
    retry = error_branch.index("update-release-notes-retry")
    assert error_branch.index("{link}") > retry, "link sits beside retry"


def test_hook_waits_for_the_desktop_auth_token():
    """A token not installed yet must not be recorded as a failed lookup."""
    src = NOTES_HOOK.read_text(encoding = "utf-8")
    assert "hasAuthToken()" in src and "AUTH_POLL_LIMIT" in src


def test_a_body_staged_as_a_comment_reads_as_unpublished(notes_module, serve_releases):
    """A release drafted inside <!-- --> renders as nothing, so it is unpublished."""
    serve_releases(
        releases_payload(
            {
                "tag_name": "v2.0",
                "body": "<!-- not ready -->\n",
                "published_at": "2026-08-10T00:00:00Z",
            }
        )
    )
    staged = notes_module.get_release_notes("2.0")
    assert staged["matched"] is False and staged["markdown"] is None
    assert staged["source"] is None, "nothing was shown, so nothing sourced it"


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
def test_visibility_check_only_hides_comments(notes_module, body, visible):
    assert notes_module._renders_visibly(body) is visible


@pytest.mark.parametrize(
    "block",
    [
        "<?php\n## 9.9.9\n?>",
        "<![CDATA[\n## 9.9.9\n]]>",
        "<!DOCTYPE\n## 9.9.9\n>",
    ],
)
def test_processing_instructions_and_declarations_are_literal(notes_module, block):
    """Raw block types 3 to 5 render literally, so a heading in one is a sample."""
    text = f"## 1.0\n\n{block}\n\n- real note\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0"]
    assert "real note" in find_section(notes_module, text, "1.0").body


def test_headings_need_a_space_or_tab_after_the_hashes(notes_module):
    """A non-breaking space after the hashes renders as text, not a heading."""
    text = "## 1.0\n\n- real note\n\n## 9.9.9\n\n- not a release\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0"]
    assert find_section(notes_module, text, "9.9.9") is None
    tabbed = "## 1.0\n\n- one\n\n##\t2.0\n\n- two\n"
    assert [e.version for e in parse_sections(notes_module, tabbed)] == ["1.0", "2.0"]


def test_preview_skips_every_raw_block_form():
    """The extractor tracks the same block forms as the parser."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "RAW_BLOCKS" in src
    assert "CDATA" in src and "[A-Za-z]" in src


@pytest.mark.parametrize("banner", [WEB_BANNER, TAURI_BANNER])
def test_expanded_popup_fits_a_short_viewport(banner):
    """A window under roughly 430px used to push the card's title off screen."""
    panel = PANEL.read_text(encoding = "utf-8")
    assert "min-h-0 flex-1" in panel, "notes height must follow the viewport"
    src = banner.read_text(encoding = "utf-8")
    assert CARD_CAP[banner] in src, "card is the backstop on tiny viewports"


def test_relative_release_body_links_point_at_the_repository():
    """Repository-relative links would resolve against Unsloth's own origin."""
    src = LINKS.read_text(encoding = "utf-8")
    assert "https://github.com/unslothai/unsloth/blob/main/" in src
    assert "https://raw.githubusercontent.com/unslothai/unsloth/main/" in src
    assert "ABSOLUTE" in src and "codeSpans" in src and "FENCE" in src
    panel = PANEL.read_text(encoding = "utf-8")
    assert "resolveReleaseBodyLinks" in panel


@pytest.mark.parametrize("query", ["latest", "main", "not-a-version", "abc"])
def test_unparseable_versions_are_rejected(notes_module, query):
    """A query that cannot parse is a bad request, not an empty result."""
    assert notes_module.is_supported_version_query(query) is False


@pytest.mark.parametrize("query", ["2026.7.5", "v2026.7.5", "2026.07.5", "1.0.0rc1"])
def test_real_versions_are_still_accepted(notes_module, query):
    assert notes_module.is_supported_version_query(query) is True


def test_reference_style_images_resolve_to_the_raw_host():
    """An image needs the raw file: the blob URL is an HTML page."""
    src = LINKS.read_text(encoding = "utf-8")
    assert "IMAGE_REFERENCE" in src
    assert "imageLabels" in src


def test_collapsed_notes_surface_is_hidden_when_nothing_previews():
    """Notes that preview as nothing leave an empty strip, worse than none."""
    src = PANEL.read_text(encoding = "utf-8")
    assert "preview?.items.length === 0" in src


def test_a_fence_closer_accepts_only_spaces_and_tabs(notes_module):
    """A delimiter followed by a non-breaking space is content, not a closer."""
    text = "## 1.0\n\n```\n```\u00a0\n## 9.9.9\n```\n\n- real note\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0"]
    plain = "## 1.0\n\n```\nx\n```\t\n\n## 2.0\n\n- two\n"
    assert [e.version for e in parse_sections(notes_module, plain)] == ["1.0", "2.0"]
    for source in (PREVIEW, LINKS):
        assert "/[^ \\t]/" in source.read_text(encoding = "utf-8")


def test_code_spans_close_on_a_run_of_equal_length():
    """`a``b [x](y.md)` is one code span, so the link inside it is literal."""
    src = CODE_SPANS.read_text(encoding = "utf-8")
    assert "candidate === ticks" in src, "closer length must match the opener"
    assert "markdown-code-spans" in PREVIEW.read_text(encoding = "utf-8")
    assert "markdown-code-spans" in LINKS.read_text(encoding = "utf-8")


def test_preview_decodes_entities_like_the_renderer():
    """Streamdown renders `AT&amp;T` as AT&T, so the raw entity must not show."""
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
    """CRLF used to hide fences and promote a code sample to a headline."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "LINE_ENDINGS" in src
    assert "LINE_ENDINGS" in LINKS.read_text(encoding = "utf-8")


def test_preview_renders_reference_links_as_text():
    """`[text][label]` renders as a link, so its raw markup must not show."""
    src = PREVIEW.read_text(encoding = "utf-8")
    assert "LINK_REFERENCE" in src and "IMAGE_REFERENCE" in src
    assert "DEFINITION" in src


def test_preview_treats_escaped_punctuation_as_literal():
    """`\\*not italic\\*` keeps its stars, and an escaped backtick opens no span."""
    assert "ESCAPE" in PREVIEW.read_text(encoding = "utf-8")
    assert "escaped(" in CODE_SPANS.read_text(encoding = "utf-8")


def test_link_resolver_skips_every_code_form():
    """Indented code and cross-line code spans render as code, so leave them."""
    src = LINKS.read_text(encoding = "utf-8")
    assert "INDENTED_CODE" in src
    assert "codeSpans(masked)" in src
    assert "definition.has(index)" in src


def test_badge_links_resolve_both_targets():
    """`[![alt](img)](link)`: the outer link needs a nested label to resolve."""
    assert "NESTED_LABEL" in LINKS.read_text(encoding = "utf-8")


def test_in_flight_requests_are_identified_not_just_versioned():
    """Two requests for one version could resolve out of order."""
    assert "requestIdRef" in NOTES_HOOK.read_text(encoding = "utf-8")


def test_notes_repair_the_shared_previews_width_reset():
    """MarkdownPreview clears max-width on descendants, so wide content escapes."""
    src = PANEL.read_text(encoding = "utf-8")
    assert "[&_img]:max-w-full" in src
    assert "[&_[data-streamdown=link-safety-modal]>*]:max-w-md" in src


@pytest.mark.parametrize("banner", [WEB_BANNER, TAURI_BANNER])
def test_only_the_notes_region_scrolls(banner):
    """The dismiss control sits inside the card, so the card must not scroll."""
    src = banner.read_text(encoding = "utf-8")
    surface = _card_surface(src)
    _assert_classes(surface, "flex", CARD_CAP[banner], "flex-col")
    # Asserted as the absence of scrolling overflow, not presence of overflow-hidden: the
    # browser card clips nothing at all.
    assert not re.search(
        r"(?<![\w-])overflow-(?:y-)?(?:auto|scroll)(?![\w-])", surface
    ), "the card scrolls, so its dismiss control can leave the viewport"
    if banner == WEB_BANNER:
        # Either class zeroes a flex item's automatic minimum, letting the rail cut the action row.
        for zeroes_the_floor in ("min-h-0", "overflow-hidden"):
            assert (
                zeroes_the_floor not in surface.split()
            ), f"{zeroes_the_floor} puts the browser card's floor back to nothing"
    else:
        _assert_classes(surface, "min-h-0", "overflow-hidden")
    layout = NOTES_LAYOUT.read_text(encoding = "utf-8")
    _assert_classes(
        _class_const(layout, "UPDATE_NOTES_ROOT_CLASS"),
        "flex",
        "min-h-0",
        "flex-1",
        "flex-col",
        "overflow-hidden",
    )
    panel = PANEL.read_text(encoding = "utf-8")
    assert "UPDATE_NOTES_EXPANDED_SCROLL_CLASS" in panel
    _assert_classes(
        _class_const(layout, "UPDATE_NOTES_EXPANDED_SCROLL_CLASS"),
        "max-h-64",
        "min-h-0",
        "flex-1",
        "overflow-y-auto",
    )
    # Without its own scroll, the collapsed bullets painted over the buttons in a small slot.
    _assert_classes(
        _class_on_testid(panel, "update-release-notes-summary"),
        "min-h-0",
        "flex-1",
        "space-y-1",
        "overflow-y-auto",
    )


def test_a_comment_marker_in_prose_cannot_swallow_later_releases(notes_module):
    """A note that mentions `<!--` used to put the parser into comment state for
    the rest of the file, hiding every release below it."""
    text = (
        "## 2026.8.0\n\n- Unsloth strips <!-- markers from pasted prompts.\n\n"
        "## 2026.7.5\n\n- SECRET: an older release\n"
    )
    assert [e.version for e in parse_sections(notes_module, text)] == ["2026.8.0", "2026.7.5"]
    assert "SECRET" not in find_section(notes_module, text, "2026.8.0").body
    assert find_section(notes_module, text, "2026.7.5") is not None
    hidden = "## 2.0\n\n<!--\n## 9.9.9\n-->\n\n- note\n"
    assert [e.version for e in parse_sections(notes_module, hidden)] == ["2.0"]


def test_unmatched_backtick_runs_stay_linear(notes_module):
    """Rescanning the suffix per opener was quadratic: 800 never-closing runs
    took 7.7s at 321 KB, and notes are reparsed on every popup request."""
    line = "".join("`" * (i + 1) + "x" for i in range(800))
    assert len(line) > 300_000
    started = time.monotonic()
    assert notes_module._code_span_ranges(line) == []
    assert time.monotonic() - started < 2.0


def test_a_base_exception_releases_the_single_flight_flag(notes_module, monkeypatch):
    """The flag was cleared only after `except Exception`, so a BaseException
    stranded it and every later caller waited out the full deadline."""
    notes_module.reset_release_notes_cache()

    def explode():
        raise KeyboardInterrupt

    monkeypatch.setattr(notes_module, "_fetch_latest_release", explode)
    with pytest.raises(KeyboardInterrupt):
        notes_module.get_latest_release()
    assert notes_module._remote_fetching is False
    notes_module.reset_release_notes_cache()


@pytest.mark.parametrize("marker", ["<!-->", "<!--->"])
def test_an_empty_comment_does_not_swallow_later_releases(notes_module, marker):
    """`<!-->` and `<!--->` are complete comments: the closer overlaps the
    opener, so searching past it hid every release below."""
    text = f"## 2.0\n\n- new stuff\n\n{marker}\n\n## 1.0\n\n- old stuff\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["2.0", "1.0"]
    assert find_section(notes_module, text, "1.0") is not None
    assert "old stuff" not in find_section(notes_module, text, "2.0").body
    assert "!line.includes(COMMENT_CLOSE)" in PREVIEW.read_text(encoding = "utf-8")


def test_an_unterminated_comment_still_hides_the_rest(notes_module):
    """The fix must not turn every `<!--` line into a no-op block."""
    text = "## 2.0\n\n<!-- never closed\n\n## 1.0\n\n- old stuff\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["2.0"]


def test_a_closing_delimiter_takes_its_whole_line(notes_module):
    """The closing line stays in the block, so a heading glued after it is not."""
    for text in (
        "## 1.0\n\n<!-- hidden -->## 9.9.9\n\n- note\n",
        "## 1.0\n\n<pre>\nx\n</pre>## 9.9.9\n\n- note\n",
    ):
        assert [e.version for e in parse_sections(notes_module, text)] == ["1.0"]


def test_an_exact_heading_is_never_shadowed(notes_module):
    """PEP 440 says 1.0 == 1.0.0, so the normalised match used to beat exact."""
    text = "## 1.0.0\n\n- padded\n\n## 1.0\n\n- exact\n"
    assert find_section(notes_module, text, "1.0").body == "- exact"
    assert find_section(notes_module, text, "1.0.0").body == "- padded"
    assert find_section(notes_module, "## 2026.7.6\n\n- x\n", "2026.07.6") is not None


def test_setext_headings_are_release_boundaries(notes_module):
    """A version over a line of dashes is the same heading in setext form."""
    text = "2.0\n---\n\n- new\n\n1.0\n---\n\n- old\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["2.0", "1.0"]
    assert find_section(notes_module, text, "2.0").body == "- new"
    assert [
        e.version for e in parse_sections(notes_module, "## 2.0\n\n- a\n\n---\n\n## 1.0\n\n- b\n")
    ] == ["2.0", "1.0"]


def test_a_long_backtick_run_does_not_stall_the_parser(notes_module):
    """The code-span guard used to backtrack: 20k backticks took over a minute.

    Asked as growth. `< 1.0` is a budget, and this runs in the `-n 4` CPU leg where a
    second of wall clock says as much about the other three workers as about the parser.
    Backtracking is superlinear, so 4x the backticks costing ~4x the time is the property.
    """
    assert_linear(
        lambda text: parse_sections(notes_module, text),
        lambda n: "## 1.0\n\n- " + "`" * n + " <!--\n",
        "backtick run",
        # 5_000, so the big leg is the 20k previously measured. See assert_linear.
        5_000,
    )


def test_the_remote_fetch_has_a_total_deadline(notes_module):
    """The socket timeout resets per read, so a trickle could hold a worker."""
    source = MODULE.read_text(encoding = "utf-8")
    assert "deadline = time.monotonic() + RELEASES_TIMEOUT_SECONDS" in source
    # read1 returns after one socket read, so the deadline is actually checked.
    assert "response.read1(" in source
    assert "Release notes are still loading." in source


def test_truncated_notes_close_their_fence(notes_module):
    """A blind slice could end inside a code block and break the rendering."""
    body = "```\n" + "x\n" * 20_000 + "```\n"
    payload = notes_module._notes_response(version = "1.0", markdown = body, source = "local")
    assert payload["truncated"] is True
    assert payload["markdown"].rstrip().endswith("```")


def test_the_opt_out_beats_the_developer_override():
    """The documented kill switch beats the dev switch, and the value must parse."""
    source = (BACKEND / "utils/update_status.py").read_text(encoding = "utf-8")
    assert "forced_version and not disabled and _is_version(forced_version)" in source


def test_a_list_item_over_dashes_is_not_a_setext_heading(notes_module):
    """`- first` over `---` is a list and a rule, not a setext heading."""
    text = "## 1.0\n\n- first\n---\n\n- second\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0"]
    body = find_section(notes_module, text, "1.0").body
    assert "first" in body and "second" in body
    setext = "2.0\n---\n\n- new\n\n1.0\n---\n\n- old\n"
    assert [e.version for e in parse_sections(notes_module, setext)] == ["2.0", "1.0"]


def test_a_backtick_in_a_fence_info_string_is_not_a_fence(notes_module):
    """A backtick fence's info string may hold no backtick, so that line is prose."""
    text = "## 2.0\n\n```bad`info\n\n## 1.0\n\n- old\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["2.0", "1.0"]
    assert [
        e.version
        for e in parse_sections(notes_module, "## 2.0\n\n```md\n## 9.9.9\n```\n\n## 1.0\n\n- old\n")
    ] == ["2.0", "1.0"]
    for source in (PREVIEW, LINKS):
        assert "info string" in source.read_text(encoding = "utf-8")


def test_preview_follows_commonmark_paragraph_rules():
    """Only an ordered list starting at 1 may interrupt a paragraph, an
    unresolved reference keeps its brackets, and a quote owns its own."""
    src = " ".join(PREVIEW.read_text(encoding = "utf-8").split())
    assert "const interrupts = collector.current === null" in src
    assert "!collector.quotedParagraph;" in src
    assert "definedLabel" in src, "a reference only renders as text when defined"
    assert "COMMENT_BLOCK_OPEN" in src


def test_link_resolver_leaves_raw_blocks_and_escapes_alone():
    src = LINKS.read_text(encoding = "utf-8")
    assert "RAW_HTML_OPEN" in src and "inRawHtml" in src
    assert "isEscaped(line, opener)" in src
    assert "BLOCK_LINE.test(structure)" in src


def test_code_span_closers_ignore_backslashes():
    """Escapes are not processed inside a code span, so a run after one closes."""
    src = CODE_SPANS.read_text(encoding = "utf-8")
    # Counted over the whole module: the scanner has moved above `codeSpans` once already.
    calls = [
        " ".join(line.split())
        for line in src.splitlines()
        if "escaped(" in line and not line.lstrip().startswith("function escaped(")
    ]
    assert len(calls) == 1, f"only an opener can be escaped, called at {calls}"
    assert '!== "`" || escaped(' in calls[0]


# The desktop card's incompressible height: a fixed part plus a part scaled by the font
# setting, since at 20px the action row wraps at every card width.
_SCALED_FLOOR_TAURI = "min-h-[calc(117px+93px*var(--ui-font-scale,1))]"
_NARROW_FLOOR_TAURI = "min-h-[calc(24px+224px*var(--ui-font-scale,1))]"
_NARROW = "max-[383px]"
# Floored while the notes panel is present, unsqueezable while it is not (#10229).
_NOTES_GATE = "has-[[data-slot=update-release-notes]]"


_BANNER_ROOT = re.compile(r'data-testid="(?:web|tauri)-update-banner"')


def _unpositioned_branch(text: str) -> str:
    """The `: ...` arm of `positioned ? ... : ...`.

    The two arms are two different elements: `positioned` is the standalone
    banner, and the rail-facing card is the alternative. Reading both at once
    would let a rule move from the card to the standalone banner and still
    satisfy a check about the card. Split at the colon at bracket depth zero
    and outside any literal, so a Tailwind variant in the first arm
    (`dark:bg-card`) is not mistaken for the separator.

    Ternary nesting is counted, not just brackets. A ternary inside the first
    arm has a colon of its own at the same bracket depth, and taking that one
    returns the tail of the `positioned` arm as though it were the card, so
    every floor could be asserted against the wrong element.
    """
    index = text.index("?", text.index("positioned")) + 1
    pending = 1
    depth = 0
    while index < len(text):
        char = text[index]
        if char in _QUOTES:
            index = _skip_literal(text, index)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "?" and depth == 0 and text[index + 1 : index + 2] not in (".", "?"):
            pending += 1
        elif char == ":" and depth == 0 and text[index - 1] != "?":
            pending -= 1
            if pending == 0:
                return text[index + 1 :]
        index += 1
    raise AssertionError("the positioned card has no unpositioned branch")


def _card_slot(source: str) -> str:
    """The rail-facing root of an update card, comments stripped.

    Not one string literal: the root is a `cn()` of several, so an assertion
    anchored on the first of them cannot see the floor at all. Anchored on the
    `data-testid` that names the card, so no class has to keep its place for
    the root to be found, and narrowed to the branch the overlay rail actually
    renders. Comments go because both files name the very classes under test in
    prose beside them, and a rule that a comment can satisfy is not tested.
    """
    clean = _without_comments(source)
    match = _BANNER_ROOT.search(clean)
    assert match, "the update card has lost its data-testid"
    start, end = _opening_tag(clean, match.start())
    key = "className={cn("
    tag = clean[start:end]
    assert key in tag, "the card's root no longer builds its classes with cn()"
    # Always-rendered classes only: a floor behind a constant is in the file but reaches no DOM.
    return _always_rendered(_unpositioned_branch(clean[start + tag.index(key) : end]))


def _card_surface(source: str) -> str:
    """The painted surface: the class string of the card's first child.

    Bounded to that child rather than taken as the next literal `className` in
    the file, which is the dismiss button's the moment the surface writes its
    own classes through `cn()` instead.
    """
    clean = _without_comments(source)
    match = _BANNER_ROOT.search(clean)
    assert match, "the update card has lost its data-testid"
    _, end = _opening_tag(clean, match.start())
    child = clean.index("<", end)
    # A fragment emits no element, so it is not the surface.
    while clean[child + 1] == ">":
        child = clean.index("<", child + 2)
    start, child_end = _opening_tag(clean, child + 1)
    return _class_value(clean[start:child_end], "the painted surface")


def _assert_floored(source: str, scaled: str, narrow: str, card: str) -> None:
    """The card keeps room for its header and buttons, in both of its states."""
    # Read off the rail-facing root, so a floor on an inner box does not count.
    root = _card_slot(source)
    # `_only_under`: a gated floor stops applying over part of its range, an ungated twin
    # reserves the empty height the gate was added to stop.
    assert _only_under(
        root, scaled, _NOTES_GATE
    ), f"the {card} card's floor is fixed, ungated, or gated more than its notes"
    assert _only_under(
        root, narrow, _NOTES_GATE, _NARROW
    ), f"the {card} card's floor misses the narrow card's extra button row"
    assert _only_under(root, "shrink-0"), f"the rail can squeeze the {card} card with no notes open"
    assert _only_under(
        root, "shrink", _NOTES_GATE
    ), f"the {card} card cannot give up its notes' height, so the rail clips its buttons"
    assert not _applies(
        root, "min-h-0"
    ), f"min-h-0 lets the rail squeeze the {card} card past its floor"


def _assert_floors_itself(source: str, card: str) -> None:
    """The card keeps room for its header and buttons without naming a height.

    The written-out floor it replaces was a constant against one type size, and
    #11458 made the spacing inside the card follow the interface font size too,
    so at the 20px setting the constant came to 209px while the content needed
    about 301px and the action row was cut. A measured floor cannot go stale
    that way, but it only exists while the surface declares neither `min-h-0`
    nor `overflow-hidden`: each of those sets a flex item's automatic minimum
    size to zero, which is what made a hand-written floor necessary at all.
    """
    root = _card_slot(source)
    surface = _card_surface(source)
    for zeroes_the_floor in ("min-h-0", "overflow-hidden"):
        assert (
            zeroes_the_floor not in surface.split()
        ), f"{zeroes_the_floor} leaves the {card} card with no floor at all"
    assert not re.search(
        r"(?<![\w-])min-h-\[", surface + root
    ), f"the {card} card names a height again, which goes stale at the next type size"
    assert _only_under(root, "shrink-0"), f"the rail can squeeze the {card} card with no notes open"
    assert _only_under(
        root, "shrink", _NOTES_GATE
    ), f"the {card} card cannot give up its notes' height, so the rail clips its buttons"
    assert not _applies(
        root, "min-h-0"
    ), f"min-h-0 lets the rail squeeze the {card} card past its floor"


RAIL_TESTID = "overlay-rail"


def _corner_rails(provider: str) -> list[str]:
    """The class strings of the bottom-right overlay rails.

    Anchored on ``data-testid`` rather than on a run of the rail's own classes. The old
    matcher spelled the corner INTO the pattern - `bottom-0 right-4` - so #11260 moving the
    rail flush to the edge (`right-0`, with the inset paid as inline px padding) made it
    match nothing, and four tests across two files failed at once while reporting a missing
    rail rather than a changed one. The corner is a CLAIM these tests make, so it belongs in
    an assertion, not in the thing that finds the element to assert about. Same reasoning as
    _class_on_testid, which this file already grew for exactly this failure.
    """
    return [_class_value(tag, RAIL_TESTID) for tag in _rail_openings(provider)]


def _rail_openings(provider: str) -> list[str]:
    """The opening tag of each rail, comments already blanked."""
    clean = _without_comments(provider)
    tags = []
    at = clean.find(f'data-testid="{RAIL_TESTID}"')
    while at != -1:
        start, end = _opening_tag(clean, at)
        tags.append(clean[start:end])
        at = clean.find(f'data-testid="{RAIL_TESTID}"', end)
    return tags


# Both Tailwind padding spellings, including the `[padding:` arbitrary form; the required `-`
# keeps pointer-events-none, peer-* and place-items-* out.
_PADS = re.compile(r"p[xytblrse]?-|\[padding[-:]")


def _rail_class_tokens(tag: str) -> list[str]:
    """Every class token the rail's className CAN render, conditionals included.

    _class_value answers what renders in EVERY state, which is what a positive guarantee
    needs and exactly wrong for a prohibition: `cn("...", compact && "!pl-0")` renders that
    override whenever compact is true, and an always-rendered reader never sees it. So this
    one reads the literals out of the whole expression on purpose. The two are not
    interchangeable, and the asymmetry is the point: "must always have X" and "must never
    have Y" cannot be answered by the same set.
    """
    key = "className="
    at = tag.index(key) + len(key)
    if tag[at] == '"':
        expression = tag[at : _skip_literal(tag, at)]
    else:
        expression = tag[at + 1 : _balanced(tag, at) - 1]

    tokens: list[str] = []
    i = 0
    while i < len(expression):
        if expression[i] in _QUOTES:
            end = _skip_literal(expression, i)
            tokens += expression[i + 1 : end - 1].split()
            i = end
        else:
            i += 1
    return tokens


def _rail_padding(tag: str) -> dict[str, str]:
    """This rail's padding properties mapped to the constant each one is set from.

    The property-to-constant binding, not merely the presence of some STACK_ name: setting
    paddingLeft from STACK_CARD_INSET_RIGHT renders a 16px left gutter while the 22px floor
    below still reads a 28px constant nothing applies, and the shadow clips anyway.
    """
    return {
        prop: const
        for prop, const in re.findall(r"\b(padding(?:Top|Bottom|Left|Right)): (\w+)", tag)
    }


def _rail_style_px(provider: str, name: str) -> int:
    """The px value of one of the rail's spacing constants, read from its definition."""
    found = re.search(rf"^const {name} = (\d+);", provider, re.MULTILINE)
    assert found, f"{name} is gone from provider.tsx, so the rail's spacing is unreadable"
    return int(found.group(1))


def _capped_rails(provider: str) -> int:
    """How many of those rails cap themselves to the viewport.

    The cap is the full viewport and the gutters are paid out of it, as inline px padding,
    so the cards keep the band they had. Reading the class alone stopped being enough when
    #11260 moved the gutters out of the class and into the style, so this reads both.
    """
    # `md:max-h-[100dvh]` contains the utility but leaves smaller viewports uncapped.
    # The desktop rail stops below the window chrome; there is none in the browser.
    caps = ("max-h-[100dvh]", "max-h-[calc(100dvh-var(--studio-window-chrome-top,0px))]")
    return sum(1 for rail in _corner_rails(provider) if any(_only_under(rail, c) for c in caps))


def test_the_class_matchers_tell_a_gated_rule_from_an_ungated_one():
    """The floor assertions are only as strong as these, so they are tested.

    A matcher that quietly says yes is how this file went wrong the first time:
    the checks read as layout guarantees and were substring searches.
    """
    gated = "max-[383px]:has-[[data-slot=update-release-notes]]:min-h-[calc(1px+2px)]"
    assert _split_variants(gated) == (
        ("max-[383px]", "has-[[data-slot=update-release-notes]]"),
        "min-h-[calc(1px+2px)]",
    ), "a bracketed variant's own colon splits the token"
    assert _applies(gated, "min-h-[calc(1px+2px)]", "max-[383px]")
    assert _applies(gated, "min-h-[calc(1px+2px)]", "max-[383px]", "has-[[data-slot=x]]") is False
    assert _applies("min-h-[calc(1px+2px)]", "min-h-[calc(1px+2px)]", "max-[383px]") is False
    assert _applies("min-h-0", "h-0") is False
    assert _applies("md:shrink-0", "shrink-0"), "the absence check must see a gated rule"
    assert _only_under("md:shrink-0", "shrink-0") is False
    assert _only_under("flex shrink-0 flex-col", "shrink-0")
    assert _only_under("has-[x]:min-h-4", "min-h-4", "has-[x]")
    assert _only_under("md:has-[x]:min-h-4", "min-h-4", "has-[x]") is False
    assert _only_under("has-[x]:min-h-4 min-h-4", "min-h-4", "has-[x]") is False
    assert _only_under("flex", "min-h-4", "has-[x]") is False, "absent is not satisfied"
    for important in ("!min-h-0", "min-h-0!"):
        assert _split_variants(important)[1] == "min-h-0"
        assert _applies(important, "min-h-0"), f"{important} escapes the prohibition"


def test_the_class_anchors_do_not_depend_on_any_order():
    """An anchor that needs an attribute or a branch to keep its place is the
    same brittleness one level up, so both are read structurally."""
    # An arrow function's `>` does not end the opening tag, and comparisons are not elements.
    for tag in (
        '<ul className="a b" data-testid="x" onClick={() => go()}>',
        '<ul onClick={() => go()} data-testid="x" className="a b">',
        '<ul disabled={count < limit} className="a b" data-testid="x">',
        '<ul disabled={count < limit} data-testid="x" className="a b">',
    ):
        assert _class_on_testid(tag, "x") == "a b", tag
    branch = _unpositioned_branch('positioned ? "fixed dark:bg-card" : cn("rail shrink-0")')
    assert "rail" in branch and "fixed" not in branch
    # Only classes rendered in every state count; a constant guard leaves text but renders nothing.
    assert _class_value('<div className="a b">', "x") == "a b"
    assert _class_value('<div className={cn("a b")}>', "x") == "a b"
    assert _class_value('<div className={cn("a b", open && "c")}>', "x").split() == ["a", "b"]
    assert _class_value('<div className={cn(false && "a", "b")}>', "x").split() == ["b"]
    assert _class_value('<div className={cn(open ? "a" : "z", "b")}>', "x").split() == ["b"]
    # Comments can hold an apostrophe or unmatched brace, and the banners' comment names `shrink-0`.
    for comment in ("// notes don't shrink", "/* an unmatched } is prose */"):
        tag = f'<ul {comment}\n className="a b" data-testid="x">'
        assert _class_on_testid(tag, "x") == "a b", comment
    blanked = _without_comments("/* shrink-0 */ flex")
    assert blanked.split() == ["flex"], "a class named in prose still reads as a class"
    assert len(blanked) == len("/* shrink-0 */ flex"), "blanking a comment moved every later index"
    # A `//` inside a literal is a URL, not a comment.
    url = '"bg-[url(https://example.com/a.svg)] flex" // gone'
    assert _without_comments(url).rstrip() == '"bg-[url(https://example.com/a.svg)] flex"'


def test_the_overlay_stack_fits_the_viewport():
    """The card's own cap does not account for a download list stacked beneath
    it, so the rail carries one of its own. A static cap, not a measured one:
    a rail whose height and offset are computed from whatever else is on screen
    is a rail that moves out of its corner (#8082 and the chain after it)."""
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    # Counted by layer, not literal z-index: the overlay rail reads its depth from Z_LAYER.
    stacks = provider.count("zIndex: Z_LAYER.OVERLAY_STACK")
    assert stacks, "the bottom-right overlay stack is gone"
    assert len(_corner_rails(provider)) == stacks, "a rail left its bottom-right corner"
    assert _capped_rails(provider) == stacks, "every stack is capped"
    panel = (FRONTEND / "features/hub/download-manager/download-manager-panel.tsx").read_text(
        encoding = "utf-8"
    )
    assert "flex min-h-0" in panel
    # The update card's header and buttons are fixed and only its notes yield, so it floors.
    _assert_floors_itself(WEB_BANNER.read_text(encoding = "utf-8"), "browser")
    # The floors can exceed the cap at large type sizes, so the rail must scroll.
    assert provider.count("overflow-y-auto") >= stacks, "a capped stack spills its cards"


def test_both_rails_are_still_pinned_to_the_bottom_right_corner():
    """The corner, asserted rather than assumed by the matcher.

    _corner_rails finds rails by testid now, so it would happily return a rail that had
    wandered to the top left. This is the claim the old regex used to make implicitly, kept
    explicit and kept failing for the right reason: it names the rail that moved.
    """
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    rails = _corner_rails(provider)
    assert len(rails) == 2, f"expected the browser and desktop rails, found {len(rails)}"
    for rail in rails:
        # _only_under, not _applies: a gated `md:fixed` would leave the rail out of the corner.
        # The rail's transparent gutters must not swallow clicks; cards opt back in with
        # pointer-events-auto.
        assert _only_under(
            rail, "pointer-events-none"
        ), f"the rail stopped passing clicks through: {rail!r}"
        assert _only_under(rail, "fixed"), f"the rail is no longer always viewport-fixed: {rail!r}"
        assert _only_under(rail, "bottom-0"), f"the rail can leave the bottom edge: {rail!r}"
        assert _only_under(rail, "right-0"), f"the rail can leave the right edge: {rail!r}"


def test_the_rail_gutters_come_out_of_the_cap_and_not_the_cards():
    """#11260's actual claim, which no class can carry any more.

    The rail caps at the whole viewport and pays its shadow gutters as inline px padding, so
    the band left for the cards is 100dvh less the two block gutters - the same band they had
    when the cap was written as calc(100dvh - 8px) and the gutter was 4px a side. px and not a
    spacing utility because those are rem and would scale the rail off its corner with the
    user's type size, which is the bug the comment above them is about.
    """
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    top = _rail_style_px(provider, "STACK_SHADOW_GUTTER_TOP")
    bottom = _rail_style_px(provider, "STACK_SHADOW_GUTTER_BOTTOM")
    left = _rail_style_px(provider, "STACK_SHADOW_GUTTER_LEFT")
    inset = _rail_style_px(provider, "STACK_CARD_INSET_RIGHT")

    assert top and bottom and left and inset, "a rail gutter went to zero, so shadows clip"
    # The deepest card shadow is 0 8px 28px -6px: 22px left of the card and 14px above it.
    assert left >= 22, f"the left gutter {left}px is inside the card shadow's 22px reach"
    assert top >= 14, f"the top gutter {top}px is inside the card shadow's 14px reach"
    expected = {
        "paddingTop": "STACK_SHADOW_GUTTER_TOP",
        "paddingBottom": "STACK_SHADOW_GUTTER_BOTTOM",
        "paddingLeft": "STACK_SHADOW_GUTTER_LEFT",
        "paddingRight": "STACK_CARD_INSET_RIGHT_PAST_PANEL",
    }
    assert re.search(
        r"const STACK_CARD_INSET_RIGHT_PAST_PANEL =\s*`calc\(\$\{STACK_CARD_INSET_RIGHT\}px \+ var\(",
        provider,
    ), "the rail's right padding is no longer the px card inset plus the panel width"
    openings = _rail_openings(provider)
    assert len(openings) == 2, f"expected the browser and desktop rails, found {len(openings)}"
    for tag in openings:
        # The exact binding per rail: a count of STACK_ names would accept the wrong constant.
        assert (
            _rail_padding(tag) == expected
        ), f"a rail's padding is not bound to its own constant: {_rail_padding(tag)}"
    # No padding utility in any spelling: the inline px style owns padding, and Tailwind padding is
    # either rem-scaled (#8082) or, with !important, overrides it.
    for tag in _rail_openings(provider):
        for token in _rail_class_tokens(tag):
            utility = _split_variants(token)[1]
            assert not _PADS.match(utility), (
                f"the rail carries the padding utility {token!r}; its padding is the inline "
                f"px style, and a class here is rem-scaled or !important-overrides it (#8082)"
            )


def test_the_desktop_stack_is_capped_like_the_browser_one():
    """The download panel shares the desktop stack, left uncapped before now."""
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    assert len(_corner_rails(provider)) == 2, "both rails sit in the bottom-right corner"
    assert _capped_rails(provider) == 2, "both stacks are capped"
    tauri = TAURI_BANNER.read_text(encoding = "utf-8")
    _assert_floored(tauri, _SCALED_FLOOR_TAURI, _NARROW_FLOOR_TAURI, "desktop")


def test_the_rail_offset_is_not_computed():
    """The rail used to place itself around the boxes in the frame store, so a
    composer growing by a line or a download row arriving moved it to the middle
    of the window, and a maximised monitor to the top. Its offset and cap must
    stay out of JS."""
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding = "utf-8")
    for banned in ("useStackGeometry", "stackGeometry", "stack.bottom", "stack.maxHeight"):
        assert banned not in provider, f"the rail is placed from JS again ({banned})"
    store = (FRONTEND / "features/settings/stores/monitor-frame-store.ts").read_text(
        encoding = "utf-8"
    )
    assert "stackBottomInset" not in store, "the dodge arithmetic is back in the frame store"


def test_desktop_notes_are_not_keyed_by_the_pinned_backend_version():
    """The banner asks with the Unsloth version it offers. `pypi_version` stays
    in latest.json as the backend pin preflight checks, not a notes key."""
    banner = TAURI_BANNER.read_text(encoding = "utf-8")
    assert "info?.version?.replace(LEADING_V" in banner
    assert "pypiVersion" not in banner, "notes are no longer keyed by the backend release"
    workflow = (REPO / ".github/workflows/release-desktop.yml").read_text(encoding = "utf-8")
    assert "'pypi_version': os.environ['PYPI_VERSION']" in workflow
    rust = (REPO / "studio/src-tauri/src/desktop_update_policy.rs").read_text(encoding = "utf-8")
    assert "pypi_version: Option<String>" in rust
    hook = NOTES_HOOK.parent.joinpath("use-tauri-update.ts").read_text(encoding = "utf-8")
    assert "rawPypiVersion(update.rawJson)" in hook


def test_one_slow_read_cannot_outlast_the_fetch_budget(notes_module):
    """The socket timeout is per operation, so two slow reads doubled the wait."""
    source = MODULE.read_text(encoding = "utf-8")
    assert "_limit_read(response, remaining)" in source
    assert "sock.settimeout(max(remaining, _RELEASES_MIN_READ_SECONDS))" in source


def test_a_heading_indented_into_a_list_item_is_not_a_release(notes_module):
    """CommonMark keeps a heading at the item's content column inside the item.
    Checked against markdown-it (commonmark preset)."""
    text = "## 1.0\n\n- Example:\n  ## 9.9.9\n\n- after\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0"]
    body = find_section(notes_module, text, "1.0").body
    assert "9.9.9" in body and "after" in body
    left = "## 1.0\n\n- Example:\n ## 2.0\n"
    assert [e.version for e in parse_sections(notes_module, left)] == ["1.0", "2.0"]


def test_a_closed_list_stops_holding_headings(notes_module):
    """Only an open item nests a heading, so a dedented block hands it back."""

    def versions(text):
        return [e.version for e in parse_sections(notes_module, text)]

    assert versions("## 1.0\n\n- Example:\n\nText.\n\n  ## 2.0\n") == ["1.0", "2.0"]
    assert versions("## 1.0\n\n- Example:\n## 2.0\n  ## 3.0\n") == ["1.0", "2.0", "3.0"]
    assert versions("## 1.0\n\n- Example:\n  Text.\n---\n  ## 2.0\n") == ["1.0", "2.0"]
    assert versions("## 1.0\n\n- Example:\n```\n```\n  ## 2.0\n") == ["1.0", "2.0"]
    assert versions("## 1.0\n\n-\n\n  ## 2.0\n") == ["1.0", "2.0"]


def test_a_version_line_is_not_an_ordered_list_marker(notes_module):
    """`2.` needs whitespace after it, or every setext version reads as an item."""
    text = "2.0\n---\n\n- new\n\n1.0\n---\n\n- old\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["2.0", "1.0"]
    # An ordered item interrupts a paragraph only when it starts at 1.
    assert [
        e.version for e in parse_sections(notes_module, "## 1.0\n\nText.\n9) one\n   ## 2.0\n")
    ] == ["1.0", "2.0"]


def test_a_wrapped_setext_heading_is_still_a_release(notes_module):
    """CommonMark promotes the whole paragraph, so a wrapped heading keeps its
    version in the first token. Reading only the last line lost the release."""
    text = "2026.7.5 - Release\nJuly 25\n---\n\n- note\n"
    entries = parse_sections(notes_module, text)
    assert [e.version for e in entries] == ["2026.7.5"]
    assert entries[0].body == "- note"
    assert "July 25" not in entries[0].body


def test_a_lowercase_declaration_is_not_a_raw_block(notes_module):
    """Only `<!` plus an uppercase letter opens one, so `<!note` is prose."""
    assert [e.version for e in parse_sections(notes_module, "<!note\n\n## 1.0\n\n- real\n")] == [
        "1.0"
    ]
    assert [
        e.version for e in parse_sections(notes_module, "<!DOCTYPE\n## 9.9.9\n>\n\n## 1.0\n")
    ] == ["1.0"]
    assert "<![A-Z]" in PREVIEW.read_text(encoding = "utf-8")


def test_link_resolver_reads_html_containers_the_way_the_others_do():
    """A `<details>` or `<div>` with no blank line inside is a type 6 block, so
    its contents are literal and a fence in it is not a fence, which stopped
    every link below from resolving. The parser and the preview already apply
    the type 6 and 7 rules, so the resolver has to share them."""
    links = LINKS.read_text(encoding = "utf-8")
    for source in (PREVIEW, LINKS):
        text = source.read_text(encoding = "utf-8")
        assert "HTML_BLOCK_TAGS" in text and "HTML_TAG_ONLY_LINE" in text
    assert "inHtmlBlock = !!container.trim()" in links
    # Type 7 cannot interrupt a paragraph, so prose above it keeps its links.
    assert "return !afterParagraph && HTML_TAG_ONLY_LINE.test(line);" in links


def test_an_escaped_mark_makes_an_image_a_link():
    """`\\![alt](path)` renders as a link, so it resolves to the blob host."""
    links = LINKS.read_text(encoding = "utf-8")
    assert 'const image = bang === "!" && !isEscaped(line, offset);' in links
    assert "isEscaped(line, match.index)" in links


def test_only_markdown_line_endings_split_the_changelog(notes_module):
    """str.splitlines also breaks on U+2028, U+2029, NEL, vertical tab and form
    feed, none of which end a CommonMark line: one in prose indexed a release
    the renderer never shows and truncated the notes above it."""
    text = "## 2.0\n\nnote with a separator  ## 9.9.9\n\n## 1.0\n\n- old\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["2.0", "1.0"]
    entry = find_section(notes_module, text, "2.0")
    assert entry is not None and "9.9.9" in entry.body
    for separator in (" ", "\x85", "\x0b", "\x0c"):
        broken = f"## 2.0\n\nnote{separator}## 9.9.9\n\n## 1.0\n\n- old\n"
        assert [e.version for e in parse_sections(notes_module, broken)] == ["2.0", "1.0"]
    for ending in ("\n", "\r\n", "\r"):
        real = f"## 2.0{ending}{ending}- new{ending}{ending}## 1.0{ending}{ending}- old{ending}"
        assert [e.version for e in parse_sections(notes_module, real)] == ["2.0", "1.0"]


def test_link_resolver_reads_comments_before_fences():
    """A fence delimiter inside an HTML comment is not a fence: reading it as one
    left the fence open, so every link below went unresolved. The order matters
    both ways, so a comment opener inside a real fence is not a comment."""
    links = LINKS.read_text(encoding="utf-8")
    # Fence state is read before comments are masked, the order the collapsed preview uses.
    assert "const fenceSource = inComment\n      ? null\n      : FENCE.exec(" in links
    # Masking happens only after the in-fence early return.
    fence_return = links.index("// Fenced content is literal")
    assert links.index("const [line, stillInComment, stillRunOn] = maskComments(") > fence_return
    assert "const spans = [...codeSpans(masked), ...comments].sort(" in links


def test_preview_heading_and_quote_markers_follow_the_backend_rule():
    """An ATX heading needs an ASCII space, tab or line end after the marker;
    `\\s` also matches a non-breaking space, so prose was read as a heading and
    dropped. A blockquote marker takes at most three leading spaces, or an
    indented sample holding "> - sample output" sheds its indent into the
    summary."""
    src = PREVIEW.read_text(encoding="utf-8")
    assert "const HEADING = /^#{1,6}(?:[ \\t]|$)/;" in src
    assert "const HEADING_LINE = /^ {0,3}#{1,6}(?:[ \\t]|$)/;" in src
    assert "const BLOCKQUOTE = /^ {0,3}>[ \\t]?/;" in src
    backend = MODULE.read_text(encoding="utf-8")
    assert "^ {0,3}(?P<hashes>#{1,6})(?:[ \\t]+(?P<title>.*?))?[ \\t]*$" in backend


def test_preview_collects_labels_only_from_real_definitions():
    """A definition-shaped line inside indented code or a deep fence is literal,
    so recording its label made toPlainText strip brackets the expanded view
    keeps. The pre-scan skips the same code the collector pass skips; a real
    definition takes at most three spaces, so the indent test cannot reject one."""
    src = PREVIEW.read_text(encoding="utf-8")
    scan = src.index("const labels = new Set<string>();")
    collect = src.index("let deepFence: string | null = null;")
    prescan = " ".join(src[scan:collect].split())
    assert "let labelFence: string | null = null;" in prescan
    assert "if (line.indent - line.column >= INDENTED_CODE_INDENT) { continue; }" in prescan
    assert "endsDeepFence(labelFence, labelColumn, line)" in prescan


def test_an_html_block_to_the_left_of_a_list_item_closes_it(notes_module):
    """Types 1 to 6 interrupt a paragraph, so an unindented <div> after "- item"
    closes the item and a shallowly indented "## 2.0" below it is a real heading.
    Read as a lazy continuation, the item stayed open and swallowed it."""
    text = "## 3.0\n\n- item\n<div>\nhidden\n</div>\n\n  ## 2.0\n\n- two\n\n## 1.0\n\n- one\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["3.0", "2.0", "1.0"]
    nested = "## 3.0\n\n- item\n\n  ## 2.0\n\n- two\n\n## 1.0\n\n- one\n"
    assert [e.version for e in parse_sections(notes_module, nested)] == ["3.0", "1.0"]
    lazy = "## 3.0\n\n- item\ncontinued\n\n  ## 2.0\n\n## 1.0\n\n- one\n"
    assert [e.version for e in parse_sections(notes_module, lazy)] == ["3.0", "1.0"]


def test_the_download_panel_can_shrink_inside_the_capped_stack():
    """The stack is capped to the viewport and a flex item defaults to
    min-height:auto, so this wrapper could not shrink and the cap was absorbed by
    the fixed update card rather than the scrolling download list. Only the
    shared-stack branch needs it; standalone is fixed and not a flex item."""
    panel = (FRONTEND / "features/hub/download-manager/download-manager-panel.tsx").read_text(
        encoding="utf-8"
    )
    assert 'positioned ? "fixed bottom-4 right-4 z-50" : "flex min-h-0 justify-end"' in panel
    provider = (FRONTEND / "app/provider.tsx").read_text(encoding="utf-8")
    stacks = provider.count("zIndex: Z_LAYER.OVERLAY_STACK")
    assert _capped_rails(provider) == stacks, "the cap this has to absorb"


@pytest.fixture(scope="module")
def run_scanner(tmp_path_factory):
    """Run the frontend's markdown scanners under node.

    Their job is to classify a line the way a CommonMark renderer would, which
    only a real run can show. The "@/lib" alias resolves through Vite, not node,
    so the copies have it rewritten."""
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
    """CommonMark measures indentation from the container (spec 0.31.2 section
    5.2), so under "- Details:" a four-space line is two columns in: a paragraph
    holding a link. Measuring from the margin called it code (section 4.4) and
    left the destination relative to Unsloth's own origin."""
    resolved = run_scanner("links", "- Details:\n\n    [guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in resolved
    code = run_scanner("links", "- Added.\n\n      [guide](docs/a.md)\n")
    assert "[guide](docs/a.md)" in code and "github.com" not in code
    top = run_scanner("links", "Intro.\n\n    [guide](docs/a.md)\n")
    assert "[guide](docs/a.md)" in top and "github.com" not in top


def test_an_indented_fence_does_not_swallow_the_bullets_below_it(run_scanner):
    """A four-space line at document level is indented code, and a top-level
    bullet does not continue it. Promoting it to a list-contained fence left a
    block open with no closer, so every bullet after it was skipped."""
    swallowed = "Example:\n\n    ```\n\n- Added the exporter\n- Fixed the crash\n"
    assert preview_leads(run_scanner("preview", swallowed)) == [
        "Added the exporter",
        "Fixed the crash",
    ]
    assert preview_leads(run_scanner("preview", "    ```\n\n- Added the exporter\n")) == [
        "Added the exporter"
    ]
    nested = "- a\n  - b\n    ```\n    - not a bullet\n    ```\n\n- Added tests\n"
    assert preview_leads(run_scanner("preview", nested)) == ["a", "Added tests"]


def test_a_table_only_release_previews_as_nothing(run_scanner):
    """A GFM table renders as a grid, so it previews as nothing and the panel
    stays collapsed. Falling through to the prose collector put the raw
    "| Change | Detail | | --- | --- |" delimiters in the popup instead."""
    table = "| Change | Detail |\n| --- | --- |\n| Exporter | Added GGUF |\n"
    assert run_scanner("preview", table)["items"] == []
    assert preview_leads(run_scanner("preview", f"Some prose.\n\n{table}")) == ["Some prose."]
    assert preview_leads(run_scanner("preview", f"{table}- Added tests\n")) == ["Added tests"]
    # Mismatched header and delimiter widths are not a table on GitHub, so both lines are prose.
    assert preview_leads(run_scanner("preview", "| a | b |\n| --- |\n")) == ["| a | b | | --- |"]


def test_a_fence_inside_a_list_item_ends_with_the_item(notes_module):
    """A fence is scoped to its container (spec 0.31.2 section 4.5), so with no
    closer it ends with the item and a dedented "## 2.0" is a real heading.
    Document-wide state kept the block open and hid every release below it."""
    text = "## 1.0\n\n- item\n  ```\n\n## 2.0\n\n- two\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0", "2.0"]
    top = "## 1.0\n\n```\n\n## 2.0\n\n- two\n"
    assert [e.version for e in parse_sections(notes_module, top)] == ["1.0"]
    closed = "## 1.0\n\n- Run:\n  ```bash\n  ## 9.9.9\n  ```\n\n## 2.0\n\n- two\n"
    assert [e.version for e in parse_sections(notes_module, closed)] == ["1.0", "2.0"]
    assert find_section(notes_module, text, "2.0").body == "- two"


def test_stripping_comments_stays_linear_in_the_code_spans(notes_module):
    """The comment scanner restarted its code-span search per opener, so N spans
    cost N squared. A 203 KiB line is well inside the 2 MiB accepted, and notes
    are reparsed on every request, so one held a worker for over ten seconds."""
    # A quarter of the budget form's size: the big leg is the 16_000 it was checked at.
    spans = 4_000
    assert len("`a` <!--x--> " * (spans * 4)) < notes_module.RELEASES_MAX_BYTES
    visible, in_comment = assert_linear(
        lambda text: notes_module._strip_comments(text, False, False),
        lambda n: "`a` <!--x--> " * n,
        "comment stripping",
        spans,
    )
    assert in_comment is False
    assert "<!--" not in visible and visible.count("`a`") == spans * 4


def test_the_three_scanners_share_one_list_column_rule():
    """The parser and both frontend scanners must classify a line the same way;
    drifting on indentation is what put a paragraph link inside a code block.
    The frontend pair shares one module, ported from the backend's tracker."""
    shared = LIST_COLUMNS.read_text(encoding="utf-8")
    assert "export function openLists(" in shared
    assert "_open_lists" in shared, "the backend function this mirrors"
    for source in (PREVIEW, LINKS):
        src = source.read_text(encoding="utf-8")
        assert 'from "@/lib/markdown-list-columns"' in src
        assert "openLists(" in src
    backend = MODULE.read_text(encoding="utf-8")
    assert "_indent_width(visible) - column >= 4" in backend
    assert "indentWidth(structure) - column >= INDENTED_CODE_INDENT" in LINKS.read_text(
        encoding="utf-8"
    )


def test_a_failed_fetch_keeps_retry_reachable():
    """A release with no notes is ready and a failed fetch is error; only the
    error is retryable, and rendering there would replace the Retry button."""
    src = " ".join(PANEL.read_text(encoding="utf-8").split())
    assert "const source = notes?.matched ? notes.markdown : null;" in src
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
    """CommonMark opens an HTML block (spec 0.31.2 section 4.6, type 2) only when
    the line begins with `<!--`; one mid-sentence is inline and cannot outlive
    its block. Carrying the unclosed state on masked every link below it."""
    repo = "https://github.com/unslothai/unsloth/blob/main/docs/a.md"
    item = run_scanner("links", "- Type <!-- to begin a comment\n- See [docs](docs/a.md)\n")
    assert repo in item
    paragraph = run_scanner("links", "Type <!-- to begin\n\nSee [docs](docs/a.md)\n")
    assert repo in paragraph
    spanned = run_scanner("links", "- Wrap in `<!--` and `-->`\n- See [docs](docs/a.md)\n")
    assert repo in spanned
    block = run_scanner("links", "<!-- staged\n- See [docs](docs/a.md)\n-->\n")
    assert repo not in block
    closer = run_scanner("links", "<!-- staged\n--> See [docs](docs/a.md)\n")
    assert repo not in closer


def test_a_bare_level_two_marker_ends_the_release(notes_module, run_scanner):
    """An ATX heading's opening sequence may be followed by the end of the line
    (spec 0.31.2 section 4.2), so a bare `##` is an empty level-two heading.
    Requiring whitespace kept everything below it inside the release above."""
    text = "## 2.0\n\n- new thing\n\n##\n\n- SECRET: not part of 2.0\n"
    entry = find_section(notes_module, text, "2.0")
    assert "new thing" in entry.body
    assert "SECRET" not in entry.body
    assert [e.version for e in parse_sections(notes_module, text)] == ["2.0"]
    prose = "## 2.0\n\n- new thing\n\n##x\n\n- still 2.0\n"
    assert "still 2.0" in find_section(notes_module, prose, "2.0").body
    preview = run_scanner("preview", "- new thing\n##\nUnrelated scratch notes\n")
    assert preview_leads(preview) == ["new thing"]


def test_a_comment_between_bullets_closes_the_list(notes_module, run_scanner):
    """A comment is an HTML block (spec 0.31.2 section 4.6, type 2), so one at
    the margin under a bullet closes the list. Blanking the line before list
    tracking saw it reads as a blank line, which leaves the item open and made
    the release heading below look like nested content."""
    text = "## 1.0\n\n- old item\n<!-- separator -->\n  ## 2.0\n\n- new item\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0", "2.0"]
    assert "new item" not in find_section(notes_module, text, "1.0").body
    assert "new item" in find_section(notes_module, text, "2.0").body
    nested = "## 1.0\n\n- old item\n  <!-- separator -->\n  ## 2.0\n\n- new item\n"
    assert [e.version for e in parse_sections(notes_module, nested)] == ["1.0"]
    code = run_scanner("links", "- old item\n<!-- separator -->\n    [guide](docs/a.md)\n")
    assert "[guide](docs/a.md)" in code and "github.com" not in code
    prose = run_scanner("links", "- old item\n  <!-- separator -->\n    [guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in prose
    preview = run_scanner(
        "preview",
        "- Details:\n<!-- separator -->\n    ```\n    - hidden sample\n- Real second item\n",
    )
    assert preview_leads(preview) == ["Details:", "Real second item"]


def test_a_parenthesised_link_destination_still_resolves(run_scanner):
    """A destination may hold parentheses while they balance (spec 0.31.2 section
    6.3), so `[x]((draft).md)` points at `(draft).md`. Stopping at the first
    paren matched an empty destination and left the link relative."""
    leading = run_scanner("links", "[details]((draft).md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/(draft).md" in leading
    image = run_scanner("links", "![shield]((badge).png)\n")
    assert "https://raw.githubusercontent.com/unslothai/unsloth/main/(badge).png" in image
    middle = run_scanner("links", "[api](docs/(v2)/api.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/(v2)/api.md" in middle
    unbalanced = run_scanner("links", "[x](a(b.md)\n")
    assert unbalanced == "[x](a(b.md)\n"
    closed = run_scanner("links", "[x](a(b.md))\n")
    assert "https://github.com/unslothai/unsloth/blob/main/a(b.md)" in closed
    nested = run_scanner("links", "[x](((draft)).md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/((draft)).md" in nested
    deep = run_scanner("links", "![shot](((((v2))))).png)\n")
    assert "https://raw.githubusercontent.com/unslothai/unsloth/main/((((v2))))" in deep
    across = run_scanner("links", "[x](((a).md\n[y](docs/y.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/y.md" in across
    assert "[x](((a).md" in across


def test_a_fence_inside_a_container_still_hides_its_sample(run_scanner):
    """A fence is measured from its container (spec 0.31.2 section 4.5), so
    `> ~~~` and one under a nested bullet both open a fence. Reading the margin
    never saw them, so a link in a code block was rewritten into verbatim text."""
    quoted = run_scanner("links", "> ~~~\n> [guide](docs/a.md)\n> ~~~\n")
    assert "[guide](docs/a.md)" in quoted and "github.com" not in quoted
    nested = run_scanner("links", "- a\n  - b\n    ~~~\n    [x](docs/x.md)\n    ~~~\n")
    assert "[x](docs/x.md)" in nested and "github.com" not in nested
    uneven = run_scanner("links", "> ```\n> [guide](docs/a.md)\n> ````\n")
    assert "[guide](docs/a.md)" in uneven and "github.com" not in uneven
    left = run_scanner("links", "> ~~~\n[guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in left
    dedented = run_scanner("links", "- a\n  ~~~\n[guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in dedented
    document = run_scanner("links", "~~~\n> [guide](docs/a.md)\n~~~\n")
    assert "[guide](docs/a.md)" in document and "github.com" not in document
    code = run_scanner("links", "- Details:\n\n      ~~~\n      [guide](docs/a.md)\n")
    assert "[guide](docs/a.md)" in code and "github.com" not in code


def test_an_html_block_inside_a_container_is_literal_too(run_scanner):
    """Type 1 and type 6 blocks are measured from their container too, so a
    `<details>` under a nested bullet and a `<pre>` in a quote are verbatim.
    Missing the opener rewrote the literal examples inside them."""
    nested = run_scanner("links", "- a\n  - b\n    <details>\n    [x](docs/x.md)\n    </details>\n")
    assert "[x](docs/x.md)" in nested and "github.com" not in nested
    quoted = run_scanner("links", "> <pre>\n> [x](docs/x.md)\n> </pre>\n")
    assert "[x](docs/x.md)" in quoted and "github.com" not in quoted
    dedented = run_scanner("links", "- a\n  - b\n    <details>\n[x](docs/x.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/x.md" in dedented
    blank = run_scanner("links", "> <details>\n>\n> [x](docs/x.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/x.md" in blank


def test_an_underline_left_of_an_item_is_lazy_text_of_it(notes_module, run_scanner):
    """A setext underline may never be a lazy continuation (spec 0.31.2 section
    4.3), so `===` left of an open item is more of the item's paragraph.
    Rejecting it ended the list and promoted the nested "## 2.0" to a release."""
    nested = "## 1.0\n- old note\n===\n  ## 2.0\n- new\n"
    assert [e.version for e in parse_sections(notes_module, nested)] == ["1.0"]
    # A row of dashes is a thematic break, which closes the item.
    broken = "## 1.0\n- old note\n---\n  ## 2.0\n"
    assert [e.version for e in parse_sections(notes_module, broken)] == ["1.0", "2.0"]
    apart = "## 1.0\n- old note\n\n===\n  ## 2.0\n"
    assert [e.version for e in parse_sections(notes_module, apart)] == ["1.0", "2.0"]
    resolved = run_scanner("links", "- Details:\n===\n\n    [guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in resolved


def test_a_quote_keeps_its_paragraph_to_itself(notes_module, run_scanner):
    """A marker written outside a blockquote is not text of the quote's
    paragraph, so `2. item` under `> quote` opens a list even though an ordered
    marker past 1 may not interrupt one (spec 0.31.2 section 5.2). Lending the
    paragraph to the document left the list closed and the heading exposed."""
    quoted = "## 1.0\n> quote\n2. item\n   ## 2.0\n- new\n"
    assert [e.version for e in parse_sections(notes_module, quoted)] == ["1.0"]
    heading = "## 1.0\n> # inner\n2. item\n   ## 2.0\n"
    assert [e.version for e in parse_sections(notes_module, heading)] == ["1.0"]
    lazy = "## 1.0\n> quote\ntext\n2. item\n   ## 2.0\n"
    assert [e.version for e in parse_sections(notes_module, lazy)] == ["1.0"]
    prose = "## 1.0\nprose\n2. item\n   ## 2.0\n"
    assert [e.version for e in parse_sections(notes_module, prose)] == ["1.0", "2.0"]
    assert preview_leads(run_scanner("preview", "> quote\n2. item\n")) == ["item"]


def test_indented_code_before_an_ordered_marker_still_opens_a_list(notes_module):
    """An indented code block ends at the first line not indented enough, and no
    paragraph is open, so `2. item` opens a list whatever its start number.
    Reading it as code text left the list closed and the nested heading exposed."""
    joined = "## 1.0\n\n    code\n2. item\n   ## 2.0\n- new\n"
    assert [e.version for e in parse_sections(notes_module, joined)] == ["1.0"]
    apart = "## 1.0\n\n    code\n\n2. item\n   ## 2.0\n- new\n"
    assert [e.version for e in parse_sections(notes_module, apart)] == ["1.0"]
    inside = "## 1.0\n\n    code\n    - item\n  ## 2.0\n"
    assert [e.version for e in parse_sections(notes_module, inside)] == ["1.0", "2.0"]


def test_a_fence_written_as_an_item_first_content_opens_in_that_item(run_scanner):
    """A block straight after a marker is the item's own first content, measured
    from where that content starts (spec 0.31.2 section 5.2), so "- ```md" opens
    a fence. Reading the whole line never saw one, so the sample below was
    treated as prose and rewritten, and its info string became a headline."""
    sample = run_scanner("links", "- ```md\n  [example](docs/a.md)\n  ```\n")
    assert "[example](docs/a.md)" in sample and "github.com" not in sample
    ordered = run_scanner("links", "1. ~~~\n   [example](docs/a.md)\n   ~~~\n")
    assert "[example](docs/a.md)" in ordered and "github.com" not in ordered
    preview = run_scanner("preview", "- ```md\n  sample text\n  ```\n- Added tests\n")
    assert preview_leads(preview) == ["Added tests"]
    padded = run_scanner("links", "-     ```\n  [example](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in padded
    lazy = run_scanner("links", "Intro.\n2. ```\n[guide](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in lazy


def test_an_html_block_ends_with_the_item_it_was_written_in(notes_module, run_scanner):
    """An HTML block takes no lazy continuation line, so one opened on an item's
    continuation line ends with the item, as a fence there does. Ending it only
    on a blank line let it swallow the next release heading."""
    text = "## 1.0\n\n- item\n\n  <div>\n## 2.0\n\n- new thing\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0", "2.0"]
    assert "new thing" in find_section(notes_module, text, "2.0").body
    raw = "## 1.0\n\n- item\n\n  <pre>\n## 2.0\n\n- new thing\n"
    assert [e.version for e in parse_sections(notes_module, raw)] == ["1.0", "2.0"]
    nested = "## 1.0\n\n- item\n\n  <div>\n  ## 2.0\n"
    assert [e.version for e in parse_sections(notes_module, nested)] == ["1.0"]
    preview = run_scanner("preview", "- item\n\n  <div>\n- Added tests\n")
    assert preview_leads(preview) == ["item", "Added tests"]
    marked = "## 1.0\n\n- <div>\n## 2.0\n\n- new thing\n"
    assert [e.version for e in parse_sections(notes_module, marked)] == ["1.0", "2.0"]


def test_a_comment_may_close_on_a_later_line_of_its_paragraph(run_scanner):
    """A comment written mid-sentence belongs to its paragraph, so its `-->` may
    arrive on a later line of it and everything between renders as nothing.
    Ending it at its own line left a backtick inside pairing with a real one
    below, hiding a link, and left the preview quoting hidden text."""
    carried = run_scanner("links", "Note <!-- ` open\nstill --> see [d](docs/a.md) and `x`\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in carried
    inside = run_scanner("links", "Note <!-- see [c](docs/c.md)\nmore --> end\n")
    assert "[c](docs/c.md)" in inside and "github.com" not in inside
    preview = run_scanner(
        "preview", "- Added X <!-- TODO: rewrite\n  this properly -->\n- Second\n"
    )
    assert preview_leads(preview) == ["Added X", "Second"]
    broken = run_scanner("links", "Note <!-- open\n\nsecret --> end [d](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in broken
    headed = run_scanner("links", "Note <!-- open\n## 2.0 --> end [d](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in headed
    assert preview_leads(run_scanner("preview", "Note <!-- open\n\n- Second\n")) == ["Second"]


def test_only_punctuation_is_escapable_in_a_link_destination(run_scanner):
    """CommonMark escapes ASCII punctuation and nothing else (spec 0.31.2 section
    2.4), so the backslash in `docs\\alpha.md` is a character of the path.
    Dropping it rewrote a Windows or namespaced path to the wrong file."""
    kept = run_scanner("links", "[guide](docs\\alpha.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs%5Calpha.md" in kept
    escaped = run_scanner("links", "[guide](docs\\\\alpha.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs%5Calpha.md" in escaped
    paren = run_scanner("links", "[guide](a\\(b.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/a(b.md" in paren
    spaced = run_scanner("links", "[guide](a\\ b.md)\n")
    assert spaced == "[guide](a\\ b.md)\n"


def test_one_definition_does_not_hide_the_next(run_scanner):
    """Definitions may run consecutively (spec 0.31.2 section 4.7) and none is
    paragraph text, so none opens a paragraph the next cannot interrupt.
    Counting one as prose left every definition after the first unresolved.
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
    run = run_scanner("links", "[a]: docs/a.md\n[b]: docs/b.md\n[c]: docs/c.md\n")
    assert run.count("https://github.com/unslothai/unsloth/blob/main/docs/") == 3
    prose = run_scanner("links", "[a]: docs/a.md\nintro\n[b]: docs/b.md\n")
    assert "[b]: docs/b.md" in prose


def test_a_comment_closed_on_its_own_line_still_closes(run_scanner):
    """A multiline comment is usually closed by a `-->` on a line of its own, and
    a wrapped line may open with emphasis. Reading any leading punctuation as a
    new block meant neither continued the paragraph, so the comment never closed
    and the popup showed the author's internal note."""
    closer = run_scanner(
        "preview",
        "- DoRA training is available in Unsloth. <!-- TODO confirm the exact\n"
        "  flag name before release\n-->\n",
    )
    assert preview_leads(closer) == ["DoRA training is available in Unsloth."]
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
    broken = run_scanner("links", "Note <!-- open\n## 2.0\nsecret --> [d](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in broken
    item = run_scanner("links", "Note <!-- open\n- bullet\nsecret --> [d](docs/a.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/a.md" in item


def test_a_comment_written_as_an_item_first_content_is_a_block(notes_module, run_scanner):
    """A comment is an HTML block (spec 0.31.2 section 4.6, type 2), so one
    written as an item's first content opens inside that item, as a fence does.
    Looking for the opener at the margin let a marker in front of it hide the
    block, so the resolver rewrote hidden text and the preview quoted it."""
    item = run_scanner("links", "- <!-- new --> AMD support, see [the guide](docs/amd.md)\n")
    assert item == "- <!-- new --> AMD support, see [the guide](docs/amd.md)\n"
    for text in (
        "* <!-- new --> see [the guide](docs/amd.md)\n",
        "1. <!-- new --> see [the guide](docs/amd.md)\n",
        "- outer\n  - <!-- new --> see [the guide](docs/amd.md)\n",
    ):
        assert "github.com" not in run_scanner("links", text)
    multiline = run_scanner("links", "- <!-- hidden\n  [a](docs/x.md)\n  -->\n")
    assert "[a](docs/x.md)" in multiline and "github.com" not in multiline
    dedented = run_scanner("links", "- <!-- hidden\n[a](docs/x.md)\n")
    assert "https://github.com/unslothai/unsloth/blob/main/docs/x.md" in dedented
    preview = run_scanner("preview", "- <!-- new --> hidden note\n- Real bullet\n")
    assert preview_leads(preview) == ["Real bullet"]
    text = "## 1.0\n\n- <!-- hidden\n\n  ## 2.0\n"
    assert [e.version for e in parse_sections(notes_module, text)] == ["1.0"]


@pytest.mark.parametrize(
    "tag,kept,dropped",
    [
        (
            "v0.1.60-beta",
            ["Meta has released Muse Glimmer", "[Run Muse Glimmer]"],
            [
                "## Updating / installing Unsloth",
                "install.sh",
                "## What's Changed",
                "## New Contributors",
                "Full Changelog",
            ],
        ),
        (
            # The install block sits between content sections; cutting there loses the Keyv notice.
            "v0.1.526-beta",
            [
                "## August 7th Update",
                "## DeepSeek V4 Flash 0731 + DSpark",
                "## Kimi K3",
                "Keyv security incident",
            ],
            ["## Updating / installing Unsloth", "install.ps1", "## What's Changed"],
        ),
        (
            "v0.1.501-beta",
            [
                "## 23rd July Update",
                "## Train LLMs Locally on AMD",
                "### Safer Agents",
            ],
            [
                "## Updating / installing Unsloth",
                "## What's Changed in Unsloth",
                "## What's Changed in Unsloth-Zoo",
                "## 23rd July Update Unsloth changelog",
            ],
        ),
        (
            # The install block's platform headings are siblings, so level alone does not end it.
            "v0.1.43-beta",
            ["## Mac Updates", "## Windows Updates", "## Blackwell GPUs Update"],
            [
                "To update Unsloth or install a new Unsloth Studio",
                "### macOS, Linux, WSL:",
                "### Windows:",
                "irm https://unsloth.ai/install.ps1",
            ],
        ),
        (
            # All level 3; the install block is introduced by a paragraph, not a heading.
            "v0.1.471-beta",
            ["### Better context length algorithm", "### Training & General Fixes"],
            [
                "## What's Changed",
                "## New Contributors",
                "To update Unsloth or install a new Unsloth Studio",
                "Ensure your version is",
                "curl -fsSL https://unsloth.ai/install.sh",
                "irm https://unsloth.ai/install.ps1",
            ],
        ),
    ],
)
def test_real_release_bodies_keep_their_announcement(notes_module, tag, kept, dropped):
    body = (BODIES / f"{tag}.md").read_text(encoding="utf-8")
    stripped = notes_module.strip_release_body(body)
    for text in kept:
        assert text in stripped, f"{tag} lost {text!r}"
    for text in dropped:
        assert text not in stripped, f"{tag} kept {text!r}"


def test_the_build_provenance_the_workflow_appends_is_dropped(notes_module):
    """release-desktop.yml appends this block, so it arrives under announcements."""
    body = "The announcement.\n\n### Build provenance\n\n- workflow run 123\n"
    stripped = notes_module.strip_release_body(body)
    assert stripped == "The announcement."


def test_a_generated_heading_inside_a_sample_is_not_one(notes_module):
    """A release documenting the notes format writes `## What's Changed` in a
    fence, and reading it as the footer would cut the announcement there."""
    body = (
        "The announcement.\n\n"
        "```md\n## What's Changed\n```\n\n"
        "## Kimi K3\n\n- still here\n\n"
        "## What's Changed\n\n* a pull request\n"
    )
    stripped = notes_module.strip_release_body(body)
    assert "```md" in stripped and "## Kimi K3" in stripped and "still here" in stripped
    assert "a pull request" not in stripped


def test_a_heading_that_only_reads_like_boilerplate_is_kept(notes_module):
    """A reviewed list, not a substring sweep: these titles only share words."""
    for title in (
        "## What changed in Gemma 4",
        "## Updating models is now 2x faster",
        "## Installing a LoRA from the Hub",
    ):
        body = f"Intro.\n\n{title}\n\n- a real change\n"
        assert "a real change" in notes_module.strip_release_body(body), title


def test_an_install_block_introduced_by_a_paragraph_is_dropped(notes_module):
    """`v0.1.471-beta` writes the install sentence with no hashes. A paragraph
    has no level, so its block runs to the next non-platform heading."""
    body = (
        "The announcement.\n\n"
        "To update Unsloth or install a new Unsloth Studio, you must use the below.\n"
        "Ensure your version is `2026.6.9` for the latest.\n\n"
        "MacOS, Linux, WSL:\n```\ncurl -fsSL https://unsloth.ai/install.sh | sh\n```\n\n"
        "Windows:\n```\nirm https://unsloth.ai/install.ps1 | iex\n```\n\n"
        "### Better context length algorithm\n\n- a real change\n"
    )
    stripped = notes_module.strip_release_body(body)
    assert stripped == (
        "The announcement.\n\n### Better context length algorithm\n\n- a real change"
    )


def test_a_paragraph_install_block_keeps_its_platform_headings(notes_module):
    """Platform headings go with the block whether it opened as heading or prose."""
    body = (
        "Intro.\n\n"
        "To update Unsloth, use the commands below:\n\n"
        "#### macOS, Linux, WSL:\n```\ncurl -fsSL https://unsloth.ai/install.sh | sh\n```\n\n"
        "#### Windows:\n```\nirm https://unsloth.ai/install.ps1 | iex\n```\n\n"
        "## Fixes\n\n1. a real fix\n"
    )
    stripped = notes_module.strip_release_body(body)
    assert stripped == "Intro.\n\n## Fixes\n\n1. a real fix"


def test_a_platform_heading_opens_an_install_block_on_its_own(notes_module):
    """`v0.1.0-beta` and `v0.1.41-beta` head their commands with a bare platform
    heading, so requiring an "Updating" above it left them in the popup. All
    seven platform headings in the 24 published bodies head an install block."""
    body = (
        "Intro.\n\n"
        "#### macOS, Linux, WSL:\n```\ncurl -fsSL https://unsloth.ai/install.sh | sh\n```\n\n"
        "#### Windows:\n```\nirm https://unsloth.ai/install.ps1 | iex\n```\n\n"
        "## Fixes\n\n1. a real fix\n"
    )
    stripped = notes_module.strip_release_body(body)
    assert stripped == "Intro.\n\n## Fixes\n\n1. a real fix"


def test_only_a_paragraph_of_its_own_opens_an_install_block(notes_module):
    """The same words inside the announcement are prose, not instructions:
    `v0.1.39-beta` says "call `curl ...` to update" mid-sentence, and generated
    entries are `* Update ...` lines. Neither opens a document-level paragraph."""
    continuation = (
        "Unsloth Studio 2026.5.2 is out.\n"
        "To update Unsloth, run the installer.\n\n"
        "## Fixes\n\n- a real fix\n"
    )
    stripped = notes_module.strip_release_body(continuation)
    assert "Unsloth Studio 2026.5.2 is out." in stripped
    assert "To update Unsloth, run the installer." in stripped
    assert "- a real fix" in stripped

    for line in ("* Update Unsloth icons by @someone", "> To update Unsloth, run it"):
        body = f"Intro.\n\n{line}\n\n## Fixes\n\n- a real fix\n"
        assert "a real fix" in notes_module.strip_release_body(body), line


def test_a_paragraph_install_block_does_not_swallow_deeper_headings(notes_module):
    """A heading section drops its subheadings, but a paragraph has no level, so
    the next heading resumes the announcement whatever its depth."""
    body = (
        "Intro.\n\n"
        "To update Unsloth, use the command below:\n"
        "```\ncurl -fsSL https://unsloth.ai/install.sh | sh\n```\n\n"
        "###### Deeply nested announcement\n\n- a real change\n"
    )
    stripped = notes_module.strip_release_body(body)
    assert stripped == "Intro.\n\n###### Deeply nested announcement\n\n- a real change"


@pytest.fixture(autouse=True)
def github_lockout(notes_module):
    from utils.prebuilt import freshness_flow

    freshness_flow._api_rate_limited_until = 0.0
    notes_module.reset_release_notes_cache()
    yield freshness_flow
    freshness_flow._api_rate_limited_until = 0.0
    notes_module.reset_release_notes_cache()


@pytest.mark.parametrize(
    ("url_override", "env", "expected"),
    [
        (None, {"GH_TOKEN": "ghp_gh", "GITHUB_TOKEN": "ghp_github"}, "Bearer ghp_github"),
        (None, {"GH_TOKEN": "ghp_gh"}, "Bearer ghp_gh"),
        ("https://mirror.example/releases", {"GH_TOKEN": "ghp_gh"}, None),
        ("http://api.github.com/repos/x/releases", {"GITHUB_TOKEN": "ghp_secret"}, None),
    ],
)
def test_where_the_release_notes_token_may_travel(
    notes_module, github_lockout, monkeypatch, url_override, env, expected
):
    import urllib.error

    seen = []

    def capture(request, timeout=None):
        seen.append(request)
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(notes_module.urllib.request, "urlopen", capture)
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    if url_override is None:
        monkeypatch.delenv(notes_module.RELEASES_URL_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(notes_module.RELEASES_URL_ENV_VAR, url_override)
    notes_module._fetch_latest_release()
    assert seen[0].get_header("Authorization") == expected
    assert "Authorization" not in seen[0].headers


@pytest.mark.parametrize(
    ("url_override", "status", "remaining", "body", "shared", "reported"),
    [
        (None, 403, "0", b"", True, True),
        (None, 429, "4998", b"", True, True),
        (None, 403, "4998", b'{"message": "secondary rate limit"}', True, True),
        (None, 403, "4998", b"", False, False),
        ("https://mirror.example/releases", 429, "0", b"", False, True),
    ],
)
def test_which_release_note_refusals_reach_the_shared_lockout(
    notes_module,
    github_lockout,
    monkeypatch,
    url_override,
    status,
    remaining,
    body,
    shared,
    reported,
):
    import email.message
    import io
    import urllib.error

    if url_override is None:
        monkeypatch.delenv(notes_module.RELEASES_URL_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(notes_module.RELEASES_URL_ENV_VAR, url_override)
    headers = email.message.Message()
    headers["X-RateLimit-Remaining"] = remaining
    headers["X-RateLimit-Reset"] = str(int(time.time() + 1800))

    def refuse(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, status, "refused", headers, io.BytesIO(body))

    monkeypatch.setattr(notes_module.urllib.request, "urlopen", refuse)
    result = notes_module.get_latest_release()
    assert (github_lockout.github_rate_limit_remaining() > 0) is shared
    assert ("rate limit" in (result.error or "").lower()) is reported


@pytest.mark.parametrize("url_override", [None, "https://mirror.example/releases"])
def test_the_shared_lockout_parks_only_the_github_api(
    notes_module, github_lockout, monkeypatch, url_override
):
    import urllib.error

    if url_override is None:
        monkeypatch.delenv(notes_module.RELEASES_URL_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(notes_module.RELEASES_URL_ENV_VAR, url_override)
    github_lockout.hold_github_api(60)
    calls = []

    def capture(request, timeout=None):
        calls.append(request.full_url)
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(notes_module.urllib.request, "urlopen", capture)
    notes_module.get_latest_release(refresh=True)
    assert calls == ([] if url_override is None else [url_override])
