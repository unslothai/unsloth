# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Main-content extraction and boilerplate stripping for the web fetch tool.

The HTML fixtures below snapshot the relevant fragments of a real GitHub repo
page (github.com/unslothai/unsloth, fetched 2026-07): the ``hidden``
client-side error placeholders ("Uh oh! There was an error while loading."),
the skip-link / nav / footer furniture, and the README rendered inside
``<article class="markdown-body">``. No network access is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference._html_to_md import _is_aria_heading, html_to_markdown
from core.inference.tools import (
    _fetch_page_text,
    _fetch_url_raw,
    _github_repo_readme_api_url,
    _looks_like_html,
)


# ── Fixtures: snapshot of GitHub repo page fragments ─────────────

# GitHub ships client-side error placeholders behind the `hidden` attribute (JS
# reveals them on a failed fetch); a text converter must not surface them.
_GITHUB_HIDDEN_ERROR_BLOCK = """
<div data-show-on-forbidden-error hidden>
  <div class="Box">
    <div class="blankslate-container">
      <h3 class="blankslate-heading">Uh oh!</h3>
      <p class="blankslate-description">
        <p class="color-fg-muted my-2 mb-2 ws-normal">There was an error while loading.
        <a class="Link--inTextBlock" href="" aria-label="Please reload this page">Please reload this page</a>.</p>
      </p>
    </div>
  </div>
</div>
"""

_GITHUB_PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head><title>unslothai/unsloth</title></head>
<body>
<a class="px-2 py-4" href="#start-of-content">Skip to content</a>
<header class="Header-old">
  <div class="AppHeader-globalBar">
    <a href="/login">Sign in</a>
    <a href="/signup">Sign up</a>
  </div>
</header>
<div class="js-notification-shelf"></div>
<div hidden>
  You signed in with another tab or window. Reload to refresh your session.
  You signed out in another tab or window. Reload to refresh your session.
  You switched accounts on another tab or window. Reload to refresh your session.
  Dismiss alert
</div>
<template>{{{{ message }}}}</template>
{_GITHUB_HIDDEN_ERROR_BLOCK}
<main id="js-repo-pjax-container">
  {_GITHUB_HIDDEN_ERROR_BLOCK}
  <div id="repository-container-header">
    <a href="/unslothai">unslothai</a> / <a href="/unslothai/unsloth">unsloth</a>
    <a href="/login?return_to=%2Funslothai%2Funsloth">Notifications</a>
    You must be signed in to change notification settings
  </div>
  <div class="repository-content">
    <table aria-labelledby="folders-and-files">
      <tr><th>Name</th><th>Last commit message</th></tr>
      <tr><td><a href="/unslothai/unsloth/tree/main/unsloth">unsloth</a></td><td></td></tr>
    </table>
    <article class="markdown-body entry-content container-lg" itemprop="text">
      <h1>Unsloth Studio</h1>
      <p>Unsloth Studio lets you run and train models locally. Fine-tune and
      run LLMs on Windows, Linux and macOS with a single install command,
      then export to GGUF, Ollama, vLLM or Hugging Face when you are done.</p>
      <h2>Install</h2>
      <pre>curl -fsSL https://unsloth.ai/install.sh | sh</pre>
      <p>See the <a href="https://unsloth.ai/docs">documentation</a> for
      quickstarts, notebooks, and fine-tuning guides for every major model
      family including Llama, Gemma, Qwen and DeepSeek.</p>
    </article>
  </div>
  <div class="Layout-sidebar">
    <h2>Languages</h2>
    <ul>
      <li><a href="/unslothai/unsloth/search?l=javascript">JavaScript 89.3%</a></li>
      <li><a href="/unslothai/unsloth/search?l=python">Python 9.7%</a></li>
    </ul>
  </div>
</main>
<footer>
  <a href="https://docs.github.com">Docs</a>
  <a href="https://github.com/contact">Contact</a>
</footer>
<div aria-live="polite" aria-hidden="true">You can't perform that action at this time.</div>
</body>
</html>
"""


# ── html_to_markdown: hidden elements ────────────────────────────


def test_hidden_attribute_subtree_is_dropped():
    html = "<body><p>visible</p><div hidden><p>secret error text</p></div><p>after</p></body>"
    out = html_to_markdown(html)
    assert "visible" in out
    assert "after" in out
    assert "secret error text" not in out


def test_aria_hidden_true_subtree_is_dropped():
    html = '<body><p>keep</p><span aria-hidden="true">decoration</span></body>'
    out = html_to_markdown(html)
    assert "keep" in out
    assert "decoration" not in out


def test_aria_hidden_false_subtree_is_kept():
    html = '<body><span aria-hidden="false">still here</span></body>'
    assert "still here" in html_to_markdown(html)


def test_inline_style_display_none_subtree_is_dropped():
    # Error/loading blocks are often hidden with inline CSS rather than the
    # ``hidden`` attribute; browsers do not render them, so they must not leak.
    html = (
        "<body><p>visible</p>"
        '<div style="display:none">secret loading block</div>'
        "<p>after</p></body>"
    )
    out = html_to_markdown(html)
    assert "visible" in out
    assert "after" in out
    assert "secret loading block" not in out


def test_inline_style_visibility_hidden_subtree_is_dropped():
    html = '<body><p>keep</p><span style="visibility:hidden">ghost</span></body>'
    out = html_to_markdown(html)
    assert "keep" in out
    assert "ghost" not in out


def test_inline_style_display_none_important_is_dropped():
    # The !important flag must not defeat the display:none detection.
    html = '<body><p>keep</p><div style="display:none !important">gone</div></body>'
    out = html_to_markdown(html)
    assert "keep" in out
    assert "gone" not in out


def test_inline_style_display_none_among_other_declarations():
    html = '<body><p>keep</p><div style="color: red; display : none ; margin:0">gone</div></body>'
    out = html_to_markdown(html)
    assert "keep" in out
    assert "gone" not in out


def test_inline_style_visible_display_is_kept():
    # Over-strip guard: display:block / visibility:visible render, and a value or
    # URL merely containing the substring "none" must not trigger the hidden path.
    html = (
        "<body>"
        '<div style="display:block">block kept</div>'
        '<div style="visibility:visible">visible kept</div>'
        '<a style="background:url(none.png)">link kept</a>'
        "</body>"
    )
    out = html_to_markdown(html)
    assert "block kept" in out
    assert "visible kept" in out
    assert "link kept" in out


def test_hidden_recovers_from_omitted_close_tags():
    # <p hidden> is never closed; the parent </div> must still end the hidden region.
    html = "<body><div><p hidden>gone</div><p>kept</p></body>"
    out = html_to_markdown(html)
    assert "gone" not in out
    assert "kept" in out


def test_nested_hidden_regions():
    html = "<body><div hidden><div hidden>inner</div>outer</div><p>ok</p></body>"
    out = html_to_markdown(html)
    assert "inner" not in out
    assert "outer" not in out
    assert "ok" in out


def test_hidden_false_is_still_hidden():
    # ``hidden`` is enumerated: the spec maps invalid/empty values to the Hidden
    # state, so hidden="false" is NOT rendered and must not reach the Markdown.
    html = '<body><p>keep</p><div hidden="false">not rendered</div></body>'
    out = html_to_markdown(html)
    assert "keep" in out
    assert "not rendered" not in out


def test_hidden_paragraph_omitted_close_does_not_swallow_siblings():
    # HTML5 optional end tags: a sibling <p> start tag implicitly closes an open
    # <p hidden>, so the hidden region ends there instead of swallowing siblings.
    html = (
        "<body><div><p hidden>secret<p>visible one</p><p>visible two</p></div><p>after</p></body>"
    )
    out = html_to_markdown(html)
    assert "secret" not in out
    assert "visible one" in out
    assert "visible two" in out
    assert "after" in out


def test_hidden_list_item_omitted_close_keeps_following_items():
    # <li hidden> without </li> is implicitly closed by the next <li>.
    html = "<body><ul><li hidden>secret<li>shown A</li><li>shown B</li></ul></body>"
    out = html_to_markdown(html)
    assert "secret" not in out
    assert "shown A" in out
    assert "shown B" in out


def test_hr_implicitly_closes_hidden_paragraph():
    # Void elements also imply closes: <hr> ends an open <p hidden>.
    html = "<body><p hidden>secret<hr>kept text</body>"
    out = html_to_markdown(html)
    assert "secret" not in out
    assert "kept text" in out


def test_skipped_tag_implicitly_closes_hidden_paragraph():
    # A skipped block (<nav>/<footer>) also closes an open <p>. The optional-close
    # bookkeeping must run before the skip, or the never-closed <p hidden> keeps its
    # hidden mark and swallows every following sibling.
    for skipped in ("nav", "footer"):
        html = f"<body><p hidden>secret<{skipped}>chrome</{skipped}>VISIBLE</body>"
        out = html_to_markdown(html)
        assert "secret" not in out
        assert "chrome" not in out
        assert "VISIBLE" in out


def test_hidden_void_element_is_suppressed():
    # A hidden void element (<hr>/<br>) never joins the open-element stack, so it
    # must be suppressed inline rather than emitting its markup.
    html = '<body><p>before</p><hr aria-hidden="true"><p>after</p></body>'
    out = html_to_markdown(html)
    assert "before" in out
    assert "after" in out
    assert "---" not in out


def test_hidden_void_br_emits_no_break():
    html = "<body><p>one<br hidden>two</p></body>"
    out = html_to_markdown(html)
    assert "one" in out
    assert "two" in out
    # The hidden <br> must not inject a newline between the two runs.
    assert "one\ntwo" not in out


def test_visible_void_hr_still_renders():
    # Guard: the suppression must not affect non-hidden void elements.
    html = "<body><p>a</p><hr><p>b</p></body>"
    out = html_to_markdown(html)
    assert "---" in out


# ── html_to_markdown: main-content scoping ───────────────────────


def test_github_page_main_content_keeps_readme_only():
    out = html_to_markdown(_GITHUB_PAGE, main_content = True)
    # README content survives.
    assert "Unsloth Studio" in out
    assert "install.sh" in out
    assert "documentation" in out
    # Client-side error placeholders and page furniture are gone.
    assert "Uh oh!" not in out
    assert "There was an error while loading" not in out
    assert "Please reload this page" not in out
    assert "You can't perform that action at this time" not in out
    assert "Skip to content" not in out
    assert "Sign in" not in out
    assert "Reload to refresh your session" not in out
    assert "JavaScript 89.3%" not in out
    assert "Languages" not in out
    assert "Last commit message" not in out


def test_main_scope_used_when_no_article():
    html = """
    <body>
      <header><a href="/login">Sign in</a></header>
      <main><h1>Doc title</h1><p>%s</p></main>
      <footer>footer junk</footer>
    </body>
    """ % ("Body text. " * 40)
    out = html_to_markdown(html, main_content = True)
    assert "Doc title" in out
    assert "Body text." in out
    assert "Sign in" not in out
    assert "footer junk" not in out


def test_main_content_falls_back_to_full_document():
    # No article/main and a tiny body: the unscoped conversion is returned.
    html = "<body><h1>Tiny</h1><p>Just a short page.</p></body>"
    out = html_to_markdown(html, main_content = True)
    assert "Tiny" in out
    assert "Just a short page." in out


def test_tiny_article_stub_does_not_hijack_scope():
    # An <article> with negligible text must not swallow the real content.
    body_text = "Real content paragraph. " * 30
    html = f"<body><article>ad</article><main><p>{body_text}</p></main></body>"
    out = html_to_markdown(html, main_content = True)
    assert "Real content paragraph." in out


def test_sibling_articles_do_not_leak_after_main_selected():
    # The size gate picks the largest single <article> and renders only that
    # subtree: sibling articles (related-post cards, comment threads) must not leak
    # in just because the real article cleared the threshold.
    real = "Main article body content for selection. " * 20
    card = "Unrelated related-post card teaser blurb. " * 3
    cards = "".join(f"<article><p>{card}</p></article>" for _ in range(5))
    html = f"<body><article><h1>Real</h1><p>{real}</p></article>{cards}</body>"
    out = html_to_markdown(html, main_content = True)
    assert "Main article body content" in out
    assert "Unrelated related-post" not in out


def test_default_conversion_unscoped_and_unstripped():
    # Without main_content the whole document converts (backwards compatible),
    # boilerplate included; only hidden subtrees are dropped.
    html = "<body><p>Skip to content</p><div hidden>gone</div><main><p>hello</p></main></body>"
    out = html_to_markdown(html)
    assert "Skip to content" in out
    assert "hello" in out
    assert "gone" not in out


def test_boilerplate_filter_preserves_phrase_inside_real_prose():
    # The furniture filter once matched by substring, deleting a real sentence that
    # merely CONTAINS a fragment ("we use cookies"). It must drop only lines COMPOSED
    # of furniture, keeping real prose that quotes one.
    body = (
        "<article><h1>Authentication</h1>"
        "<p>We use cookies to authenticate API requests and keep sessions safe.</p>"
        "<p>%s</p></article>"
    ) % ("Additional documentation content to select the article. " * 8)
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "We use cookies to authenticate API requests" in out


def test_boilerplate_filter_still_drops_standalone_and_stacked_furniture():
    # A line that is purely furniture is dropped, as is one stacking several
    # furniture phrases (as GitHub renders them).
    body = (
        "<article>"
        "<p>Skip to content</p>"
        "<p>You signed in with another tab or window. Reload to refresh your session.</p>"
        "<p>Real README body. %s</p>"
        "</article>"
    ) % ("Genuine documentation text. " * 8)
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Real README body." in out
    assert "Skip to content" not in out
    assert "Reload to refresh your session" not in out


def test_boilerplate_not_stripped_inside_code_fences():
    html = (
        "<body><article><p>%s</p>"
        "<pre>assert 'There was an error while loading' in page</pre>"
        "</article></body>" % ("Prose. " * 40)
    )
    out = html_to_markdown(html, main_content = True)
    assert "There was an error while loading" in out


def test_aside_callout_inside_article_is_kept():
    # Docs render notes/warnings as <aside> callouts. An aside inside the selected
    # article/main scope is real content and must survive; dropping it unconditionally
    # loses page text.
    body = (
        "<article><h1>Guide</h1>"
        "<p>%s</p>"
        "<aside class='admonition warning'><strong>Warning:</strong> "
        "This operation is destructive and cannot be undone.</aside>"
        "<p>Trailing paragraph.</p></article>"
    ) % ("Documentation body text to select the article scope. " * 6)
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "This operation is destructive and cannot be undone." in out
    assert "Warning:" in out
    # Also kept in the unscoped (backwards-compatible) conversion.
    out_full = html_to_markdown(f"<body>{body}</body>")
    assert "This operation is destructive and cannot be undone." in out_full


# ── GitHub README rewrite ────────────────────────────────────────


def test_github_repo_url_maps_to_readme_api():
    assert (
        _github_repo_readme_api_url("https://github.com/unslothai/unsloth")
        == "https://api.github.com/repos/unslothai/unsloth/readme"
    )
    assert (
        _github_repo_readme_api_url("https://github.com/unslothai/unsloth/")
        == "https://api.github.com/repos/unslothai/unsloth/readme"
    )
    assert (
        _github_repo_readme_api_url("http://www.github.com/owner/repo.git")
        == "https://api.github.com/repos/owner/repo/readme"
    )


def test_github_non_repo_urls_are_not_rewritten():
    for url in (
        "https://github.com/unslothai/unsloth/tree/main/studio",
        "https://github.com/unslothai/unsloth/issues/123",
        "https://github.com/topics/llm",
        "https://github.com/orgs/unslothai/repositories",
        "https://github.com/login/oauth",
        "https://github.com/unslothai",
        "https://example.com/owner/repo",
        "https://raw.githubusercontent.com/owner/repo/main/README.md",
    ):
        assert _github_repo_readme_api_url(url) is None, url


def test_fetch_page_text_prefers_github_readme(monkeypatch):
    calls = []

    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        calls.append((url, extra_headers))
        assert url == "https://api.github.com/repos/unslothai/unsloth/readme"
        return None, "# Unsloth\n\nFine-tune LLMs faster.", "text/plain"

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    out = _fetch_page_text("https://github.com/unslothai/unsloth")
    assert "Fine-tune LLMs faster." in out
    assert "README of https://github.com/unslothai/unsloth" in out
    assert len(calls) == 1
    assert calls[0][1]["Accept"] == "application/vnd.github.raw+json"


def test_fetch_page_text_keeps_html_readme_from_api(monkeypatch):
    # A repo whose README is HTML returns HTML from the README API with a 200. That
    # success is authoritative: convert to Markdown and keep it, never discard it in
    # favour of the repo root page's UI chrome.
    html_readme = (
        "<!doctype html><html><body>"
        "<h1>Project Title</h1>"
        "<p>Install with the one-line script and read the docs.</p>"
        "</body></html>"
    )
    calls = []

    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        calls.append(url)
        assert url == "https://api.github.com/repos/unslothai/unsloth/readme"
        return None, html_readme, "text/html"

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    out = _fetch_page_text("https://github.com/unslothai/unsloth")
    # The successful README is converted and returned; no fallback fetch fires.
    assert "README of https://github.com/unslothai/unsloth" in out
    assert "Project Title" in out
    assert "Install with the one-line script" in out
    assert "<html" not in out
    assert len(calls) == 1


def test_fetch_page_text_falls_back_to_html_when_readme_api_fails(monkeypatch):
    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        if url.startswith("https://api.github.com/"):
            return "Failed to fetch URL: HTTP 403 rate limited", "", ""
        return None, _GITHUB_PAGE, "text/html"

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    out = _fetch_page_text("https://github.com/unslothai/unsloth")
    # Fallback converts the HTML page with the main-content heuristic.
    assert "Unsloth Studio" in out
    assert "Uh oh!" not in out
    assert "There was an error while loading" not in out


def test_fetch_page_text_non_html_returned_raw(monkeypatch):
    raw = "line one\n    indented code\nline three"

    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        return None, raw, "text/plain"

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    out = _fetch_page_text("https://raw.githubusercontent.com/o/r/main/file.txt")
    # Whitespace preserved: the HTML renderer would have collapsed it.
    assert "    indented code" in out


def test_fetch_page_text_html_conversion(monkeypatch):
    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        return None, _GITHUB_PAGE, "text/html"

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    out = _fetch_page_text("https://github.com/unslothai/unsloth/tree/main")
    assert "Unsloth Studio" in out
    assert "Uh oh!" not in out


def test_fetch_page_text_propagates_fetch_errors(monkeypatch):
    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        return "Failed to fetch URL: HTTP 404 Not Found", "", ""

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    assert _fetch_page_text("https://example.com/missing") == (
        "Failed to fetch URL: HTTP 404 Not Found"
    )


def test_looks_like_html():
    assert _looks_like_html("<!DOCTYPE html><html></html>")
    assert _looks_like_html("\n  <HTML lang='en'>")
    assert not _looks_like_html("# Markdown README\n\n<h1>embedded html later</h1>")
    assert not _looks_like_html("plain text")


def test_looks_like_html_markdown_with_leading_fenced_example_stays_markdown():
    # A Markdown README OPENING with a fenced HTML example must not be sniffed as
    # HTML just because a doctype/tag appears in the first 256 chars; html_to_markdown
    # would corrupt the fences and prose.
    fenced = (
        "```html\n<!DOCTYPE html>\n<html><body><div>hi</div></body></html>\n```\n\n# Real README\n"
    )
    assert not _looks_like_html(fenced)
    # Prose that mentions a tag inline, and a centered-logo README that opens
    # with <p align>/<div align>/<h1 align>, also stay Markdown.
    assert not _looks_like_html("Use the <html> element to start a page.")
    assert not _looks_like_html('<p align="center"><img src="logo.png"></p>\n\n# Project\n')
    assert not _looks_like_html('<div align="center">\n\n# Project\n\n</div>\n')
    assert not _looks_like_html('<h1 align="center">Project</h1>\n\nMarkdown body.\n')
    # An autolink is not a tag opener.
    assert not _looks_like_html("<https://example.com> is the homepage")


def test_looks_like_html_detects_bare_fragments():
    # A body that is a bare HTML fragment (no <html>/doctype) must still be
    # recognized so it is converted to Markdown.
    assert _looks_like_html("<body><p>hello</p></body>")
    assert _looks_like_html("\n<article><h1>Title</h1><p>Body</p></article>")
    assert _looks_like_html("<section>content</section>")


def test_looks_like_html_leading_table_stays_markdown():
    # Markdown READMEs routinely open with a raw HTML <table> badge row or logo
    # layout, then continue in Markdown. Sniffing that as HTML would collapse the
    # Markdown body, so a leading <table> (and its row/cell children) must stay
    # Markdown, like the excluded <div align>/<p align> layout headers.
    assert not _looks_like_html("<table><tr><td>cell</td></tr></table>")
    assert not _looks_like_html(
        '<table align="center"><tr><td><img src="logo.png"></td></tr></table>\n\n# Project\n'
    )
    assert not _looks_like_html("<tr><td>cell</td></tr>")


def test_fetch_page_text_keeps_markdown_readme_with_html_example(monkeypatch):
    # A Markdown README opening with a fenced HTML snippet must be served verbatim,
    # never run through html_to_markdown (which would drop the fences/tags).
    md_readme = (
        "```html\n"
        "<!DOCTYPE html>\n"
        "<html><body><h1>Demo</h1></body></html>\n"
        "```\n\n"
        "# My Project\n\nInstall and run.\n"
    )

    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        assert url == "https://api.github.com/repos/unslothai/unsloth/readme"
        return None, md_readme, "text/plain"

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    out = _fetch_page_text("https://github.com/unslothai/unsloth")
    assert "README of https://github.com/unslothai/unsloth" in out
    # Markdown preserved verbatim: the fence and literal tags survive.
    assert "```html" in out
    assert "<!DOCTYPE html>" in out
    assert "# My Project" in out


def test_fetch_page_text_keeps_markdown_readme_with_leading_table(monkeypatch):
    # A README opening with a raw HTML <table> badge/layout row then continuing in
    # Markdown must be served verbatim, never run through html_to_markdown (which
    # would collapse the list/fence/heading body onto one line).
    md_readme = (
        '<table align="center">\n'
        '<tr><td><img src="logo.png"></td><td>Badges</td></tr>\n'
        "</table>\n\n"
        "# My Project\n\n"
        "- feature one\n"
        "- feature two\n\n"
        "```python\nprint('hi')\n```\n"
    )

    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        assert url == "https://api.github.com/repos/unslothai/unsloth/readme"
        return None, md_readme, "text/plain"

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    out = _fetch_page_text("https://github.com/unslothai/unsloth")
    assert "README of https://github.com/unslothai/unsloth" in out
    # Markdown body verbatim: list, fence and heading survive on their own lines.
    assert "- feature one\n- feature two" in out
    assert "```python" in out
    assert "# My Project" in out


def test_fetch_url_raw_missing_content_type_reported_empty(monkeypatch):
    # Message.get_content_type() falls back to the RFC 2045 "text/plain" default
    # when the header is absent; _fetch_url_raw must report "" instead so the HTML
    # sniffing fallback can fire.
    import email
    import urllib.request

    class _FakeResp:
        headers = email.message_from_string("")

        def __init__(self):
            self._body = b"<html><body>hello</body></html>"

        def read(self, n = -1):
            # Hand back the body once, then EOF, so the chunked reader terminates.
            body, self._body = self._body, b""
            return body

    class _FakeOpener:
        def open(
            self,
            req,
            timeout = None,
        ):
            return _FakeResp()

    monkeypatch.setattr(
        "core.inference.tools._validate_and_resolve_host",
        lambda host, port: (True, "", "203.0.113.7"),
    )
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: _FakeOpener())
    err, body, content_type = _fetch_url_raw("https://example.com/")
    assert err is None
    assert "hello" in body
    assert content_type == ""


@pytest.mark.parametrize(
    "disable_dns_pinning,expected_url",
    [
        (False, "https://203.0.113.7:8443/page?q=1"),
        (True, "https://example.com:8443/page?q=1"),
    ],
)
def test_fetch_url_raw_dns_pinning_proxy_opt_out(monkeypatch, disable_dns_pinning, expected_url):
    import email
    import urllib.request

    import core.inference.tools as tools_mod

    class _FakeResp:
        headers = email.message_from_string("Content-Type: text/plain\n")

        def __init__(self):
            self._body = b"ok"

        def read(self, n = -1):
            body, self._body = self._body, b""
            return body

    requested = []

    class _FakeOpener:
        def open(
            self,
            req,
            timeout = None,
        ):
            requested.append(req)
            return _FakeResp()

    resolved = []

    def resolve(host, port):
        resolved.append((host, port))
        return True, "", "203.0.113.7"

    monkeypatch.setenv("UNSLOTH_STUDIO_DISABLE_DNS_PINNING", "1" if disable_dns_pinning else "0")
    monkeypatch.setattr(tools_mod, "_validate_and_resolve_host", resolve)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: _FakeOpener())

    # No embedded credentials: the web access policy rejects those outright
    # (see test_fetch_url_raw_rejects_embedded_credentials).
    err, body, _content_type = tools_mod._fetch_url_raw("https://example.com:8443/page?q=1")

    assert err is None
    assert body == "ok"
    assert resolved == [("example.com", 8443)]
    assert [req.full_url for req in requested] == [expected_url]
    assert requested[0].get_header("Host") == "example.com:8443"


def test_fetch_url_raw_rejects_embedded_credentials(monkeypatch):
    # Credentials in the URL are blocked rather than stripped, so they can never
    # leak to a redirect target or into logs.
    import core.inference.tools as tools_mod

    def resolve(host, port):
        raise AssertionError("must be rejected before DNS resolution")

    monkeypatch.setattr(tools_mod, "_validate_and_resolve_host", resolve)

    err, body, _content_type = tools_mod._fetch_url_raw(
        "https://user:secret@example.com:8443/page?q=1"
    )

    assert err is not None and "credentials" in err
    assert body == ""


def test_fetch_page_text_missing_content_type_html_sniffed(monkeypatch):
    # A header-less server returning an HTML body must still be converted.
    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        return None, _GITHUB_PAGE, ""

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    out = _fetch_page_text("https://example.com/no-content-type")
    assert "Unsloth Studio" in out
    assert "<html" not in out
    assert "Uh oh!" not in out


def test_fetch_page_text_missing_content_type_fragment_converted(monkeypatch):
    # A header-less server returning a bare HTML fragment (no <html>/doctype) must
    # still be sniffed as HTML and converted, not served as raw markup.
    fragment = "<article><h1>Doc Title</h1><p>Readable fragment body.</p></article>"

    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        return None, fragment, ""

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    out = _fetch_page_text("https://example.com/fragment")
    assert "Doc Title" in out
    assert "Readable fragment body." in out
    assert "<article" not in out


def test_fetch_page_text_missing_content_type_plain_text_raw(monkeypatch):
    # A header-less server returning plain text stays raw (whitespace kept).
    raw = "line one\n    indented code\nline three"

    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        return None, raw, ""

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    out = _fetch_page_text("https://example.com/no-content-type.txt")
    assert "    indented code" in out


def test_fetch_page_text_mislabeled_text_plain_html_converted(monkeypatch):
    # An explicit text/plain header on an HTML body is sniffed and converted, like
    # the pre-extraction behavior of always converting HTML pages.
    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        return None, _GITHUB_PAGE, "text/plain"

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    out = _fetch_page_text("https://example.com/mislabeled")
    assert "Unsloth Studio" in out
    assert "<html" not in out


# ── implicit-close past unclosed inline descendants (finding 14) ──


def test_hidden_paragraph_with_inline_child_implicitly_closed_by_block():
    # A browser closes an open <p> when a <div> arrives, even with an unclosed
    # <span> on top of it. The hidden region must end there, not swallow the
    # following visible blocks.
    html = "<body><p hidden><span>secret<div>visible div</div><p>visible paragraph</body>"
    out = html_to_markdown(html)
    assert "secret" not in out
    assert "visible div" in out
    assert "visible paragraph" in out


def test_hidden_list_item_with_inline_child_closed_by_next_item():
    html = "<body><ul><li hidden><span>secret<li>visible item</ul><p>after</p></body>"
    out = html_to_markdown(html)
    assert "secret" not in out
    assert "visible item" in out
    assert "after" in out


# ── nested hidden list/table contents must stay suppressed ──


def test_nested_hidden_list_does_not_leak_child_items():
    # The nested <ul> re-scopes the item, so the inner <li> is a DESCENDANT of the
    # hidden outer <li>, not an optional-close sibling. Optional-end-tag recovery
    # must not cross the intervening <ul>, or the outer li's hidden mark is popped
    # and the nested text leaks.
    html = (
        "<body><ul>"
        "<li hidden>parent<ul><li>secret child</li></ul></li>"
        "<li>visible sibling</li>"
        "</ul></body>"
    )
    out = html_to_markdown(html)
    assert "parent" not in out
    assert "secret child" not in out
    assert "visible sibling" in out


def test_nested_hidden_list_with_omitted_closes_stays_suppressed():
    # Same leak, doubly nested with omitted </li>/</ul>. Every hidden descendant
    # stays gone; the following visible sibling (which implicitly closes the hidden
    # outer <li>) still renders.
    html = (
        "<body><ul>"
        "<li hidden>parent<ul><li>secret child<ul><li>deeper secret</ul></li></ul>"
        "<li>visible sibling"
        "</ul></body>"
    )
    out = html_to_markdown(html)
    assert "parent" not in out
    assert "secret child" not in out
    assert "deeper secret" not in out
    assert "visible sibling" in out


def test_nested_hidden_table_does_not_leak_inner_cells():
    # A nested <table> re-scopes <tr>/<td>: an inner <td> must not be an
    # optional-close sibling of a hidden outer <td> across the nested table.
    html = (
        "<body><table><tr>"
        "<td hidden>outer<table><tr><td>secret cell</td></tr></table></td>"
        "<td>visible cell</td>"
        "</tr></table></body>"
    )
    out = html_to_markdown(html)
    assert "secret cell" not in out
    assert "visible cell" in out


# ── aggregate tiny <article> cards must not displace <main> (finding 15) ──


def test_many_tiny_articles_do_not_displace_substantial_main():
    cards = "".join(
        f"<article><h2>Teaser {i}</h2><p>Advertisement card blurb.</p></article>" for i in range(12)
    )
    main_body = "Authoritative main documentation content. " * 30
    html = f"<body>{cards}<main><h1>Real page</h1><p>{main_body}</p></main></body>"
    out = html_to_markdown(html, main_content = True)
    assert "Authoritative main documentation content." in out
    assert "Advertisement card blurb." not in out


def test_single_substantial_article_still_preferred_over_main():
    # GitHub-README case: one substantial <article> inside <main> must still win
    # over sibling <main> furniture.
    article_body = "Real README documentation body text. " * 20
    html = (
        "<body><main>"
        f"<article><h1>Guide</h1><p>{article_body}</p></article>"
        "<div><h2>Languages</h2><p>JavaScript 89.3%</p></div>"
        "</main></body>"
    )
    out = html_to_markdown(html, main_content = True)
    assert "Real README documentation body text." in out
    assert "JavaScript 89.3%" not in out


# ── truncated (unclosed) main-content scopes must still be scored ──


def test_truncated_open_article_scope_is_scored_and_preferred():
    # _fetch_url_raw caps large pages, so the download can end before the closing
    # </article>. The scope is still the main content and must be preferred over the
    # whole document (which re-leaks the page chrome).
    chrome = "<nav>Skip to content</nav><div>Repository file tree and page chrome.</div>"
    article_body = "Real README documentation body text. " * 20
    # No closing </article> / </body> -- the fetch cap truncated the page.
    html = f"<body>{chrome}<article><h1>Guide</h1><p>{article_body}</p>"
    out = html_to_markdown(html, main_content = True)
    assert "Real README documentation body text." in out
    assert "Repository file tree and page chrome." not in out


def test_truncated_open_main_scope_is_scored_and_preferred():
    chrome = "<nav>Skip to content</nav><div>Repository file tree and page chrome.</div>"
    main_body = "Authoritative main documentation content. " * 30
    html = f"<body>{chrome}<main><h1>Doc</h1><p>{main_body}</p>"
    out = html_to_markdown(html, main_content = True)
    assert "Authoritative main documentation content." in out
    assert "Repository file tree and page chrome." not in out


# ── overall fetch deadline + cancellation (no per-hop timeout blowup) ──


def test_fetch_url_raw_overall_deadline_aborts_across_redirects(monkeypatch):
    # Each hop advances a fake clock by 5s; an 8s overall budget is exhausted on the
    # third hop even though every hop stays within its own socket timeout. Without
    # the deadline this would redirect until the 5-hop cap, so the "timed out" error
    # proves the overall budget aborted it, not the hop cap.
    import urllib.request
    from urllib.error import HTTPError

    import core.inference.tools as tools_mod

    clock = {"t": 1000.0}
    monkeypatch.setattr(tools_mod.time, "monotonic", lambda: clock["t"])

    hops = {"n": 0}

    class _RedirectingOpener:
        def open(
            self,
            req,
            timeout = None,
        ):
            clock["t"] += 5.0
            hops["n"] += 1
            raise HTTPError(
                req.full_url,
                302,
                "Found",
                {"Location": "https://example.com/next"},
                None,
            )

    monkeypatch.setattr(
        tools_mod,
        "_validate_and_resolve_host",
        lambda host, port: (True, "", "203.0.113.7"),
    )
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: _RedirectingOpener())

    err, body, content_type = tools_mod._fetch_url_raw(
        "https://example.com/start",
        timeout = 30,
        deadline = clock["t"] + 8.0,
    )
    assert err == "Failed to fetch URL: timed out."
    assert body == ""
    assert hops["n"] < 5


def test_fetch_url_raw_cancel_event_aborts_before_network(monkeypatch):
    # A set cancel_event (client disconnected) stops the fetch before it opens any
    # socket, so a dropped stream cannot leave a tool blocking on the wire.
    import threading
    import urllib.request

    import core.inference.tools as tools_mod

    ev = threading.Event()
    ev.set()
    opened = {"n": 0}

    class _Opener:
        def open(
            self,
            req,
            timeout = None,
        ):
            opened["n"] += 1
            raise AssertionError("network must not be touched after cancel")

    monkeypatch.setattr(
        tools_mod,
        "_validate_and_resolve_host",
        lambda host, port: (True, "", "203.0.113.7"),
    )
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: _Opener())

    err, body, content_type = tools_mod._fetch_url_raw(
        "https://example.com/",
        cancel_event = ev,
    )
    assert err == "Failed to fetch URL: cancelled."
    assert opened["n"] == 0


def test_fetch_page_text_shares_one_deadline_across_readme_and_fallback(monkeypatch):
    # The README API attempt and its HTML fallback must draw from ONE budget: a
    # failed API call cannot hand the fallback a fresh full timeout.
    seen_deadlines = []

    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        seen_deadlines.append(deadline)
        # Fail the README API so the HTML fallback also runs.
        return "Failed to fetch URL: HTTP 429 rate limited", "", ""

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    out = _fetch_page_text("https://github.com/unslothai/unsloth", timeout = 30)
    assert out == "Failed to fetch URL: HTTP 429 rate limited"
    # Both attempts ran and shared the same, single deadline value.
    assert len(seen_deadlines) == 2
    assert seen_deadlines[0] is not None
    assert seen_deadlines[0] == seen_deadlines[1]


# -- overall deadline reaches the body read, the resolver, and the query path --


def test_fetch_url_raw_deadline_aborts_slow_body(monkeypatch):
    # A server dribbling the body must not stretch the read past the overall
    # deadline: the body is read in chunks with the budget re-checked between them,
    # so a single slow resp.read cannot outlast the fetch budget.
    import email
    import urllib.request

    import core.inference.tools as tools_mod

    clock = {"t": 1000.0}
    monkeypatch.setattr(tools_mod.time, "monotonic", lambda: clock["t"])

    class _DrippingResp:
        headers = email.message_from_string("")

        def read(self, n = -1):
            # One chunk, then jump the clock past the deadline so the next
            # between-chunk budget check aborts instead of reading forever.
            clock["t"] += 10.0
            return b"x" * 16

        def close(self):
            pass

    class _Opener:
        def open(
            self,
            req,
            timeout = None,
        ):
            return _DrippingResp()

    monkeypatch.setattr(
        tools_mod,
        "_validate_and_resolve_host",
        lambda host, port: (True, "", "203.0.113.7"),
    )
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: _Opener())

    err, body, content_type = tools_mod._fetch_url_raw(
        "https://example.com/",
        timeout = 30,
        deadline = clock["t"] + 5.0,
    )
    assert err == "Failed to fetch URL: timed out."
    assert body == ""


def test_resolve_with_budget_aborts_on_slow_resolver(monkeypatch):
    # getaddrinfo has no deadline of its own; a resolver slower than the budget must
    # abort on time instead of blocking the whole fetch.
    import threading

    import core.inference.tools as tools_mod

    clock = {"t": 1000.0}
    monkeypatch.setattr(tools_mod.time, "monotonic", lambda: clock["t"])

    release = threading.Event()

    def slow_resolve(host, port):
        release.wait(5.0)  # block until released; the budget should abort first
        return True, "", "203.0.113.7"

    monkeypatch.setattr(tools_mod, "_validate_and_resolve_host", slow_resolve)

    def advance_past_deadline():
        import time as _t
        _t.sleep(0.1)
        clock["t"] += 100.0

    t = threading.Thread(target = advance_past_deadline, daemon = True)
    t.start()
    try:
        ok, reason, ip = tools_mod._resolve_with_budget(
            "example.com",
            443,
            1005.0,
            None,
        )
    finally:
        release.set()
    assert ok is False
    assert reason == "Failed to fetch URL: timed out."


def test_web_search_query_cancelled_skips_search(monkeypatch):
    # A pre-set cancel_event (client disconnected) skips the blocking DDGS query,
    # matching the direct-URL path's cancellation.
    import sys
    import threading
    import types

    import core.inference.tools as tools_mod

    ev = threading.Event()
    ev.set()
    called = {"n": 0}

    class _DDGS:
        def __init__(self, *a, **k):
            called["n"] += 1

        def text(self, *a, **k):
            called["n"] += 1
            return []

    fake_mod = types.ModuleType("ddgs")
    fake_mod.DDGS = _DDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_mod)

    out = tools_mod._web_search("some query", cancel_event = ev)
    assert out == "Search cancelled."
    assert called["n"] == 0


def test_fetch_page_text_markdown_readme_with_leading_block_tag_stays_markdown(monkeypatch):
    # A raw-Markdown README that OPENS with an HTML block tag (<blockquote>, <ul>,
    # <pre>, ...) must not be run through html_to_markdown, which would collapse its
    # headings/list/fence. Only a real HTML document (doctype / <html>) is converted.
    md_readme = (
        "<blockquote>Note: pre-release.</blockquote>\n\n"
        "# My Project\n\n"
        "Install:\n\n"
        "- step one\n"
        "- step two\n\n"
        "```bash\npip install myproject\n```\n"
    )

    def fake_fetch(
        url,
        timeout = 30,
        extra_headers = None,
        deadline = None,
        cancel_event = None,
    ):
        assert url == "https://api.github.com/repos/unslothai/unsloth/readme"
        return None, md_readme, "text/plain"

    monkeypatch.setattr("core.inference.tools._fetch_url_raw", fake_fetch)
    out = _fetch_page_text("https://github.com/unslothai/unsloth")
    assert "README of https://github.com/unslothai/unsloth" in out
    # Markdown structure survives verbatim (heading, list, fenced code).
    assert "# My Project" in out
    assert "- step one" in out
    assert "```bash" in out


def test_looks_like_html_document_only_matches_real_documents():
    from core.inference.tools import _looks_like_html_document

    assert _looks_like_html_document("<!doctype html><html><body>x</body></html>")
    assert _looks_like_html_document("\n  <HTML lang='en'>")
    assert _looks_like_html_document("<body><h1>x</h1></body>")
    # Block tags a Markdown README can open with are NOT full documents.
    for frag in (
        "<blockquote>q</blockquote>",
        "<ul><li>x</li></ul>",
        "<pre>x</pre>",
        "<dl><dt>x</dt></dl>",
    ):
        assert not _looks_like_html_document(frag), frag


# <header> inside the selected scope (Wikipedia's Vector 2022 skin)
def _interlanguage_list(count: int) -> str:
    return "".join(
        f'<li class="interlanguage-link">'
        f'<a href="https://x{i}.wikipedia.org/wiki/K">Lang{i}</a></li>'
        for i in range(count)
    )


def test_in_main_header_language_list_does_not_displace_article():
    body = (
        "<main><header><h1>Cat</h1><div id='p-lang-btn'><ul>%s</ul></div></header>"
        "<div id='mw-content-text'><p>%s</p></div></main>"
    ) % (_interlanguage_list(300), "The cat (Felis catus) is a small mammal. " * 30)
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Felis catus" in out
    assert "Lang0" not in out
    assert "x0.wikipedia.org" not in out
    assert "# Cat" in out
    assert out.index("Felis catus") < 200


def test_header_title_kept_when_article_is_shorter_than_its_language_list():
    body = ("<main><header><h1>Stub</h1><ul>%s</ul></header><p>%s</p></main>") % (
        _interlanguage_list(300),
        "Short article body. " * 15,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Short article body." in out
    assert "Lang0" not in out
    assert "# Stub" in out


def test_link_only_article_header_reduces_to_its_heading():
    body = ("<article><header><h1>Post title</h1><ul>%s</ul></header><p>%s</p></article>") % (
        _interlanguage_list(300),
        "Real article content. " * 40,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "# Post title" in out
    assert "Real article content." in out
    assert "Lang0" not in out


def test_article_header_byline_and_date_are_kept():
    # Standard semantic blog markup: only near-pure link lists are furniture.
    body = (
        "<article><header><h1>Why Rust</h1><p>By Jane Doe</p>"
        "<time>2026-07-12</time><p>A summary of what this essay argues.</p></header>"
        "<p>%s</p></article>"
    ) % ("Real article content. " * 40)
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "# Why Rust" in out
    assert "By Jane Doe" in out
    assert "2026-07-12" in out
    assert "A summary of what this essay argues." in out


def test_small_link_header_is_left_alone():
    # Under the size floor there is nothing large enough to displace an article.
    body = (
        "<article><header><h1>Post title</h1>"
        "<a href='/subscribe'>Subscribe now</a></header><p>%s</p></article>"
    ) % ("Real article content. " * 40)
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Subscribe now" in out
    assert "Real article content." in out


def test_unclosed_header_does_not_swallow_the_body():
    # Browsers adopt the rest of the subtree into an unclosed <header>.
    body = "<main><header><h1>Title</h1><p>%s</p></main>" % ("Article body text. " * 40)
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Article body text." in out
    assert "Title" in out


def test_unclosed_header_with_many_headings_keeps_body():
    # Headings survive, so a heading-rich page clears the size gate alone.
    sections = "".join(f"<h2>Section {i}</h2><p>{'Body prose here. ' * 10}</p>" for i in range(12))
    body = f"<main><header><h1>T</h1>{sections}</main>"
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Body prose here." in out
    assert "Section 0" in out


def test_header_strip_applies_without_article_or_main():
    body = ("<header><h1>Site name</h1><ul>%s</ul></header><p>%s</p>") % (
        _interlanguage_list(300),
        "Page prose without a main landmark. " * 20,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Page prose without a main landmark." in out
    assert "# Site name" in out
    assert "Lang0" not in out


def test_header_kept_in_unscoped_conversion():
    body = "<header><h1>Site</h1><a href='/x'>Nav link</a></header><p>Text.</p>"
    out = html_to_markdown(f"<body>{body}</body>")
    assert "Nav link" in out
    assert "Text." in out


def test_unclosed_header_in_truncated_scope_keeps_body():
    # A capped fetch ends before </main>: the flushed segment must still carry its dropped prose.
    sections = "".join(
        f"<h2>Section {i} of the article</h2><p>{'Body prose here. ' * 10}</p>" for i in range(12)
    )
    out = html_to_markdown(
        f"<body><main><header><h1>T</h1>{sections}",
        main_content = True,
    )
    assert "Body prose here." in out
    assert "Section 0 of the article" in out


def test_sibling_card_does_not_beat_an_article_with_a_swallowed_body():
    real = "<article><header><h1>Real</h1><p>%s</p></article>" % ("Real article body. " * 40)
    card = "<article><p>%s</p></article>" % ("Related card teaser. " * 12)
    out = html_to_markdown(f"<body>{real}{card}</body>", main_content = True)
    assert "Real article body." in out
    assert "Related card teaser." not in out


def test_text_heavy_header_is_kept_even_when_longer_than_the_body():
    # python.org keeps its hero carousel here, so size alone cannot condemn it.
    body = "<main><header><h1>Title</h1><p>%s</p></header><p>%s</p></main>" % (
        "Introductory hero text. " * 30,
        "The real body prose. " * 20,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "The real body prose." in out
    assert "Introductory hero text." in out
    assert "# Title" in out


def test_unclosed_link_in_unclosed_header_keeps_the_body():
    # The <a> adopts the body, so its text is not link furniture.
    body = "<main><header><h1>Title</h1><a href='/'>Home<p>%s</p>" % ("Article body text. " * 40)
    out = html_to_markdown(f"<body>{body}", main_content = True)
    assert "Article body text." in out
    assert "# Title" in out


def test_unclosed_link_does_not_hand_the_scope_to_a_sibling_card():
    real = "<article><header><h1>Real</h1><a href='/'>Home<p>%s</p></article>" % ("REAL " * 60)
    card = "<article><p>%s</p></article>" % ("CARD " * 30)
    out = html_to_markdown(f"<body>{real}{card}</body>", main_content = True)
    assert "REAL" in out
    assert "CARD" not in out


def test_entity_encoded_body_is_not_lost_to_an_unclosed_header():
    out = html_to_markdown(
        "<body><main><header><h1>T</h1><p>%s</p>" % ("&alpha;" * 400),
        main_content = True,
    )
    assert out.count("α") == 400


def test_header_closed_by_an_ancestor_is_kept_whole():
    # Without a matching </header> the header may have adopted the body, so keep all.
    body = "<div><header><h1>Site</h1><ul>%s</ul></div><p>%s</p>" % (
        _interlanguage_list(300),
        "The real body prose. " * 20,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "The real body prose." in out
    assert "# Site" in out
    assert "Lang0" in out


def test_unclosed_header_does_not_strip_a_short_article_it_adopted():
    body = "<main><header><h1>T</h1><ul>%s</ul><p>%s</p></main>" % (
        _interlanguage_list(300),
        "Short real article. " * 4,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Short real article." in out


def test_heading_inside_a_nested_buffer_is_kept_and_the_links_still_go():
    # Teeing keeps the title without forcing the language list back in.
    body = (
        "<main><header><blockquote><h1>Page Title</h1></blockquote><ul>%s</ul></header><p>%s</p></main>"
        % (
            _interlanguage_list(400),
            "Article body. " * 30,
        )
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Page Title" in out
    assert "Lang0" not in out
    assert out.index("Article body.") < 16000


def test_long_hrefs_count_toward_the_header_size_floor():
    # Short labels, huge destinations: the rendered links are what displace it.
    nav = "".join('<a href="https://e.com/p?%s=%d">L%d</a>' % ("q" * 1000, i, i) for i in range(30))
    body = "<main><header>%s</header><p>%s</p></main>" % (nav, "Article body. " * 30)
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "L0" not in out
    assert out.index("Article body.") < 16000


def test_linked_heading_does_not_condemn_the_byline_beside_it():
    body = (
        "<article><header><h1><a href='/p'>%s</a></h1><p>By Jane Doe, July 2026</p></header><p>%s</p></article>"
        % (
            "A Very Long Linked Headline About Assorted Things In The World Today " * 5,
            "Article body. " * 30,
        )
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "By Jane Doe" in out
    assert "Very Long Linked" in out


def test_anchor_without_href_is_prose_not_link_furniture():
    body = (
        "<article><header><h1>T</h1><a name='intro'>%s</a><p>Byline</p></header><p>%s</p></article>"
        % (
            "Introductory prose that renders as plain text. " * 8,
            "Article body. " * 30,
        )
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Introductory prose" in out
    assert "Byline" in out


def test_linked_heading_is_not_emitted_twice():
    body = (
        "<main><header><h1><a href='/post'>Title</a></h1><ul>%s</ul></header><p>%s</p></main>"
        % (
            _interlanguage_list(300),
            "Article body. " * 30,
        )
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "# [Title](/post)" in out
    assert out.count("Title") == 1


def test_article_still_beats_a_bigger_card_after_its_header_goes():
    # Dropped furniture is added back when ranking siblings, so a stripped header still wins.
    real = "<article><header><h1>Real</h1><ul>%s</ul></header><p>%s</p></article>" % (
        _interlanguage_list(300),
        "Short real body. " * 14,
    )
    card = "<article><p>%s</p></article>" % ("Related card teaser text here. " * 12)
    out = html_to_markdown(f"<body>{real}{card}</body>", main_content = True)
    assert "Short real body." in out
    assert "Related card teaser" not in out


def test_furniture_only_card_does_not_suppress_the_main_it_sits_in():
    # Furniture ranks, never makes eligible: a nav-only header returned a 225 char astro.build card.
    card = "<article><header>%s</header><p>%s</p></article>" % (
        "".join('<a href="/l%d">Language %d</a>' % (i, i) for i in range(120)),
        "Short teaser about the related thing, read more.",
    )
    body = "<p>%s</p>" % ("The real article body the reader wants. " * 40)
    out = html_to_markdown(f"<body><main>{body}{card}</main></body>", main_content = True)
    assert "The real article body the reader wants." in out
    assert "Language 7" not in out


def test_a_long_heading_href_does_not_condemn_the_rest_of_the_header():
    # The heading is kept anyway, so its size must not clear the floor for the metadata beside it.
    body = (
        "<article><header><h1><a href='/p?%s'>Title</a></h1>"
        "<a href='/author/jane'>Jane</a></header><p>%s</p></article>"
        % ("q" * 900, "Article body. " * 30)
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Title" in out
    assert "Jane" in out


def test_a_preserved_heading_is_terminated():
    # The closing tag's blank line lands after the heading mark pops, so render must supply it.
    body = "<main><header><h1>Title</h1><ul>%s</ul></header>Article body text here.</main>" % (
        _interlanguage_list(300)
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "# TitleArticle" not in out
    assert out.startswith("# Title")
    assert "Article body text here." in out


def test_scope_holding_only_furniture_does_not_win_or_blank_the_page():
    # Removed furniture must not make an empty candidate eligible, or the fetch returns nothing.
    body = "<main><header><ul>%s</ul></header></main><p>%s</p>" % (
        _interlanguage_list(300),
        "Real page body prose. " * 30,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Real page body prose." in out


def test_header_inside_an_enclosing_link_renders_once():
    body = "<main><a href='/x'><header><h1>T</h1></header></a><p>%s</p></main>" % (
        "Article body. " * 30
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert out.count("[# T](/x)") == 1


def test_table_nested_in_a_header_inside_a_cell_keeps_its_columns():
    html = "<table><tr><td><header><table><tr><td>x</td></tr></table></header></td></tr></table>"
    assert html_to_markdown(f"<body>{html}</body>", main_content = True) == html_to_markdown(
        f"<body>{html}</body>"
    )


def test_truncated_header_and_blockquote_keep_source_order():
    out = html_to_markdown("<body><main><header><h1>Title</h1><blockquote>Quote", main_content = True)
    assert out.index("Title") < out.index("Quote")


# Headers interact with every buffer, so enumerate the grid: that is where the one-off bugs live.
_GRID_HEADINGS = {
    "h1": "<h1>Page Title</h1>",
    "aria": "<div role='heading'>Page Title</div>",
    "in_quote": "<blockquote><h1>Page Title</h1></blockquote>",
    "is_link": "<a role='heading' href='/t'>Page Title</a>",
    "none": "",
}
_GRID_WRAPPERS = {
    "bare": ("", ""),
    "quote": ("<blockquote>", "</blockquote>"),
    "cell": ("<table><tr><td>", "</td></tr></table>"),
    "link": ("<a href='/x'>", "</a>"),
    "item": ("<ul><li>", "</li></ul>"),
    "pre": ("<pre>", "</pre>"),
}
_GRID_NESTED = {
    "bare": ("", ""),
    "quote": ("<blockquote>", "</blockquote>"),
    "cell": ("<table><tr><td>", "</td></tr></table>"),
    "pre": ("<pre>", "</pre>"),
}
_GRID_BODY = "Article body sentence. " * 30


@pytest.mark.parametrize("heading", sorted(_GRID_HEADINGS))
@pytest.mark.parametrize("wrapper", sorted(_GRID_WRAPPERS))
@pytest.mark.parametrize("nested", sorted(_GRID_NESTED))
@pytest.mark.parametrize("inner", ("link_dense", "content"))
@pytest.mark.parametrize("close_header", (True, False))
@pytest.mark.parametrize("close_nested", (True, False))
def test_header_survives_every_buffer_combination(
    heading, wrapper, nested, inner, close_header, close_nested
):
    open_wrap, close_wrap = _GRID_WRAPPERS[wrapper]
    open_nest, close_nest = _GRID_NESTED[nested]
    payload = (
        f"<ul>{_interlanguage_list(300)}</ul>"
        if inner == "link_dense"
        else "<p>By Jane Doe, 12 July 2026</p><p>A summary of the argument.</p>"
    )
    header = (
        f"<header>{_GRID_HEADINGS[heading]}{open_nest}{payload}"
        f"{close_nest if close_nested else ''}{'</header>' if close_header else ''}"
    )
    html = (
        f"<body><main>{open_wrap}{header}{close_wrap if close_header else ''}"
        f"<p>{_GRID_BODY}</p></main></body>"
    )
    out = html_to_markdown(html, main_content = True)
    # These hold for every shape, however malformed.
    assert "Article body sentence." in out, "body lost"
    assert out.strip(), "empty output"
    assert out.index("Article body sentence.") < 16000, "body pushed past the fetch cap"
    # The title holds only for closed markup with no <pre>: there a heading is verbatim text, and
    # unclosed shapes get best-effort recovery predating this pass.
    well_formed = close_header and close_nested and "pre" not in (wrapper, nested)
    if _GRID_HEADINGS[heading] and well_formed:
        assert out.count("Page Title") == 1, "title duplicated or lost"


def test_a_stub_cannot_outrank_a_real_article_on_removed_navigation():
    # A long title alone is not substantive retained content.
    stub = "<article><header><h1>%s</h1><ul>%s</ul></header></article>" % (
        "A Long Title That Exceeds Fifty Characters Easily Here",
        _interlanguage_list(300),
    )
    real = "<article><p>%s</p></article>" % ("Genuine full article body text. " * 20)
    out = html_to_markdown(f"<body>{stub}{real}</body>", main_content = True)
    assert "Genuine full article body" in out


def test_unclosed_link_in_a_closed_header_counts_as_furniture():
    # The </header> proves the anchor did not adopt the body.
    body = "<main><header><h1>T</h1><a href='/nav'>%s</header><p>%s</p></main>" % (
        "Navigation label text. " * 20,
        "Article body. " * 30,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Navigation label text." not in out
    assert "Article body." in out


def test_enclosing_anchor_text_counts_toward_header_density():
    body = "<main><a href='/nav'><header>%s</header></a><p>%s</p></main>" % (
        "".join(f"<span>Nav label {i}</span>" for i in range(40)),
        "Article body. " * 30,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Nav label 0" not in out
    assert "Article body." in out


def test_truncated_header_flushes_into_its_enclosing_buffer():
    # The header is inside the link/cell, so its text belongs there, in order.
    assert html_to_markdown(
        "<body><main><a href='/x'>before<header><h1>Title", main_content = True
    ).startswith("[before")
    assert html_to_markdown(
        "<body><main><table><tr><td><header><h1>Title", main_content = True
    ).startswith("|")


def test_empty_blocks_do_not_inflate_the_header_size():
    # _cleanup collapses them, so the size threshold must see the cleaned form.
    body = (
        "<article><header><h1>%s</h1>%s<a href='/x'>L</a></header></article>"
        "<article><p>%s</p></article>"
        % (
            "A Fairly Long Heading Title Here" * 2,
            "<div></div>" * 500,
            "Genuine full article body text. " * 20,
        )
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Genuine full article body" in out


def test_blockquoted_heading_is_not_counted_as_retained_prose():
    stub = "<article><blockquote><header><h1>%s</h1><ul>%s</ul></header></blockquote></article>" % (
        "A Long Title That Exceeds Fifty Characters Easily Here",
        _interlanguage_list(300),
    )
    real = "<article><p>%s</p></article>" % ("Genuine full article body text. " * 20)
    out = html_to_markdown(f"<body>{stub}{real}</body>", main_content = True)
    assert "Genuine full article body" in out


def test_list_left_open_in_a_header_does_not_indent_the_body():
    body = "<main><header><h1>T</h1><ul>%s</header><ul><li>Body item one</li></ul></main>" % (
        _interlanguage_list(300)
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    item = next(line for line in out.split("\n") if "Body item one" in line)
    assert item.startswith("*"), item


def test_deeply_nested_headers_do_not_blow_up_quadratically():
    # Header sizing must not rescan each parent's cumulative buffer.
    chunk = "<p>%s</p>" % ("filler text here. " * 100)

    def build(depth):
        return (
            "<body><main>"
            + "".join(f"<header>{chunk}" for _ in range(depth))
            + "</header>" * depth
            + "<p>body</p></main></body>"
        )

    import time

    timings = []
    for depth in (20, 80):
        html = build(depth)
        start = time.perf_counter()
        html_to_markdown(html, main_content = True)
        timings.append(time.perf_counter() - start)
    # Four times the depth and payload, nowhere near quadratic cost.
    assert timings[1] < timings[0] * 12, timings


def test_linked_heading_text_is_counted_once_for_the_size_floor():
    body = (
        "<article><header><h1><a href='/p'>%s</a></h1>"
        "<a href='/author'>Jane</a></header><p>%s</p></article>"
        % ("Headline word " * 60, "Article body. " * 30)
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Jane" in out


def test_fenced_code_counts_as_retained_content():
    # A leading # inside a fence is a comment, not a heading; scoring it as one loses this article.
    code = "<pre>%s</pre>" % "\n".join("# comment line %d" % i for i in range(16))
    article = "<article><header><h1>T</h1><ul>%s</ul></header>%s</article>" % (
        _interlanguage_list(300),
        code,
    )
    sibling = "<article><p>%s</p></article>" % ("Sibling teaser text here. " * 12)
    out = html_to_markdown(f"<body>{article}{sibling}</body>", main_content = True)
    assert "comment line 1" in out
    assert "Sibling teaser" not in out


_FENCE = "`" * 3


def test_dropped_furniture_cannot_dominate_sibling_ranking():
    # Credit must not decide the match: a teaser with a 1000 link header outranked five times its
    # own real text.
    teaser = "<article><header>%s</header><p>%s</p></article>" % (
        "".join('<a href="/l%d">Lang%d</a>' % (i, i) for i in range(1000)),
        "Teaser words here. " * 20,
    )
    real = "<article><p>%s</p></article>" % ("The genuine article body text goes here. " * 55)
    out = html_to_markdown(f"<body>{teaser}{real}</body>", main_content = True)
    assert "The genuine article body text goes here." in out
    assert "Teaser words here." not in out


def test_literal_bracket_paren_is_prose_not_a_destination():
    # No [ opened it, so "](" is literal and the parens hold prose; skipping them scored 192 of 295.
    article = "<article><p>%s](%s) %s</p></article>" % (
        "Real article prose that the reader wants to see. " * 3,
        "y" * 100,
        "Tail prose to finish the paragraph off here.",
    )
    sibling = "<article><p>%s</p></article>" % ("Unrelated sibling teaser words. " * 8)
    out = html_to_markdown(f"<body>{article}{sibling}</body>", main_content = True)
    assert "Real article prose that the reader wants to see." in out
    assert "Unrelated sibling teaser" not in out


def test_hand_preserved_heading_reaches_the_eligibility_tally():
    # The partial branch writes the title straight into heading_parts, so the gate needs telling
    # too or a title-only card reads as body prose.
    card = (
        '<article><header><a href="/h"><div role="heading">%s</div>%s</a><ul>%s</ul></header></article>'
        % (
            "Card Title Words " * 14,
            "nav text " * 40,
            _interlanguage_list(300),
        )
    )
    real = "<p>%s</p>" % ("The real page body text here. " * 30)
    out = html_to_markdown(f"<body><main>{real}{card}</main></body>", main_content = True)
    assert "The real page body text here." in out


def test_pre_inside_a_table_cell_is_drained_before_the_row():
    # The row is emitted, so an open <pre> swallowed it into a fence as CODEMARKER|  |.
    body = "<main><header><h1>T</h1><table><tr><td><pre>CODEMARKER</header><p>%s</p></main>" % (
        "Article body. " * 30,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "CODEMARKER|" not in out
    assert "Article body." in out


def test_post_processing_respects_the_widened_fence():
    # Both passes toggled on any ``` line, so the literal one closed the block and its code was
    # cleaned and de-boilerplated.
    code = "<pre>%s\nskip to content\n\nreal code line   \nmore code</pre>" % _FENCE
    body = "<article>%s<p>%s</p></article>" % (code, "Body text here. " * 20)
    out = html_to_markdown(f"<body><main>{body}</main></body>", main_content = True)
    assert "skip to content" in out
    assert "skip to content\n\nreal code line" in out
    assert "real code line   " in out


def test_unbalanced_destination_keeps_scoring_the_rest_of_the_line():
    # /docs/(draft never balances, so it is not a link; the scan must keep the prose on that line.
    article = '<article><p><a href="/docs/(draft">Doc</a> %s</p></article>' % (
        "Substantial article prose continues here. " * 6,
    )
    sibling = "<article><p>%s</p></article>" % ("Sibling teaser text. " * 12)
    out = html_to_markdown(f"<body>{article}{sibling}</body>", main_content = True)
    assert "Substantial article prose continues here." in out
    assert "Sibling teaser" not in out


def test_structural_headings_do_not_satisfy_the_eligibility_gate():
    # role="heading" renders as prose, so ATX reparsing missed it and a header-only card cleared
    # the gate on its title plus dropped-list credit.
    card = '<article><header><div role="heading">%s</div><ul>%s</ul></header></article>' % (
        "Card Title Words " * 14,
        _interlanguage_list(300),
    )
    real = "<article><p>%s</p></article>" % ("The real article body text here. " * 20)
    out = html_to_markdown(f"<body>{real}{card}</body>", main_content = True)
    assert "The real article body text here." in out
    assert "Card Title Words" not in out


def test_anchor_wrapping_a_heading_preserves_only_the_heading():
    # <a><h1>Title</h1>...nav...</a> carries a title; teeing the whole anchor returned 14k of nav.
    bulk = " ".join("NavWord%04d" % i for i in range(1200))
    body = (
        '<main><header><a href="/home"><h1>Title</h1>%s</a><ul>%s</ul></header><p>%s</p></main>'
        % (
            bulk,
            _interlanguage_list(300),
            "Article body. " * 30,
        )
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Title" in out
    assert "NavWord0000" not in out
    assert out.index("Article body.") < 16000


def test_link_destination_scan_balances_parentheses():
    # A destination may hold parens, so stopping at the first ) scored the rest of the URL as prose.
    query = "utm_source=x&" * 25
    card = '<article><header>%s</header><p><a href="/card(foo)?%s">Read</a></p></article>' % (
        "".join('<a href="/l%d">Lang%d</a>' % (i, i) for i in range(120)),
        query,
    )
    real = "<article><p>%s</p></article>" % ("The genuine article body text here. " * 20)
    out = html_to_markdown(f"<body>{real}{card}</body>", main_content = True)
    assert "The genuine article body text here." in out
    assert "/card(foo)?" not in out


def test_literal_fence_inside_pre_does_not_end_the_code_region():
    # A ``` line in the source is content; ending the fence there made the rest read as headings.
    code = "<pre>%s\n%s</pre>" % (
        _FENCE,
        "\n".join("# code line %d" % i for i in range(20)),
    )
    article = "<article><header><h1>T</h1><ul>%s</ul></header>%s</article>" % (
        _interlanguage_list(300),
        code,
    )
    sibling = "<article><p>%s</p></article>" % ("Sibling teaser text here. " * 12)
    out = html_to_markdown(f"<body>{article}{sibling}</body>", main_content = True)
    assert "code line 1" in out
    assert "Sibling teaser" not in out


def test_heading_through_a_nested_buffer_is_emitted_once():
    # The title was teed entering the blockquote and again on flush, so it was kept twice.
    body = (
        '<main><header><div role="heading"><blockquote>UniqueTitle</blockquote></div><ul>%s</ul></header><p>%s</p></main>'
        % (
            _interlanguage_list(300),
            "Article body. " * 30,
        )
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert out.count("UniqueTitle") == 1


def test_late_code_end_tag_after_a_recovered_header_is_a_no_op():
    # </code> arrives after </header>; the frame already closed the span, so a second emit is odd.
    body = "<main><header><h1>T</h1><code>navcode<ul>%s</ul></header><p>%s</p></code></main>" % (
        _interlanguage_list(300),
        "Article body. " * 30,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert out.count("`") % 2 == 0
    assert "Article body." in out


def test_header_text_is_sized_after_whitespace_collapses():
    # The run collapses to one space, so counting it raw pushed a short byline over the floor.
    byline = '<a href="/a">Jane%sDoe</a><time>July 2026</time>' % (" " * 300)
    body = "<main><header><h1>T</h1>%s</header><p>%s</p></main>" % (
        byline,
        "Article body. " * 30,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Jane Doe" in out
    assert "July 2026" in out


def test_link_destinations_do_not_count_as_retained_prose():
    # A tracking URL is not prose: this card shows 4 visible chars but its destination scored 339.
    query = "utm_source=x&" * 25
    card = '<article><header>%s</header><p><a href="/card?%s">Read</a></p></article>' % (
        "".join('<a href="/l%d">Lang%d</a>' % (i, i) for i in range(120)),
        query,
    )
    real = "<article><p>%s</p></article>" % ("The genuine article body text here. " * 20)
    out = html_to_markdown(f"<body>{real}{card}</body>", main_content = True)
    assert "The genuine article body text here." in out
    assert "/card?" not in out


def test_late_end_tag_cannot_replay_a_recovered_pre_block():
    # </header> arrives while <pre> is open, so the frame drains it; the later </pre> replayed it.
    body = "<main><header><h1>T</h1><ul>%s</ul><pre>%s</header><p>%s</p></pre></main>" % (
        _interlanguage_list(300),
        "NAVJUNK_MARKER\n" * 3,
        "Article body. " * 30,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "NAVJUNK_MARKER" not in out
    assert out.index("Article body.") < 16000


def test_aria_heading_accepts_a_fallback_role_token_list():
    # role is a token list authors use for fallbacks, so an exact match dropped the title.
    assert _is_aria_heading({"role": "heading"})
    assert _is_aria_heading({"role": "future-role heading"})
    assert _is_aria_heading({"role": "HEADING"})
    assert not _is_aria_heading({"role": "banner"})
    assert not _is_aria_heading({})
    body = (
        '<main><header><div role="future-role heading">Page Title</div><ul>%s</ul></header><p>%s</p></main>'
        % (
            _interlanguage_list(300),
            "Article body. " * 30,
        )
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Page Title" in out
    assert "Lang0" not in out


@pytest.mark.parametrize(
    ("heading_markup", "marker"),
    [
        ("<h1>Page Title</h1>", "Page Title"),
        ("<table><tr><td><h1>Page Title</h1></td></tr></table>", "Page Title"),
        ("<div role='heading' aria-level='1'>Page Title</div>", "Page Title"),
        ("<a role='heading' href='/title'>Page Title</a>", "Page Title"),
        ("<a href='/post'><h1>Page Title</h1></a>", "Page Title"),
        ("<hgroup><h1>Title</h1><p>Subtitle here</p></hgroup>", "Subtitle here"),
    ],
)
def test_heading_survives_a_stripped_header(heading_markup, marker):
    # However the title is expressed, reducing a link-only header keeps it and nothing else.
    body = "<main><header>%s<ul>%s</ul></header><p>%s</p></main>" % (
        heading_markup,
        _interlanguage_list(300),
        "Article body. " * 30,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert marker in out
    assert "Lang0" not in out
    assert "Article body." in out


@pytest.mark.parametrize(
    "body_markup",
    [
        # The blockquote encloses the header, so its content belongs to the frame.
        "<main><blockquote><header><h1>T</h1><ul>{links}</ul></header></blockquote><p>{body}</p></main>",
        # </blockquote> omitted: the frame must claim the content before judging.
        "<main><header><h1>T</h1><blockquote><ul>{links}</ul></header><p>{body}</p></main>",
        # </td> omitted, same requirement through the cell buffer.
        "<main><header><h1>T</h1><table><tr><td><ul>{links}</ul></header><p>{body}</p></main>",
    ],
)
def test_link_list_is_stripped_through_a_nested_buffer(body_markup):
    body = body_markup.format(links = _interlanguage_list(300), body = "Article body. " * 30)
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert "Lang0" not in out
    assert out.index("Article body.") < 16000


def test_hash_prefixed_prose_still_wins_its_scope():
    # Every line opens with a hash, so treating those as headings left the scope looking empty.
    lines = "".join("<p>#include &lt;header_%02d.h&gt;</p>" % i for i in range(14))
    article = "<article><header><h1>T</h1><ul>%s</ul></header>%s</article>" % (
        _interlanguage_list(300),
        lines,
    )
    sibling = "<article><p>%s</p></article>" % ("Sibling teaser text here. " * 12)
    out = html_to_markdown(f"<body>{article}{sibling}</body>", main_content = True)
    assert "#include" in out
    assert "Sibling teaser" not in out


def test_header_size_is_independent_of_the_buffer_it_renders_through():
    # Text counted entering a nested buffer AND on flush doubled it, so links stripped once quoted.
    links = "".join(
        '<a href="/very/long/section/path/number/%03d/index">L%03d</a>' % (i, i) for i in range(14)
    )
    wrapped = {
        "bare": links,
        "blockquote": f"<blockquote>{links}</blockquote>",
        "cell": f"<table><tr><td>{links}</td></tr></table>",
    }
    kept = set()
    for inner in wrapped.values():
        body = "<main><header><h1>T</h1>%s</header><p>%s</p></main>" % (
            inner,
            "Article body. " * 30,
        )
        kept.add("L000" in html_to_markdown(f"<body>{body}</body>", main_content = True))
    assert len(kept) == 1


def test_nested_inline_code_closes_every_span_it_opened():
    # Two <code> elements owe two backticks; as a flag the first </code> answered for both.
    body = "<main><article><p><code><code>x</code></code></p><p>%s</p></article></main>" % (
        "Body text here. " * 20,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert out.count("`") % 2 == 0
    assert "``x``" in out


def test_header_inside_open_inline_code_leaves_delimiters_paired():
    # The <code> opened outside the header, so closing it in the frame left </code> unpaired.
    body = "<main><code>head<header><h1>T</h1></header>tail</code><p>%s</p></main>" % (
        "Article body. " * 30,
    )
    out = html_to_markdown(f"<body>{body}</body>", main_content = True)
    assert out.count("`") % 2 == 0
    assert "`head`" not in out
    assert "Article body." in out


def test_nested_blockquote_prose_does_not_backtrack():
    # Heading detection must scan linearly: a regex backtracks exponentially on "> > > ... prose".
    import time

    html = (
        "<body><main>"
        + "<blockquote>" * 40
        + "<p>prose text here</p>"
        + "</blockquote>" * 40
        + "</main></body>"
    )
    start = time.perf_counter()
    html_to_markdown(html, main_content = True)
    assert time.perf_counter() - start < 1.0


def test_deeply_nested_tags_stay_linear():
    # _close_implicit must not rescan the whole open-tag stack per start tag.
    import time

    def build(count):
        return (
            "<body><main>"
            + "<header><h1>T</h1>" * count
            + "</header>" * count
            + "<p>body</p></main></body>"
        )

    timings = []
    for count in (1000, 4000):
        start = time.perf_counter()
        html_to_markdown(build(count), main_content = True)
        timings.append(time.perf_counter() - start)
    # Four times the tags, well under sixteen times the work.
    assert timings[1] < timings[0] * 12, timings
