# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat HTML opened in the user's default browser is served at a capability URL. It must keep
the in-app frame's isolation (an opaque-origin sandbox, the network switch) as a top-level tab."""

import asyncio

import routes.inference as inf_mod


def _stage(
    html: str,
    allow_network: bool = False,
    subject: str = "u",
) -> str:
    request = inf_mod.ArtifactPreviewPageRequest(html = html, allow_network = allow_network)
    result = asyncio.run(inf_mod.create_artifact_preview_page(request, current_subject = subject))
    return result["path"].rsplit("/", 1)[1]


def _serve(token: str):
    return asyncio.run(inf_mod.artifact_preview_page(token))


def test_a_staged_page_is_served_sandboxed_with_the_strict_policy():
    response = _serve(_stage("<!doctype html><html><head><title>Hi</title></head></html>"))
    csp = response.headers["content-security-policy"]
    assert response.status_code == 200
    assert "sandbox allow-scripts" in csp
    assert "allow-same-origin" not in csp
    assert "connect-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    assert response.headers["cache-control"] == "no-store"


def test_the_network_switch_carries_over():
    csp = _serve(_stage("<p>x</p>", allow_network = True)).headers["content-security-policy"]
    assert "connect-src http: https:" in csp
    assert "allow-same-origin" not in csp


def test_the_storage_fallback_goes_right_after_a_leading_doctype():
    body = _serve(_stage("<!DOCTYPE html><html><head><title>Hi</title>")).body.decode()
    assert body.startswith("<!DOCTYPE html><script>")
    assert "localStorage" in body
    commented = _serve(_stage("<!-- note -->\n<!doctype html><p>x</p>")).body.decode()
    assert commented.startswith("<!-- note -->\n<!doctype html><script>")


def test_head_text_in_a_script_or_comment_is_not_taken_for_the_head():
    page = '<!doctype html><script>const template = "<head>";</script><!-- <html> -->'
    body = _serve(_stage(page)).body.decode()
    assert body.startswith("<!doctype html><script>(() =>")
    assert body.endswith(page[len("<!doctype html>") :])
    # No doctype: first, so the page's own markup stays whole.
    assert _serve(_stage('<p>"<head>"</p>')).body.decode().endswith('</script><p>"<head>"</p>')


def test_an_unknown_token_is_not_found():
    assert _serve("not-a-token").status_code == 404


def test_tokens_are_unguessable_and_distinct():
    first, second = _stage("<p>a</p>"), _stage("<p>a</p>")
    assert first != second
    assert len(first) >= 24


def test_one_account_cannot_evict_another_accounts_pages(monkeypatch):
    monkeypatch.setattr(inf_mod, "_artifact_pages", {})
    theirs = _stage("<p>theirs</p>", subject = "a")
    mine = [_stage(f"<p>{i}</p>", subject = "b") for i in range(inf_mod._ARTIFACT_PAGE_MAX_PAGES + 5)]
    assert _serve(theirs).status_code == 200
    # The account's own oldest pages go first.
    assert _serve(mine[0]).status_code == 404
    assert _serve(mine[-1]).status_code == 200
    assert (
        sum(entry[3] == "b" for entry in inf_mod._artifact_pages.values())
        == inf_mod._ARTIFACT_PAGE_MAX_PAGES
    )


def test_byte_budgets_bound_each_account_and_the_total(monkeypatch):
    big = "x" * 1500
    size = len(inf_mod._artifact_page_with_prelude(big).encode())
    monkeypatch.setattr(inf_mod, "_artifact_pages", {})
    monkeypatch.setattr(inf_mod, "_ARTIFACT_PAGE_MAX_SUBJECT_BYTES", 2 * size - 1)
    monkeypatch.setattr(inf_mod, "_ARTIFACT_PAGE_MAX_TOTAL_BYTES", 2 * size)
    first, second = _stage(big, subject = "a"), _stage(big, subject = "a")
    # One account holds one page of this size: its second evicts its first.
    assert _serve(first).status_code == 404 and _serve(second).status_code == 200
    third = _stage(big, subject = "b")
    assert _serve(second).status_code == 200 and _serve(third).status_code == 200
    # The total holds two: a third account's page evicts the oldest overall.
    fourth = _stage(big, subject = "c")
    assert _serve(second).status_code == 404 and _serve(fourth).status_code == 200
