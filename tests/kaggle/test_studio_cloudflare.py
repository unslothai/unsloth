# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The quick tunnel: the one assertion here that reaches the public internet.

Because it does, what it claims is narrow and what it refuses to claim is
written down.

Three rules, and the third is the one that makes opening a public URL from a CI
machine defensible at all: the tunnel must REFUSE an unauthenticated request.
Without it this check would prove that anyone holding the URL can drive
inference on the box, and call that a pass.

Two details are load-bearing rather than stylistic:

* `--host 0.0.0.0`. Studio raises a quick tunnel for WILDCARD binds only; on
  127.0.0.1 there is nothing to publish, no URL is printed, and a working
  feature reads as a broken one.
* the negative lookahead on `api.trycloudflare.com`. That host appears in
  cloudflared's own FAILURE lines ("failed to request quick Tunnel: Post
  https://api.trycloudflare.com/tunnel"), so a naive match extracts a URL from
  the message that says there is no URL. Studio's own matcher carries the same
  lookahead, and this one is checked against both strings.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "studio_gpu" / "run_studio_gpu.py"
SRC = PAYLOAD.read_text(encoding = "utf-8")


def _func(name: str) -> ast.FunctionDef:
    for cls in ast.walk(ast.parse(SRC)):
        if not isinstance(cls, ast.ClassDef):
            continue
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
    raise AssertionError(f"no method named {name!r}")


def _body(name: str = "assert_cloudflare") -> str:
    return ast.get_source_segment(SRC, _func(name)) or ""


def test_the_assertion_exists_and_is_driven_from_the_run():
    assert _body()
    assert "self.assert_cloudflare()" in _body("execute")


def _flat(text: str) -> str:
    """Whitespace-stripped, because the repo's formatter reflows an argument
    list to one entry per line and a guard matching the unformatted spelling
    goes red on a reformat rather than on a regression. That has happened three
    times in this payload now."""
    return "".join(text.split())


def test_it_binds_a_wildcard_or_there_is_nothing_to_publish():
    body = _flat(_body())
    assert '"--host","0.0.0.0",' in body
    assert '"--cloudflare",' in body


def test_the_url_matcher_rejects_cloudflareds_own_failure_host():
    """Executed, not eyeballed. The failure line contains a trycloudflare URL
    and means the opposite of a tunnel."""
    body = _body()
    literal = re.search(r"re\.search\((r\"[^\"]+\")", body)
    assert literal, "no URL matcher found"
    pattern = re.compile(ast.literal_eval(literal.group(1)))
    assert pattern.search("Your quick Tunnel: https://brave-cat-runs.trycloudflare.com")
    assert not pattern.search(
        'failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel"'
    ), "the failure host was accepted as a working tunnel URL"


def test_the_tunnel_must_actually_serve():
    """ "a URL was printed" and "a URL that works" are different claims. A
    tunnel whose edge never registers answers 530."""
    func = _func("assert_cloudflare")
    assert any(
        isinstance(n, ast.If) and ast.unparse(n.test) == "code != 200" for n in ast.walk(func)
    ), "nothing checks that the public URL answers"


def test_an_unauthenticated_request_through_the_tunnel_must_be_refused():
    """The rule that makes opening the URL defensible. Asserted structurally so
    a message rewrite cannot satisfy it."""
    func = _func("assert_cloudflare")
    unauth = [
        n
        for n in ast.walk(func)
        if isinstance(n, ast.Call)
        and any(k.arg == "auth" and k.value.value is False for k in n.keywords)
    ]
    assert len(unauth) >= 2, (
        "both the health probe and the inference probe must go through the "
        "tunnel WITHOUT credentials"
    )
    assert any(
        isinstance(n, ast.If) and "code < 400" in ast.unparse(n.test) for n in ast.walk(func)
    ), "an accepted unauthenticated request must be a failure"


def test_the_no_tunnel_excuse_is_narrow():
    """Reported-not-failed applies ONLY when nothing was published, because
    then there is nothing to have gone wrong with. Once a URL exists, every
    rule above is live."""
    func = _func("assert_cloudflare")
    excuses = [
        n for n in ast.walk(func) if isinstance(n, ast.If) and ast.unparse(n.test) == "url is None"
    ]
    assert excuses, "the excuse branch is gone or is no longer gated on the URL"
    for node in excuses:
        # node.body only. Walking the whole If would sweep in the `else`
        # branch, which is where every real failure lives, and the guard would
        # then fail on correct code -- it did, the first time this was written.
        appended = [
            n
            for stmt in node.body
            for n in ast.walk(stmt)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "append"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "failures"
        ]
        assert not appended, "the no-tunnel branch reports; it does not fail"
    body = _body()
    assert "no_tunnel_reason" in body, "a reason must be carried from the log"


def test_the_public_url_is_registered_as_a_secret():
    """It is a live public route to this machine and the artifact is read by
    people who are not running it."""
    body = _body()
    assert "self.secrets.add(url)" in body
    assert '"unsloth_cloudflare.log",' in _body(
        "emit_evidence"
    ), "a log nobody packs is a log nobody redacts"


def test_the_child_is_always_torn_down():
    """A tunnel left up outlives the assertion and keeps the machine public."""
    func = _func("assert_cloudflare")
    finals = "\n".join(
        ast.unparse(n) for t in ast.walk(func) if isinstance(t, ast.Try) for n in t.finalbody
    )
    assert "proc.terminate()" in finals
    assert "proc.kill()" in finals


def test_it_can_be_switched_off():
    """The only assertion here that reaches the public internet should be
    refusable without editing the payload."""
    assert '"--no-cloudflare-check"' in SRC
    assert "self.args.cloudflare_check" in _body("execute")
