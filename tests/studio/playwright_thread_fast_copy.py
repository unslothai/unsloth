# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""BYTE-FOR-BYTE, over every construct and over many partial selections.

studio/frontend/src/components/assistant-ui/thread-fast-copy.ts writes the thread's `text/plain`
clipboard payload itself instead of letting the browser build the styled flavour. That is only
safe if the string it writes is the string the browser would have written, and no unit test can
show that: the unit tests run against a hand-rolled DOM stub, so they check the patch's own
bookkeeping and cannot see how a real engine serialises anything. This file is the part that
needs a real browser and a real clipboard.

Two claims are under test and they are different claims:

  1. where the serialiser ANSWERS, the string is identical to what the engine puts on the
     clipboard for the same selection;
  2. where the gate REFUSES, the refusal is not gratuitous -- a form control is refused because
     the engine really does emit its value, and an unmapped engine is refused because its own
     `toString()` really does disagree with its clipboard.

A refusal that happens to be right by accident is not a pass, so the refusals are checked
separately rather than counted as successes.

WHAT IS BUILT, AND WHY IT IS BUILT. The bundle comes from the TypeScript in this tree, compiled
by studio/frontend/scripts/build-fast-copy-bundle.mjs. Proving a hand-kept reference implementation would
only prove the reference implementation; the thing that reaches a user is the module, so the
module is what is loaded into the page.

WHAT THIS DOES NOT COVER. `decideThreadCopy`'s branches that need no DOM -- an already-handled
event, a missing `clipboardData`, a copy originating in a text control -- are unit tested in
studio/frontend/tests/thread-fast-copy.test.ts and are not re-driven here. This file drives the
two halves that are only true of a real engine: `faithfulSelectionText` and
`engineClipboardIsMapped`. The gate itself is called for real, so a construct's verdict comes
from the shipped decision function rather than from a re-statement of it; only `scopeElement`,
which the module does not export, is restated in the page shim.

Prerequisites:

    (cd studio/frontend && npm ci)      # the bundle is built with the frontend's own vite
    python3 -m playwright install chromium webkit

Run:

    python3 tests/studio/playwright_thread_fast_copy.py                # chromium, then webkit
    python3 tests/studio/playwright_thread_fast_copy.py chromium

Exit codes:

    0  every claim above held, OR a prerequisite is missing and the run was SKIPPED
    1  at least one divergence was proven
    2  the harness itself could not run (the bundle would not build)

It is deliberately not wired into Frontend CI, and is listed with its reason in
tests/studio/test_playwright_suites_run_in_ci.py. The short version: that job installs Chromium
only, and half of what is asserted here is about the engine that is NOT Chromium.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FRONTEND = REPO / "studio" / "frontend"
BUNDLER = FRONTEND / "scripts" / "build-fast-copy-bundle.mjs"

sys.path.insert(0, str(HERE))
from _thread_fast_copy_constructs import (  # noqa: E402
    CONSTRUCTS,
    IMG,
    INSIDE_SCOPE,
    MUST_REFUSE,
    NO_COPY,
)

IMG_TAG = IMG + 'alt="a cat">'


class MissingPrerequisite(Exception):
    """Something the run needs is not installed. Reported as a SKIP, never as a pass."""


# : The bundle exposes the module's exports;
SHIM = """
window.__fastCopy = function () {
  const M = window.SBFastCopy;
  if (!M) return { text: null, reason: "the module did not load" };
  const viewport = document.getElementById("v");
  const selection = window.getSelection();
  const event = {
    defaultPrevented: false,
    target: selection.anchorNode,
    clipboardData: { setData() {} },
  };
  const decision = M.decideThreadCopy(
    event, selection, viewport, M.engineClipboardIsMapped(window),
  );
  if (decision.kind !== "fast") return { text: null, reason: decision.reason };
  let root = selection.getRangeAt(0).commonAncestorContainer;
  if (root.nodeType !== 1) root = root.parentElement;
  return { text: M.faithfulSelectionText(selection, root || viewport), reason: null };
};
"""

PAGE = """<!doctype html><meta charset=utf-8>
<div id="v">__BODY__</div>
<textarea id="sink" style="position:fixed;bottom:0"></textarea>"""

SELECT_ALL = "window.getSelection().selectAllChildren(document.getElementById('v'))"

# : Partial selections, because a whole-element selection is the easy case and the range's :
PARTIALS = """
  const v = document.getElementById('v');
  const walk = document.createTreeWalker(v, NodeFilter.SHOW_TEXT);
  const texts = []; let n; while ((n = walk.nextNode())) texts.push(n);
  const out = [];
  if (texts.length) {
    const mk = (a, ao, b, bo) => { const r = document.createRange();
      r.setStart(a, ao); r.setEnd(b, bo); return r; };
    out.push(mk(texts[0], Math.min(2, texts[0].length), texts[texts.length-1],
                Math.max(0, texts[texts.length-1].length - 2)));
    if (texts.length > 1) out.push(mk(texts[0], 0, texts[1], texts[1].length));
    out.push(mk(texts[texts.length-1], 0, texts[texts.length-1], texts[texts.length-1].length));
  }
  window.__ranges = out;
"""

UNMAPPED = "unmapped-engine"
FORM_CONTROL = "form-control"


def build_module() -> str:
    """The shipped TypeScript, compiled to an IIFE bundle, as source text."""
    if not (FRONTEND / "node_modules").is_dir():
        raise MissingPrerequisite(
            f"{FRONTEND.relative_to(REPO)}/node_modules is missing; "
            f"run `cd {FRONTEND.relative_to(REPO)} && npm ci` first"
        )
    with tempfile.TemporaryDirectory(prefix = "fastcopy-bundle") as out:
        done = subprocess.run(
            ["node", str(BUNDLER.relative_to(FRONTEND)), out],
            cwd = FRONTEND,
            capture_output = True,
            text = True,
        )
        built = Path(out) / "fastcopy.js"
        if done.returncode != 0 or not built.is_file():
            raise RuntimeError(
                f"building {BUNDLER.name} failed ({done.returncode}):\n"
                f"{done.stdout}\n{done.stderr}"
            )
        return built.read_text(encoding = "utf-8")


def launch(playwright, engine: str):
    """The browser, or a MissingPrerequisite if its executable was never downloaded."""
    args = {"args": ["--no-sandbox"]} if engine == "chromium" else {}
    try:
        return getattr(playwright, engine).launch(**args)
    except Exception as error:  # noqa: BLE001 - playwright raises its own Error type
        text = str(error)
        if "Executable doesn't exist" in text or "playwright install" in text:
            raise MissingPrerequisite(
                f"the {engine} browser is not installed; "
                f"run `python3 -m playwright install {engine}`"
            ) from error
        raise


def prime_clipboard(page, sentinel: str) -> None:
    """Put a known string on the clipboard, so a copy that writes nothing is visible as such.

    Without it, a selection the engine considers empty leaves the clipboard untouched and the
    paste below reports the PREVIOUS construct's result as this one's.
    """
    page.evaluate(
        "(s) => { const t = document.getElementById('sink');"
        " t.value = s; t.focus(); t.select(); document.execCommand('copy');"
        " t.value = ''; }",
        sentinel,
    )


def read_clipboard(page, sentinel: str, restore: str) -> str | None:
    """What the engine's own copy of this selection puts on the clipboard, or None for nothing."""
    page.evaluate(f"() => {{ {restore} }}")
    page.evaluate("() => document.execCommand('copy')")
    page.click("#sink")
    page.keyboard.press("ControlOrMeta+v")
    got = page.evaluate("() => document.getElementById('sink').value")
    page.evaluate("() => document.getElementById('sink').value = ''")
    return None if got == sentinel else got


def compare(engine: str, label: str, verdict: dict, native: str) -> list[str]:
    """One answered-or-refused verdict against one real clipboard string."""
    if verdict["text"] is None:
        if verdict["reason"] == UNMAPPED:
            return []
        return [f"{engine}/{label}: refused ({verdict['reason']}) with no divergence"]
    if verdict["text"] != native:
        return [
            f"{engine}/{label}: MISMATCH\n"
            f"      native  {native!r}\n      ours    {verdict['text']!r}"
        ]
    return []


# : The floor on how many selections a run has to actually reach a verdict on.
# A comparison is : skipped when the engine copied nothing at all, and a driver that skipped every one of them : would
# print CLEAN having proven nothing
EXPECTED_VERDICTS = len(CONSTRUCTS) - len(MUST_REFUSE) - len(NO_COPY)


class Tally:
    """What a run reached a verdict on, so an empty run cannot read as a clean one."""

    def __init__(self) -> None:
        self.problems: list[str] = []
        # : Answered, and compared byte for byte against the real clipboard.
        self.compared = 0
        # : Refused because this engine's clipboard mapping is not one the module has proven.
        self.refused = 0
        # : Of those, how many really do differ from the engine's own `toString()`.
        self.diverged = 0

    def record(self, engine: str, label: str, verdict: dict, native: str) -> None:
        if verdict["reason"] == UNMAPPED:
            self.refused += 1
            return
        self.compared += 1
        self.problems += compare(engine, label, verdict, native)


def check(engine: str, candidate: str) -> Tally:
    """Every claim in the module docstring, driven against one real engine."""
    tally = Tally()
    problems = tally.problems
    with sync_playwright() as playwright:
        browser = launch(playwright, engine)
        page = browser.new_page()

        for index, (name, body) in enumerate(CONSTRUCTS.items()):
            page.set_content(PAGE.replace("__BODY__", body))
            page.add_script_tag(content = candidate)
            sentinel = f"__s{index}__"
            page.evaluate(f"() => {{ {SELECT_ALL} }}")
            # THE SERIALISED DOM, not the computed style.
            dom_before = page.evaluate("() => document.getElementById('v').outerHTML")
            verdict = page.evaluate("() => window.__fastCopy()")
            dom_after = page.evaluate("() => document.getElementById('v').outerHTML")
            if dom_before != dom_after:
                problems.append(
                    f"{engine}/{name}: the serialised DOM changed, "
                    f"{len(dom_before)} chars -> {len(dom_after)}"
                )
            after = page.evaluate("() => window.getSelection().toString()")
            page.evaluate(f"() => {{ {SELECT_ALL} }}")
            before = page.evaluate("() => window.getSelection().toString()")
            if after != before:
                problems.append(f"{engine}/{name}: the serialiser did not restore the selection")

            # Both engines' clipboards fold a no-break space to a plain one and neither `toString()` does, so that
            raw = before.replace("\u00a0", " ")
            prime_clipboard(page, sentinel)
            native = read_clipboard(page, sentinel, SELECT_ALL)

            if name in MUST_REFUSE:
                # Not merely "did not answer":
                if verdict["text"] is not None:
                    problems.append(f"{engine}/{name}: ANSWERED a construct measured as divergent")
                elif verdict["reason"] not in (FORM_CONTROL, UNMAPPED):
                    problems.append(
                        f"{engine}/{name}: refused as {verdict['reason']!r}, "
                        f"not as a form control"
                    )
                continue
            if name in NO_COPY or native is None:
                continue
            # An engine-wide refusal is only honest if this engine's clipboard really does disagree with its own
            if verdict["reason"] == UNMAPPED and native != raw:
                tally.diverged += 1
            tally.record(engine, name, verdict, native)

        # ENDPOINTS EXPRESSED AS ELEMENT/CHILD OFFSETS ---- The alt holders are inserted BEFORE their images, so a
        for label, selection_js in (
            ("element offsets forward", "s.setBaseAndExtent(p, 0, p, 3)"),
            ("element offsets backward", "s.setBaseAndExtent(p, 3, p, 0)"),
        ):
            page.set_content(
                PAGE.replace("__BODY__", f'<p id="ep">{IMG_TAG}{IMG_TAG}tail text</p>')
            )
            page.add_script_tag(content = candidate)
            restore = (
                "const p = document.getElementById('ep');"
                " const s = window.getSelection(); " + selection_js + ";"
            )
            sentinel = f"__ep_{label.replace(' ', '_')}__"
            page.evaluate(f"() => {{ {restore} }}")
            verdict = page.evaluate("() => window.__fastCopy()")
            prime_clipboard(page, sentinel)
            native = read_clipboard(page, sentinel, restore)
            if native is None:
                continue
            tally.record(engine, label, verdict, native)

        # A BACKWARD SELECTION MUST COME BACK BACKWARD ---- A cloned Range carries ordered boundaries and no direction
        page.set_content(
            PAGE.replace(
                "__BODY__",
                '<p id="pa">first paragraph</p><p id="pb">second ' + IMG_TAG + " paragraph</p>",
            )
        )
        page.add_script_tag(content = candidate)
        page.evaluate("""() => {
          const a = document.getElementById('pa').firstChild;
          const b = document.getElementById('pb').firstChild;
          window.getSelection().setBaseAndExtent(b, b.length, a, 0);
          // The engine probe caches its answer on the window, and `set_content` does not replace
          // the window, so by here it has already run against some earlier construct. Clearing it
          // makes THIS selection the one the probe takes away and rebuilds -- which is a second
          // place the direction can be lost, and the only one a user meets on their first copy.
          delete window.__sbFastCopyMapped;
        }""")
        backward = page.evaluate("() => window.__fastCopy()")
        still = page.evaluate("""() => {
          const a = document.getElementById('pa').firstChild;
          return window.getSelection().focusNode === a;
        }""")
        if backward["reason"] != UNMAPPED and not still:
            problems.append(
                f"{engine}/backward selection: came back forward; the anchor and focus were "
                "swapped, so the user's next Shift+Arrow would move the opposite edge"
            )

        for name, (body, target) in INSIDE_SCOPE.items():
            page.set_content(PAGE.replace("__BODY__", body))
            page.add_script_tag(content = candidate)
            restore = f"window.getSelection().selectAllChildren({target})"
            sentinel = f"__inside_{name.replace(' ', '_')}__"
            page.evaluate(f"() => {{ {restore} }}")
            verdict = page.evaluate("() => window.__fastCopy()")
            prime_clipboard(page, sentinel)
            native = read_clipboard(page, sentinel, restore)
            if native is None:
                continue
            tally.record(engine, name, verdict, native)

        safe = "".join(
            body
            for name, body in CONSTRUCTS.items()
            if name not in MUST_REFUSE and name not in NO_COPY
        )
        page.set_content(PAGE.replace("__BODY__", safe))
        page.add_script_tag(content = candidate)
        page.evaluate(f"() => {{ {PARTIALS} }}")
        for index in range(page.evaluate("() => window.__ranges.length")):
            restore = (
                "const s = window.getSelection(); s.removeAllRanges();"
                f" s.addRange(window.__ranges[{index}]);"
            )
            sentinel = f"__partial{index}__"
            page.evaluate(f"() => {{ {restore} }}")
            verdict = page.evaluate("() => window.__fastCopy()")
            prime_clipboard(page, sentinel)
            native = read_clipboard(page, sentinel, restore)
            if native is None:
                continue
            tally.record(engine, f"partial{index}", verdict, native)

        browser.close()

    if tally.refused and not tally.diverged:
        problems.append(
            f"{engine}: refused as an unmapped engine, but not one construct's clipboard "
            f"differs from its own toString() here, so the refusal is unjustified"
        )
    if tally.compared + tally.refused < EXPECTED_VERDICTS:
        problems.append(
            f"{engine}: only {tally.compared + tally.refused} selections reached a verdict, "
            f"under the {EXPECTED_VERDICTS} this document contains. The run proved nothing; "
            f"the fixture or the clipboard round trip is broken, not the module."
        )
    return tally


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "engines",
        nargs = "*",
        default = ["chromium", "webkit"],
        help = "browser engines to drive (default: chromium webkit)",
    )
    engines = parser.parse_args().engines

    try:
        candidate = build_module() + SHIM
    except MissingPrerequisite as error:
        print(f"SKIP: {error}")
        return 0
    except RuntimeError as error:
        print(f"HARNESS FAILURE: {error}")
        return 2

    problems: list[str] = []
    ran = 0
    for engine in engines:
        try:
            tally = check(engine, candidate)
        except MissingPrerequisite as error:
            print(f"=== {engine}: SKIP, {error}")
            continue
        ran += 1
        problems += tally.problems
        outcome = "CLEAN" if not tally.problems else f"{len(tally.problems)} problem(s)"
        tail = f", {tally.compared} selections matched the clipboard byte for byte"
        if tally.refused:
            tail += (
                f", {tally.refused} refused as an unmapped engine "
                f"({tally.diverged} of them measurably divergent)"
            )
        print(f"=== {engine}: {outcome}{tail}")
        for problem in tally.problems:
            print("    " + problem)
    if not ran:
        print("SKIP: no engine could be launched; nothing was proven")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
