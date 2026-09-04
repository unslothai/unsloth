// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import test from "node:test";

/**
 * The one property that makes this change different from the viewport gate that came before it,
 * pinned as source facts.
 *
 * The earlier attempt gated a fence on viewport entry AND on viewport exit. Because the gate ran
 * both ways, collapsing a reasoning pane pushed fences out of view and generated re-highlight
 * work instead of saving it: predicted -55% on `reasoning_toggle.close_ms`, measured +12.7%
 * slower, and closed on that number.
 *
 * Everything below exists so that reintroducing a downgrade edge fails a test rather than a
 * benchmark two days later. These are deliberately source-level assertions: the module is a React
 * hook over IntersectionObserver, so a behavioural test would need a DOM, and the invariant worth
 * protecting is structural anyway -- "no code path sets this back to false".
 */

const SOURCE = readFileSync(
  new URL("../src/components/assistant-ui/code-fence-defer.tsx", import.meta.url),
  "utf8",
);

const MARKDOWN_TEXT = readFileSync(
  new URL("../src/components/assistant-ui/markdown-text.tsx", import.meta.url),
  "utf8",
);

test("the latch is only ever set to true", () => {
  const writes = SOURCE.match(/setLatched\([^)]*\)/g) ?? [];
  assert.ok(writes.length > 0, "expected at least one write to the latch");
  for (const write of writes) {
    assert.equal(
      write,
      "setLatched(true)",
      `the latch must never be cleared; found ${write}. A downgrade edge is what made the ` +
        "previous viewport gate measure slower than doing nothing.",
    );
  }
});





test("a completing stream cannot downgrade a fence that was highlighted while it streamed", () => {
  // `streaming` goes true -> FALSE at the closing delimiter. Deriving `reached` from it alone
  // hands a finished fence back the plain shell, which is the reverse edge in miniature.
  assert.ok(
    /if\s*\(!enabled\s*\|\|\s*latched\s*\|\|\s*!streaming\)\s*return;/.test(SOURCE),
    "a streaming fence must LATCH, not merely read as reached while the flag is live",
  );
  const derived = SOURCE.match(/const reached = [^;]+;/)?.[0] ?? "";
  assert.ok(
    derived.includes("latched"),
    `the derived value must include the latch; found ${derived}`,
  );
});

test("the observer is rooted at the nearest SCROLLING ancestor, found not named", () => {
  // Two failures this pins, and they are different from each other.
  //
  // `root: null` is the document viewport, so `rootMargin` expands a rectangle that is not the
  // one clipping and the lookahead is worth nothing. That was the review item.
  //
  // Matching two known selectors walks past the reasoning pane, which while streaming is an
  // `overflow-y-auto` `max-h-64` window holding an arbitrarily long trace. Intersection was still
  // correct there, because intermediate scrollers clip, but the one-viewport lookahead was not:
  // measured 3 of 10 fences intersecting with and without the margin when rooted at the thread
  // viewport, against 5 of 10 rooted at the 256px pane.
  assert.ok(
    /const near = scrollerOf\(node\);/.test(SOURCE) && /\{ root, rootMargin: REACH_MARGIN \}/.test(SOURCE),
    "the observer root must be the fence's own scrolling ancestor",
  );
  assert.ok(
    !/closest<HTMLElement>\("\[data-slot='thread-viewport'\]"\)/.test(SOURCE),
    "a named-selector lookup walks past the reasoning pane's scroller, which matches neither name",
  );
  const fn = SOURCE.slice(SOURCE.indexOf("const scrollerOf"), SOURCE.indexOf("const scrollerOf") + 320);
  assert.ok(
    fn.includes("el.parentElement") && fn.includes("isScrollable(el)"),
    "it must WALK to the nearest scrollable ancestor rather than matching known names",
  );
  const pred = SOURCE.slice(SOURCE.indexOf("const isScrollable"), SOURCE.indexOf("const scrollerOf"));
  for (const token of ['"auto"', '"scroll"', '"overlay"', "scrollHeight > el.clientHeight"]) {
    assert.ok(pred.includes(token), `the scrollable test must consider ${token}`);
  }
});

test("the pre-paint gate re-runs when the roots are rebound", () => {
  // The ResizeObserver bumps `generation` when the reasoning pane stops scrolling, and the
  // passive effect rebuilds its observers off that. The PRE-PAINT effect has to re-run on the
  // same bump, or a fence that the expanding pane has just brought inside the outer viewport
  // stays on its plain shell through the commit the rebind causes, and the replacement observer
  // delivers asynchronously, so the shell is painted.
  //
  // That is an ON SCREEN difference, which is the one kind this change is not allowed to have.
  const prepaint = SOURCE.slice(
    SOURCE.indexOf("THE FIRST FRAME, which the observer cannot cover"),
    SOURCE.indexOf("// The one-way edge."),
  );
  assert.ok(prepaint.length > 200, "the pre-paint effect must still be findable by its comment");
  assert.ok(
    /\}, \[reached, host, generation\]\);/.test(prepaint),
    "the pre-paint gate must depend on the rebind generation, not just on reached and host",
  );
  // And it has to be the gate that actually latches, not some other effect in the slice.
  assert.ok(
    prepaint.includes("setLatched(true)") && prepaint.includes("useLayoutEffect"),
    "the effect this pins must be the pre-paint latch itself",
  );
});

test("with the flag off the hook writes no state, builds no observer and reads no layout", () => {
  const hook = SOURCE.slice(SOURCE.indexOf("export function useFenceReached"));
  for (const guard of ["if (!enabled || latched || !streaming) return;", "if (reached) return;"]) {
    assert.ok(hook.includes(guard), `expected the early return ${guard}`);
  }
  assert.ok(
    /const reached = !enabled \|\|/.test(hook),
    "the disabled path must short-circuit to reached, so every effect below takes its early return",
  );
});

test("the observer disconnects itself on the upgrade", () => {
  const callback = SOURCE.slice(
    SOURCE.indexOf("new IntersectionObserver"),
    SOURCE.indexOf("for (const observer of observers) observer.observe(node)"),
  );
  assert.ok(
    callback.indexOf("each.disconnect()") < callback.indexOf("setLatched(true)"),
    "every observer must disconnect before the state write, so an upgraded fence carries no " +
      "residual per-scroll cost",
  );
});

test("a nested scroller is gated by the outermost one as well", () => {
  // An explicit root is clipped by the ancestors BETWEEN the target and the root and by nothing
  // above it, so rooting at the reasoning pane asks only "is this fence inside the pane's window".
  // Two ways that upgrades fences nobody can see: a pane scrolled out of the thread still reports
  // the fences inside its 256 px window as intersecting, and `reasoning.tsx` drops `max-h-64` at
  // the end of a stream while KEEPING `overflow-y-auto`, so the pane stops being scrollable, its
  // box becomes the whole trace, and an observer still rooted at it reports every fence in that
  // trace at once.
  //
  // The outermost scroller answers the question the inner root cannot, and it cannot go stale the
  // same way: a pane below it ceasing to scroll does not change which element is outermost.
  const walk = SOURCE.slice(SOURCE.indexOf("const outermostScrollerOf"));
  assert.ok(
    walk.slice(0, 260).includes("found = el") && !walk.slice(0, 260).includes("return el;"),
    "outermostScrollerOf must keep walking rather than returning the first match",
  );
  assert.ok(
    SOURCE.includes("? [[node, near]]")
      && SOURCE.includes("[[node, near], [near as HTMLElement, outer]]"),
    "one gate when the two scrollers agree; otherwise the FENCE against the nearest and the "
      + "PANE against the outermost",
  );
  assert.ok(
    SOURCE.includes("if (!seen.every(Boolean)) return;"),
    "the latch must need EVERY gate, not any of them",
  );
  assert.ok(
    /inBand\(node, near\) && \(near === outer \|\| inBand\(near as HTMLElement, outer\)\)/
      .test(SOURCE),
    "the pre-paint door must ask the same two questions of the same two elements",
  );
  assert.ok(
    !/\[\[node, near\], \[node, outer\]\]/.test(SOURCE),
    "watching the FENCE through the outer root clips it at the pane and cancels the lookahead "
      + "it was rooted at the pane to get: measured 2 of 10 against 4 with the pane in view",
  );

  // The rebind. The conjunction alone is not enough: scroll an expanded pane partly on screen and
  // the outer gate is true, so a stale inner root decides alone and reports the whole trace,
  // measured at 10 of 10 on a 4,080 px trace against a 900 px viewport where the right answer is
  // about 3, and 4 of 10 once the inner root is re-resolved. Both engines.
  assert.ok(
    /resize = new ResizeObserver\(\(\) => \{\s*if \(!isScrollable\(near\)\) setGeneration/
      .test(SOURCE),
    "the gates must be rebuilt when the nested scroller stops being one",
  );
  assert.ok(
    /\}, \[reached, host, generation\]\);/.test(SOURCE),
    "and the rebind has to be a dependency of the effect that builds them",
  );
  assert.ok(
    /if \(near !== null && near !== outer && typeof ResizeObserver !== "undefined"\)/.test(SOURCE),
    "watched only for fences that actually have a nested scroller, and only one element",
  );
  assert.ok(
    !/setGeneration\(0\)|setLatched\(false\)/.test(SOURCE),
    "the rebind must stay one-way: it can withhold a latch, never clear one",
  );

  // The conjunction, run rather than described: a fence inside a pane's window while the pane is
  // far outside the thread viewport must NOT be reached.
  const band = (rect: {top: number; bottom: number}, root: {top: number; height: number}) =>
    rect.bottom > root.top - root.height && rect.top < root.top + root.height * 2;
  const pane = { top: 4000, height: 256, bottom: 4256 };
  const viewport = { top: 0, height: 800 };
  const fence = { top: 4100, bottom: 4200 };
  assert.equal(band(fence, pane), true, "inside the pane's own window");
  assert.equal(band(pane, viewport), false, "but the pane is nowhere the reader can see");

  // And the lookahead survives when the pane IS in view: the outer gate asks about the pane, so
  // it cannot clip the fence a second time.
  const onScreen = { top: 100, height: 256, bottom: 356 };
  const ahead = { top: 500, bottom: 620 };
  assert.equal(band(onScreen, viewport), true, "the pane is on screen");
  assert.equal(band(ahead, onScreen), true, "so a fence one window below it still pre-warms");
});

test("the mode is decided in one place, and `off` still means the pre-default behaviour", () => {
  // The table itself is RUN row by row in `tests/code-fence-mode.test.ts`. What this file pins is
  // that this module grows no second opinion, and that `off` still switches the whole hook out.
  assert.ok(
    SOURCE.includes('export { type FenceMode, resolveFenceMode, SHIP_DEFAULT } from "./code-fence-mode";'),
    "the mode module is the single source of the decision",
  );
  assert.ok(
    !/raw === "defer"|SHIP_DEFAULT: FenceMode|const raw =/.test(SOURCE),
    "no copy of the decision table may live here as well",
  );
  assert.ok(
    /useFenceReached\(\s*host,\s*mode !== "off" && !plainCode,\s*Boolean\(isIncomplete\),/.test(
      MARKDOWN_TEXT,
    ),
    "with the mode off every fence must render immediately, exactly as it did before the default " +
      "moved. `plainCode` is the ONE other thing allowed in this condition, and it is not a second " +
      "opinion about the mode: a plain subtree is never highlighted at all, so there is nothing " +
      "for the latch to wait for. Every fence outside one still reads the mode alone.",
  );
  // A plain subtree renders the shell whatever the mode says, so the render gate has to carry the
  // same condition. Pinned because a `plainCode` that reached only the hook would leave a
  // reasoning fence tokenized and then thrown away, which is the cost this exists to remove.
  assert.ok(
    /\{reached && !plainCode \? \(/.test(MARKDOWN_TEXT),
    "the render gate must honour the plain subtree as well as the latch",
  );
});

test("a streaming fence never defers", () => {
  assert.ok(
    MARKDOWN_TEXT.includes("Boolean(isIncomplete)"),
    "an incomplete (streaming) fence must be immediate: deferring it would change what " +
      "streaming renders rather than what a settled thread costs",
  );
});

test("the shell carries the same streamdown hooks the real block does", () => {
  for (const attribute of [
    'data-streamdown="code-block"',
    'data-streamdown="code-block-header"',
    'data-streamdown="code-block-body"',
  ]) {
    assert.ok(
      SOURCE.includes(attribute),
      `the shell must carry ${attribute} or the stylesheet rules that size a code block do ` +
        "not apply to it and the two arms lay out differently",
    );
  }
});

test("the shell trims trailing newlines the way streamdown does", () => {
  const trim = (text: string): string => {
    let end = text.length;
    while (end > 0 && text[end - 1] === "\n") end -= 1;
    return text.slice(0, end);
  };
  assert.equal(trim("a\nb\n\n\n"), "a\nb");
  assert.equal(trim("a\nb"), "a\nb");
  assert.equal(trim("\n\n"), "");
  assert.ok(
    SOURCE.includes("trimTrailingNewlines"),
    "an untrimmed shell is one blank line taller than the block it stands in for",
  );

  // ...and an EMPTY one is one line SHORTER, in the other direction. Streamdown special-cases the
  // empty token line, from its own renderer:
  //
  //   children: c.length === 0 || (c.length === 1 && c[0].content === "") ? `\n` : c.map(...)
  //
  // so a fence whose body is empty, or nothing but newlines, is one line box tall. A <code> with
  // an empty text node has no line box, so the fence would grow by a line on upgrade and move
  // everything below it.
  const body = (source: string): string => (trim(source) === "" ? "\n" : trim(source));
  assert.equal(body(""), "\n");
  assert.equal(body("\n\n\n"), "\n");
  assert.equal(body("x"), "x");
  assert.ok(
    /const trimmed = trimTrailingNewlines\(source\);\s*return trimmed === "" \? "\\n" : trimmed;/
      .test(SOURCE),
    "the shell must reproduce streamdown's empty line rather than collapse to no line at all",
  );
  assert.ok(
    !/<code>\{trimTrailingNewlines\(source\)\}<\/code>/.test(SOURCE),
    "the raw trim must not be rendered directly; it loses the empty-line case",
  );
});

test("the gate does not mount a wrapper element of its own", () => {
  assert.ok(
    !SOURCE.includes("<div ref={host}>"),
    "an extra div between a list item and its code block breaks the direct-child selector in " +
      "index.css and pushes the block a level deeper than the :last-child margin chain walks",
  );
  assert.ok(
    MARKDOWN_TEXT.includes('<div className="relative isolate" ref={host}>'),
    "the intersection target must be the wrapper markdown-text already rendered",
  );
});

test("the tokenize arm is measurement only and is not reachable from a boolean flag", () => {
  // The selection rule, and every shape that must NOT reach it, is exercised in
  // `tests/code-fence-mode.test.ts`. Here: no route into the arm except that resolved mode.
  assert.ok(
    !/"tokenize"/.test(SOURCE),
    "this module must not name the measurement arm at all; it only consumes a resolved mode",
  );
  assert.ok(
    MARKDOWN_TEXT.includes('const pretokenize = mode === "tokenize" && !reached'),
    "pretokenizing must be confined to the tokenize arm",
  );
});

test("a print upgrades the whole document, and never puts it back", () => {
  // An earlier `beforeprint` path was removed after 53 of 56 blocks still printed on streamdown's
  // raw fallback out to twenty seconds. The latch was not the problem: what it renders is, since
  // the highlighted body asks for tokens from a PASSIVE effect and the plugin answers `null` while
  // a grammar loads. `latchNow` closes both halves, warming then flushing twice. Keep both.
  for (const door of ["beforeprint", 'matchMedia?.("print")']) {
    assert.ok(
      SOURCE.includes(door),
      `${door} is one of the two ways a document reaches a printer, and both must be covered`,
    );
  }
  assert.ok(
    !/addEventListener\(\s*"afterprint"/.test(SOURCE),
    "reverting on afterprint would be exactly the bidirectional edge this design removes",
  );
  // A PRINT IS NOT A SESSION-WIDE SWITCH. As a module-global `printed` folded into every future
  // fence's `reached` it measured, at the 100K rung: print once, navigate away in-app and back,
  // and the thread remounts with 0 of 56 fences deferred, 41,410 spans and 61,747 elements instead
  // of 53, 2,458 and 22,794. One Ctrl+P turned the default off for the rest of the tab.
  assert.ok(
    /const reached = !enabled \|\| !CAN_OBSERVE \|\| streaming \|\| latched;/.test(SOURCE),
    "no print state may be folded into a fence's reached: a fence mounted after a print was not " +
      "on the printed page and has nothing to latch for",
  );
  assert.ok(
    /const upgradeEverythingForPrint = \(\): void => \{\s*latchNow\(\[\.\.\.unreached\]\);\s*\};/
      .test(SOURCE),
    "a print latches exactly what is unreached when it happens, and every print does it again",
  );
});

test("an upgrade taken inside one task warms, flushes, and flushes again", () => {
  // Dropping any one of the three puts a plain frame back on a jump, or a colourless fence on a
  // printed page.
  const latchNow = SOURCE.slice(SOURCE.indexOf("const latchNow"));
  const body = latchNow.slice(0, latchNow.indexOf("\n};"));
  assert.ok(body.includes("gate.warm(true)"), "the tokens have to exist before the swap renders");
  assert.ok(
    body.indexOf("gate.warm(true)") < body.indexOf("flushSync"),
    "warming after the flush is warming after the paint",
  );
  assert.equal(
    body.split("flushSync").length - 1,
    3,
    "an outer flush holds the update priority discrete; one inner flush commits the swap and the "
      + "second runs the passive effect that colours it",
  );
  assert.ok(
    body.includes("gate.poke()"),
    "react only runs pending passive effects when it has sync work, so the second flush needs some",
  );
});

test("a jump is recognised from the lookahead, not from a tuned number", () => {
  // `REACH_MARGIN` grows the band by one root height, so a scroll of at most one height can only
  // reveal fences already reached: the pass runs exactly when the movement beat the lookahead. A
  // literal pixel threshold would be a number nobody could derive or maintain.
  assert.ok(
    /Math\.abs\(top - before\) <= height/.test(SOURCE),
    "the jump test compares the movement against the root height the margin is one of",
  );
  assert.ok(
    !/[^a-zA-Z_]\d{2,}\s*(?:px)?\s*[;)]/.test(SOURCE.slice(SOURCE.indexOf("const onScroll"), SOURCE.indexOf("const watchScrolling"))),
    "no pixel constant may appear in the jump test",
  );
});

test("nothing is watched once there is nothing left to defer", () => {
  // This change claims a reached fence carries no residual per-scroll cost. The one shared
  // capturing listener must therefore be removed when the last fence latches.
  assert.ok(
    /document\.addEventListener\("scroll", onScroll, \{ capture: true, passive: true \}\)/.test(SOURCE),
    "one capturing, passive listener sees scrolling on nested panes as well as on the thread",
  );
  assert.ok(
    /unreached\.size > 0/.test(SOURCE) &&
      /document\.removeEventListener\("scroll", onScroll/.test(SOURCE),
    "the listener is removed when the register empties",
  );
});

const CODE_PLUGIN = readFileSync(
  new URL("../src/components/assistant-ui/code-plugin.ts", import.meta.url),
  "utf8",
);

test("the fence language is a language, not the whole info string", () => {
  // `getCodeFence` captures everything after the backticks, so ```python startLine=10 arrives as
  // "python startLine=10". Markdown treats everything past the first word as metadata and
  // Streamdown highlights the block as `python`. Passing the raw string through would label the
  // deferred shell with the metadata attached, and would hand the measurement arm a language no
  // grammar matches -- so it would tokenize as plain text and silently stop measuring the
  // tokenizer work it exists to measure.
  assert.ok(
    /const languageToken = language\?\.trim\(\)\.split\(\/\\s\+\/\)\[0\] \|\| null;/
      .test(MARKDOWN_TEXT),
    "the info string must be split before it is used as a language",
  );
  for (const use of [
    "language: (languageToken ?? \"text\") as never",
    "<DeferredFenceShell language={languageToken}",
  ]) {
    assert.ok(
      MARKDOWN_TEXT.includes(use),
      `both the shell and the measurement arm must use the parsed token: ${use}`,
    );
  }
  assert.ok(
    !/language: \(language \?\? "text"\)/.test(MARKDOWN_TEXT),
    "no path may pass the unparsed info string to the highlighter",
  );

  // The parse itself, run rather than described.
  const token = (info: string | null) => info?.trim().split(/\s+/)[0] || null;
  assert.equal(token("python startLine=10"), "python");
  assert.equal(token("  ts  "), "ts");
  assert.equal(token(""), null);
  assert.equal(token(null), null);
});

test("token coalescing was measured at zero and is not carried as code", () => {
  // Shiki already emits maximally coalesced tokens: 72,550 -> 72,550 over the 100K rung's 99 real
  // fences, in every theme mode. An implementation that removes no spans cannot make anything
  // faster, and carrying a runtime-flippable flag through the fence cache for it only creates
  // ways for a cached result to disagree with the flag that produced it.
  for (const gone of ["coalesceTokens", "coalesceLine", "mergeable", "__UNSLOTH_COALESCE_TOKENS__",
                      "VITE_UNSLOTH_COALESCE_TOKENS"]) {
    assert.ok(
      !CODE_PLUGIN.includes(gone),
      `${gone} was removed after measuring 0.0%; re-adding it needs a number first`,
    );
  }
  assert.ok(
    CODE_PLUGIN.includes("537013 -> merged 537013"),
    "the null belongs in the file it was measured on, so nobody repeats it",
  );
  assert.ok(
    CODE_PLUGIN.includes("scripts/coal-span-census.mjs"),
    "and it must name a reproducer, so the number can be checked rather than trusted",
  );
  // The reproducer has to BE here. The first version of that comment pointed at a script that
  // only existed on the machine the census was run on, which makes the citation worth nothing.
  assert.ok(
    existsSync(new URL("../scripts/coal-span-census.mjs", import.meta.url)),
    "the cited reproducer must exist in this repository",
  );
});

/**
 * THE PLAIN SUBTREE IS THE REASONING PANE AND NOTHING ELSE.
 *
 * Rendering a fence without colours is a deliberate trade, taken for the one place a reader skims
 * a wall of code rather than reads it. It is only defensible while it is contained: a reply body
 * that lost its highlighting would be a rendering regression smuggled in by a performance change,
 * which is the one thing this file exists to catch. Source-level, for the reason at the top of
 * this file -- the containment is structural, and a DOM test could only sample one thread.
 */
test("only the reasoning pane renders its fences plain", () => {
  const REASONING = readFileSync(
    new URL("../src/components/assistant-ui/reasoning.tsx", import.meta.url),
    "utf8",
  );
  assert.ok(
    /<MarkdownCodeHighlightingContext\.Provider value="plain">/.test(REASONING),
    "the reasoning pane is what asks for plain fences",
  );
  // The DEFAULT is the highlighted one, so anything not wrapped -- every reply body -- is
  // untouched by this change. React resolves a context read against the closest provider ABOVE
  // it, so a sibling subtree cannot see this one.
  assert.ok(
    /createContext<MarkdownCodeHighlighting>\("syntax"\)/.test(MARKDOWN_TEXT),
    "an unwrapped subtree must keep its syntax highlighting",
  );
  // And there is exactly one provider in the app. A second one is how "reasoning only" quietly
  // becomes "reasoning and whatever else somebody wrapped".
  const providers: string[] = [];
  const walk = (dir: URL) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const child = new URL(`${entry.name}${entry.isDirectory() ? "/" : ""}`, dir);
      if (entry.isDirectory()) walk(child);
      else if (/\.tsx?$/.test(entry.name)) {
        const text = readFileSync(child, "utf8");
        if (text.includes("MarkdownCodeHighlightingContext.Provider")) {
          providers.push(entry.name);
        }
      }
    }
  };
  walk(new URL("../src/", import.meta.url));
  assert.deepEqual(
    providers,
    ["reasoning.tsx"],
    "the plain subtree must stay the reasoning pane alone",
  );
});

test("the idle pre-warm drives the tokenizer over real text, not an empty string", () => {
  /*
   * Loading a grammar is cheap; running it over text the first time is not, and `""` never does
   * the second. With deferral on the whole one-off cost therefore landed in one frame on the fence
   * the reader scrolled to: 1200 and 1085 ms at the 100K rung on WebKitGTK, against 183 and 190 ms
   * on real text. Asserted at the source because the invariant is structural, and the alternative
   * is a benchmark noticing it two days later, which is how it was found.
   */
  const warm = SOURCE.slice(SOURCE.indexOf("const warmGrammars"));
  const body = warm.slice(0, warm.indexOf("\n};"));
  assert.ok(
    body.includes("gate.warm(true)"),
    "warming on an empty string leaves the first real tokenization to happen during a scroll",
  );
  // The only surviving `gate.warm(false)` is the eager grammar-load pass, which runs BEFORE the
  // real warm and is deliberately not gated on size. Nothing may reach `warm(false)` afterwards.
  assert.match(
    body,
    /grammarsLoaded\.add\(language\);\s*gate\.warm\(false\);[\s\S]*gate\.warm\(true\)/,
    "an unconditional false warm in place of the real one is the regression this test catches",
  );
  assert.equal(
    (body.match(/gate\.warm\(false\)/g) ?? []).length,
    1,
    "one false warm, in the load pass; a second one means a language can be marked warmed on nothing",
  );
  // Anti-vacuity: renamed or restructured, the checks above would pass on an empty slice.
  assert.ok(body.length > 60 && body.includes("grammarsWarmed"), "found the real warmGrammars body");
});

test("a speculative warm is capped, and the cap is the shared one", () => {
  /*
   * The chat renderer never applies MAX_HIGHLIGHT_CHARS: `markdown-text.tsx` supplies the code
   * plugin unconditionally and `FenceBlock` warms the whole body, so a real-text warm would
   * tokenize an arbitrarily large off-screen fence, and `code-plugin.ts`'s `evict` keeps the last
   * fence whatever its size. The latch is demanded work and stays uncapped; this half is
   * speculative, so it is bounded.
   */
  assert.match(
    SOURCE,
    /import \{ MAX_HIGHLIGHT_CHARS \} from "@\/lib\/markdown-plugins";/,
    "the cap must be the shared constant, not a second copy that can drift",
  );
  assert.ok(
    !/const MAX_HIGHLIGHT_CHARS\s*=/.test(SOURCE),
    "a local redefinition would let this cap drift away from the one every other reader uses",
  );
  assert.ok(
    /gate\.chars === 0 \|\| gate\.chars > MAX_HIGHLIGHT_CHARS/.test(SOURCE),
    "the warm must consult the fence's size before tokenizing it",
  );
  // An EMPTY fence trims to "", so warming it teaches the grammar nothing and would still mark the
  // language done, leaving every later fence in it to tokenize on the scroll path.
  assert.match(
    SOURCE,
    /if \(gate\.chars === 0 \|\| gate\.chars > MAX_HIGHLIGHT_CHARS\) continue;/,
    "both cases must `continue`, so the language is left unwarmed for a fence that can warm it",
  );
  // The other half: the latch must NOT have grown a cap.
  const latch = SOURCE.slice(SOURCE.indexOf("const latchNow"));
  assert.ok(
    !latch.slice(0, latch.indexOf("\n};")).includes("MAX_HIGHLIGHT_CHARS"),
    "a fence the reader has actually reached is highlighted whatever its size",
  );
  assert.match(
    MARKDOWN_TEXT,
    /useFenceReached\([\s\S]{0,200}?trimmedLength\(source\),/,
    "the hook can only cap what the caller tells it about, and `warm` tokenizes the TRIMMED body",
  );
});

test("the idle warm yields between languages", () => {
  /*
   * requestIdleCallback only controls when a callback STARTS, and WebKitGTK has none at all, so
   * this venue takes the setTimeout fallback and cannot even do that. A warm tokenizes
   * synchronously once its grammar is loaded, so every language in one callback is one unyieldable
   * block: 746 ms for the 100K rung's five languages driven through shiki, worst single 334 ms.
   *
   * The LOADS are the other half and must NOT be yielded: 500 ms x N on that fallback would leave
   * a jump or a print inside the window with an unloaded grammar, which is the plain-fallback
   * frame this whole pre-warm exists to prevent.
   */
  const warm = SOURCE.slice(SOURCE.indexOf("const warmGrammars"));
  const body = warm.slice(0, warm.indexOf("\n};"));
  assert.match(
    body,
    /gate\.warm\(true\);[\s\S]*scheduleGrammarWarm\(\);[\s\S]*return;/,
    "one tokenization per task: warm, re-schedule, and leave the rest to the next idle slot",
  );
  // Everything before the second loop, which is the one that tokenizes.
  const loadPass = body.slice(0, body.indexOf("grammarsWarmed.has"));
  assert.ok(
    !loadPass.includes("scheduleGrammarWarm") && !loadPass.includes("return;"),
    "the grammar loads all start in the first pass; yielding them costs the jump and the print",
  );
  assert.ok(
    !loadPass.includes("MAX_HIGHLIGHT_CHARS") && !loadPass.includes("gate.chars"),
    "a load ignores size: it tokenizes nothing, and an over-cap language still needs its grammar",
  );
  assert.ok(
    body.includes("grammarsWarmed.add(language)"),
    "the chain terminates only because each task marks one more language done",
  );
});

test("the warm dedupes on the grammar, not on the spelling", async () => {
  /*
   * `grammarsWarmed` keyed the raw fence tag while `code.highlight` lower-cases and resolves
   * aliases, so a thread mixing ```py and ```python warmed one grammar twice -- and after this PR
   * each spelling is a real tokenization of a different fence, not the old empty-string cache hit.
   */
  const { normalizeLanguage } = await import(
    "../src/components/assistant-ui/code-plugin.ts"
  );
  // Run the identity rather than describe it: aliases, overrides and case all collapse.
  for (const [tag, canonical] of [["py", "python"], ["Python", "python"], ["JS", "javascript"],
                                  ["c++", "cpp"], ["bash", "shellscript"], ["text", "text"]]) {
    assert.equal(normalizeLanguage(tag), canonical, tag);
  }
  assert.match(
    SOURCE,
    /const grammarOf = \(gate: FenceGate\): string =>\s*normalizeLanguage\(gate\.language \?\? "text"\);/,
    "the warm sets must be keyed by the same identity the highlighter uses",
  );
  assert.ok(
    CODE_PLUGIN.includes("export const normalizeLanguage"),
    "one definition, exported, so the two keyings cannot drift apart",
  );
});
