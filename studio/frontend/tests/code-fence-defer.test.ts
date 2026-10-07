// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

import { markdownBlockFallback } from "../src/components/assistant-ui/markdown-block-fallback.ts";
import { readSrc } from "./helpers/kit.ts";

/** Pins that the reach latch never reverts; a two-way viewport gate measured slower. */

const SOURCE = readSrc("components/assistant-ui/code-fence-defer.tsx");

const MARKDOWN_TEXT = readSrc("components/assistant-ui/markdown-text.tsx");

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
  // `root: null` makes rootMargin useless, and matching known selectors skips the reasoning pane.
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
  // The pre-paint effect must re-run on the generation bump, or a shell is painted on screen.
  const prepaint = SOURCE.slice(
    SOURCE.indexOf("THE FIRST FRAME, which the observer cannot cover"),
    SOURCE.indexOf("// The one-way edge."),
  );
  assert.ok(prepaint.length > 200, "the pre-paint effect must still be findable by its comment");
  assert.ok(
    /\}, \[reached, host, generation\]\);/.test(prepaint),
    "the pre-paint gate must depend on the rebind generation, not just on reached and host",
  );
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
  // Rooting at an inner pane over-reports fences, so the observer roots at the outermost scroller.
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

  const band = (rect: {top: number; bottom: number}, root: {top: number; height: number}) =>
    rect.bottom > root.top - root.height && rect.top < root.top + root.height * 2;
  const pane = { top: 4000, height: 256, bottom: 4256 };
  const viewport = { top: 0, height: 800 };
  const fence = { top: 4100, bottom: 4200 };
  assert.equal(band(fence, pane), true, "inside the pane's own window");
  assert.equal(band(pane, viewport), false, "but the pane is nowhere the reader can see");

  const onScreen = { top: 100, height: 256, bottom: 356 };
  const ahead = { top: 500, bottom: 620 };
  assert.equal(band(onScreen, viewport), true, "the pane is on screen");
  assert.equal(band(ahead, onScreen), true, "so a fence one window below it still pre-warms");
});

test("the mode is decided in one place, and `off` still means the pre-default behaviour", () => {
  assert.ok(
    SOURCE.includes('export { type FenceMode, resolveFenceMode, SHIP_DEFAULT } from "./code-fence-mode";'),
    "the mode module is the single source of the decision",
  );
  assert.ok(
    !/raw === "defer"|SHIP_DEFAULT: FenceMode|const raw =/.test(SOURCE),
    "no copy of the decision table may live here as well",
  );
  assert.ok(
    /useFenceReached\(\s*host,\s*mode !== "off",\s*Boolean\(isIncomplete\),/.test(MARKDOWN_TEXT),
    "with the mode off every fence must render immediately, exactly as it did before the default " +
      "moved",
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

  // Streamdown renders an empty body as one line box, so an empty body maps to "\n".
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
  // latchNow must warm then flush twice, or printed fences fall back to raw output.
  for (const door of ["beforeprint", 'matchMedia?.("print")']) {
    assert.ok(
      SOURCE.includes(door),
      `${door} is one of the two ways a document reaches a printer, and both must be covered`,
    );
  }
  /* Latches never revert; an afterprint handler may clear only the print flag. */
  const afterPrint = SOURCE.match(/addEventListener\("afterprint", (.*?)\);/);
  if (afterPrint) {
    assert.equal(
      afterPrint[1],
      "() => setPrinting(false)",
      "an afterprint handler may do one thing: end the print. Anything else here is the " +
        "bidirectional edge this design removes",
    );
  }
  assert.ok(
    !/setLatched\(false\)|latched = false/.test(SOURCE),
    "nothing anywhere may hand a reached fence back its plain shell",
  );
  // Printing must not be a session-wide switch that disables deferral for later mounts.
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

const CODE_PLUGIN = readSrc("components/assistant-ui/code-plugin.ts");

test("mermaid detection walks the block with fence context", () => {
  assert.match(
    MARKDOWN_TEXT,
    /function findMermaidFence\(blockContent: string\): MermaidFence \{/,
    "one walk must answer both the opener and the source question",
  );
  assert.match(MARKDOWN_TEXT, /let enclosing: \{ char: string; run: number \} \| null = null;/);
  assert.match(MARKDOWN_TEXT, /enclosing = \{ char: marker\[0\], run: marker\.length \};/);
  assert.match(MARKDOWN_TEXT, /marker\.length >= enclosing\.run/);
  assert.ok(!MARKDOWN_TEXT.includes("MERMAID_INFO_RE"), "the context-free matcher is gone");
});

test("a settled alternative fence is not marked incomplete", () => {
  assert.match(
    MARKDOWN_TEXT,
    /<StreamingFenceBlock[\s\S]{0,160}isIncomplete=\{false\}/,
    "the settled branch must pass isIncomplete={false}",
  );
  assert.match(
    MARKDOWN_TEXT,
    /function StreamingFenceBlock\(\{[\s\S]{0,120}isIncomplete = true,/,
    "and the parameter must default to true for the streaming branch",
  );
});

test("a settled alternative fence is not marked incomplete", () => {
  assert.match(
    MARKDOWN_TEXT,
    /<StreamingFenceBlock[\s\S]{0,160}isIncomplete=\{false\}/,
    "the settled branch must pass isIncomplete={false}",
  );
  assert.match(
    MARKDOWN_TEXT,
    /function StreamingFenceBlock\(\{[\s\S]{0,120}isIncomplete = true,/,
    "and the parameter must default to true for the streaming branch",
  );
});

test("the fence language is a language, not the whole info string", () => {
  // Only the first word of the info string names the grammar; the rest is metadata.
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

  const token = (info: string | null) => info?.trim().split(/\s+/)[0] || null;
  assert.equal(token("python startLine=10"), "python");
  assert.equal(token("  ts  "), "ts");
  assert.equal(token(""), null);
  assert.equal(token(null), null);
});

test("token coalescing was measured at zero and is not carried as code", () => {
  // Shiki tokens are already maximally coalesced, so a coalescing flag would only add cache skew.
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
  assert.ok(
    existsSync(new URL("../scripts/coal-span-census.mjs", import.meta.url)),
    "the cited reproducer must exist in this repository",
  );
});

test("the idle pre-warm drives the tokenizer over real text, not an empty string", () => {
  /* A real-text warm is required; warming with "" leaves the first-run cost on scroll. */
  const warm = SOURCE.slice(SOURCE.indexOf("const warmGrammars"));
  const body = warm.slice(0, warm.indexOf("\n};"));
  assert.ok(
    body.includes("gate.warm(true)"),
    "warming on an empty string leaves the first real tokenization to happen during a scroll",
  );
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
  assert.ok(body.length > 60 && body.includes("grammarsWarmed"), "found the real warmGrammars body");
});

test("a speculative warm is capped, and the cap is the shared one", () => {
  /* The speculative warm is capped by MAX_HIGHLIGHT_CHARS; the demanded latch is not. */
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
  // An empty fence teaches the grammar nothing but would still mark the language warmed.
  assert.match(
    SOURCE,
    /if \(gate\.chars === 0 \|\| gate\.chars > MAX_HIGHLIGHT_CHARS\) continue;/,
    "both cases must `continue`, so the language is left unwarmed for a fence that can warm it",
  );
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
  /* Tokenizing warms must yield per language; grammar loads must not be yielded. */
  const warm = SOURCE.slice(SOURCE.indexOf("const warmGrammars"));
  const body = warm.slice(0, warm.indexOf("\n};"));
  assert.match(
    body,
    /gate\.warm\(true\);[\s\S]*scheduleGrammarWarm\(\);[\s\S]*return;/,
    "one tokenization per task: warm, re-schedule, and leave the rest to the next idle slot",
  );
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
  /* grammarsWarmed must key the normalized language, or aliases warm the same grammar twice. */
  const { normalizeLanguage } = await import(
    "../src/components/assistant-ui/code-plugin.ts"
  );
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
