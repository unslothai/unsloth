// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * Whether maths-bearing blocks take containment, decided in one pure function.
 * Kept out of any `.tsx` and free of `import.meta`, as `code-fence-mode.ts` is: the frontend tests
 * run under `node --experimental-strip-types` and can neither load JSX nor evaluate
 * `import.meta.env`. Every row below is RUN by `tests/math-block-mode.test.ts`.
 *   "off"      the marker class is still emitted, the stylesheet rule that reads it is not armed.
 *   "contain"  ship default, subject to the engine gate below: `content-visibility: auto` applies
 *              to maths-bearing blocks, so off-screen maths generates no boxes and no RenderLayers
 *              until it is scrolled to.
 * Two states, not three: unset resolves to the ship default, which is ON, while an UNRECOGNISED
 * value resolves to "off". The asymmetry is deliberate and is the safe direction. An operator who
 * mistypes the flag was reaching for it, and the only reason to reach for a flag already on is to
 * turn it off, so a typo turning it off does what they were trying to do.
 */
export type MathBlockMode = "off" | "contain";

/*
 * Ship default, and the evidence for it.
 * WHAT IT BUYS. At the 500K rung on a real GPU (AMD gfx1151, production bundle, two repetitions
 * per rung, runs 32902943628 and 32906688232) a one pixel scroll costs 285 ms of blocked main
 * thread and the interface runs at 3.2 fps; with this on it costs 18 to 19.5 ms at 35.6 to 38.2
 * fps, +92% on the mean rAF gap in all four sessions. At 100K it is +80 to +85% at 62 fps. At 0K
 * the selector matches nothing, so the rule cannot cost anything there for a structural reason
 * rather than a statistical one. RENDERING IS UNCHANGED: two production bundles, off against on,
 * differ by 10 pixels over seven frames with `scrollHeight` matching exactly at 307,915.
 * The three things that could have stopped it, each measured rather than argued. FIND-IN-PAGE is
 * handled by `gateOnEngine` below, not by this constant: an engine that cannot find skipped
 * content does not get containment at all, verified by driving the real `WebKitFindController` on
 * WebKitGTK 2.50.4. LIST NUMBERING is fixed by making `li`, `ol` and `ul` uncontainable; see
 * `math-block-marker.ts`. A REMEMBERED HEIGHT GOING STALE ACROSS A RESIZE is accepted at a
 * measured size: narrowing 1440 to 1008 px on the 500K corpus leaves `scrollHeight` 2.0% short,
 * converging as blocks are scrolled past.
 * Also accepted rather than solved: paint containment CLIPS an inline formula wider than the chat
 * column instead of letting it overflow, though zero blocks overflow in the 500K corpus.
 * `overflow-x: auto` is NOT the remedy, because it would make 595 marked blocks into 595 scroll
 * containers, which is the RenderLayer population this whole change exists to remove.
 * Moving this line is the whole of "turn block containment off again".
 */
export const SHIP_DEFAULT: MathBlockMode = "contain";

/*
 * The find-in-page gate, and why it is a proxy rather than a direct test.
 * WebKit below Safari 26 cannot find SKIPPED `content-visibility` content with native find-in-page
 * (webkit.org/b/283846); `index.css` already refuses `content-visibility: auto` on code blocks for
 * exactly this reason. Here the marked element is a whole paragraph or heading, so ordinary PROSE
 * in a maths-bearing block would stop being findable too, not just the formula.
 * Nothing on the platform exposes "can find-in-page reach skipped content", so this gates on
 * anchor positioning, a CSS feature that shipped in the SAME release as the fix and is therefore
 * absent on exactly the builds that are affected. It asserts a release train, not the bug, and if
 * some engine ever ships one without the other this gate is wrong in whichever direction that
 * engine chose. The alternative is parsing a user-agent string that WebKitGTK freezes at
 * `AppleWebKit/605.1.15` on every version it has ever shipped. Studio ships against whatever
 * WebKitGTK the host provides, which is the case the gate exists for.
 */
export const FIND_IN_PAGE_PROBE = "anchor-name: --unsloth-probe";

/**
 * Whether the engine may take containment at all, given the outcome of the probe above.
 *
 * An EXPLICIT RUNTIME override wins, because that global exists so a measurement or a bug report
 * can force an arm from the devtools console, and a gate that silently refused would make the
 * console flip look like it had worked while measuring the other arm. A build flag does NOT win: a
 * build is shipped to machines whose engines the builder cannot see.
 */
export const gateOnEngine = (
  mode: MathBlockMode,
  engineFindsSkippedContent: boolean,
  forcedByRuntime: boolean,
): MathBlockMode =>
  mode !== "contain" || engineFindsSkippedContent || forcedByRuntime ? mode : "off";

/** Whether `runtime` is an explicit instruction rather than an absent global. */
export const isRuntimeForced = (runtime: unknown): boolean =>
  runtime === true || runtime === "1" || runtime === "contain";

/**
 * @param runtime  `__UNSLOTH_MATH_BLOCK_CONTAINMENT__`: string, boolean or absent. The boolean is
 *                 the devtools-console form and has to work in BOTH directions, so that a session
 *                 can be flipped without a rebuild.
 * @param build    `VITE_UNSLOTH_MATH_BLOCK_CONTAINMENT`, `""` when never set.
 */
export const resolveMathBlockMode = (
  runtime: unknown,
  build: string,
): MathBlockMode => {
  const raw =
    typeof runtime === "string"
      ? runtime
      : runtime === true
        ? "contain"
        : runtime === false
          ? "off"
          : build;
  return raw === "1" || raw === "contain"
    ? "contain"
    : raw === ""
      ? SHIP_DEFAULT
      : "off";
};

/**
 * The attribute the stylesheet reads, on `document.documentElement`, following the
 * `html[data-panel-resizing]` precedent already in `index.css`. An attribute rather than a class on
 * the thread root because it has to be reachable before any thread has mounted, and because a
 * measurement can flip it without provoking a React render.
 */
export const MATH_BLOCK_CONTAINMENT_ATTRIBUTE = "data-math-block-containment";
export const MATH_BLOCK_CONTAINMENT_ON = "on";

/*
 * Reapply when the console flips the global.
 * `applyMathBlockContainment()` is called once, before the first render. Without this, a tester who
 * set `__UNSLOTH_MATH_BLOCK_CONTAINMENT__` from devtools AFTER load changed nothing: the session
 * went on measuring the arm it was already in, silently, which is the worst failure mode an escape
 * hatch can have, because the number it produces looks like an answer.
 * The global is redefined as an accessor so that ASSIGNING it reapplies, with the value held in a
 * closure so reading the property still returns what was written. It lives here, in the module with
 * no `import.meta` and no `document`, because the `node --experimental-strip-types` runner cannot
 * load the other one.
 */
export const installOverrideWatcher = (
  scope: Record<string, unknown>,
  apply: () => MathBlockMode,
): boolean => {
  try {
    let held = scope.__UNSLOTH_MATH_BLOCK_CONTAINMENT__;
    Object.defineProperty(scope, "__UNSLOTH_MATH_BLOCK_CONTAINMENT__", {
      configurable: true,
      enumerable: true,
      get: () => held,
      set: (next: unknown) => {
        held = next;
        apply();
      },
    });
    return true;
  } catch {
    // A frozen or otherwise hostile global is not worth failing startup over: this runs before the
    // first render, so throwing here is a white screen. The flag still works when set BEFORE load,
    // which is how the build flag and the measurement harness use it.
    return false;
  }
};
