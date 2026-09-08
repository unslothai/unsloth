// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * WHETHER MATHS-BEARING BLOCKS TAKE CONTAINMENT, decided in one pure function.
 *
 * Kept out of any `.tsx` and free of `import.meta`, exactly as `code-fence-mode.ts` is, because
 * the frontend's tests run under `node --experimental-strip-types` and can neither load JSX nor
 * evaluate `import.meta.env`. Every row below is RUN by `tests/math-block-mode.test.ts` rather
 * than checked by regexes over the source.
 *
 *   "off"      The marker class is still emitted, the stylesheet rule that reads it is not armed,
 *              and nothing about rendering changes.
 *   "contain"  SHIP DEFAULT, subject to the engine gate below. `content-visibility: auto` applies
 *              to maths-bearing blocks, so off-screen maths generates no boxes and no RenderLayers
 *              until it is scrolled to. See the comment on `SHIP_DEFAULT` for what that buys and
 *              for the two costs that are accepted rather than solved.
 *
 * TWO STATES, NOT THREE. `code-fence-mode.ts` distinguishes an unset flag from a mistyped one; this
 * file does not, and the reason has CHANGED now that the default is "contain". It used to be that
 * unset and unrecognised both resolved to "off" and there was no distinction to draw. They no
 * longer land in the same place: unset resolves to the ship default, which is ON, while an
 * unrecognised value resolves to "off". That asymmetry is deliberate and is the safe direction. An
 * operator who mistypes `VITE_UNSLOTH_MATH_BLOCK_CONTAINMENT=conatin` was reaching for the flag,
 * and the only reason to reach for a flag that is already on is to turn it off, so a typo turning
 * it off does what they were trying to do. The opposite rule, resolving a typo to the default,
 * would ignore them.
 */
export type MathBlockMode = "off" | "contain";

/*
 * SHIP DEFAULT, and the evidence for it.
 *
 * WHAT IT BUYS. At the 500K rung on a real GPU (AMD gfx1151, production bundle, two repetitions per
 * rung, each arm scored against its own two neighbouring baselines, runs 32902943628 and
 * 32906688232) a one pixel scroll costs 285 ms of blocked main thread and the interface runs at
 * 3.2 fps. With this on it costs 18 to 19.5 ms and runs at 35.6 to 38.2 fps: +92% on the mean and
 * +88% on the median rAF gap, in all four sessions. At 100K it is +80 to +85% at 62 fps. At 0K the
 * selector matches nothing, so the rule cannot cost anything there for a structural reason rather
 * than a statistical one.
 *
 * RENDERING IS UNCHANGED. Two production bundles, feature off against on, photographed at seven
 * maths blocks after one full walk of the thread: 10 differing pixels over seven frames, four of
 * them bit-identical, `scrollHeight` matching exactly at 307,915.
 *
 * THE THREE THINGS THAT COULD HAVE STOPPED IT, each measured rather than argued.
 *
 *   1. FIND-IN-PAGE. Handled by `gateOnEngine` below, not by this constant: an engine that cannot
 *      find skipped content does not get containment at all. Verified once by driving the real
 *      `WebKitFindController` over four states on WebKitGTK 2.50.4 (plain 1 match, contained 1,
 *      `visibility: hidden` 0, absent 0; the two controls are what make the middle row mean
 *      something). Studio Desktop wires no find-in-page on any platform, so the exposed surface is
 *      the web UI in a browser older than Safari 26, which is exactly what the gate switches off.
 *   2. LIST NUMBERING. Fixed by making `li`, `ol` and `ul` uncontainable; see
 *      `math-block-marker.ts`. Censused at zero cost: 595 of 595 marked blocks in the 500K corpus
 *      are paragraphs.
 *   3. A REMEMBERED HEIGHT GOING STALE ACROSS A RESIZE. Accepted, at a measured size. Narrowing the
 *      window from 1440 to 1008 px on the 500K corpus leaves `scrollHeight` 6,350 px short of the
 *      truth, 2.0%, against 0.0% for the same thread with the feature off. It converges as blocks
 *      are scrolled past. A synthetic fixture in which EVERY block is contained reads 100%, which
 *      is an upper bound and not what a real thread does.
 *
 * WHAT IS ACCEPTED RATHER THAN SOLVED. The 2.0% above, and clipping: paint containment clips an
 * inline formula wider than the chat column instead of letting it overflow, confirmed by hit
 * testing. Zero blocks overflow in the 500K corpus. `overflow-x: auto` is NOT the remedy, because
 * it would make 595 marked blocks into 595 scroll containers, which is the RenderLayer population
 * this whole change exists to remove.
 *
 * Moving this line is the whole of "turn block containment off again".
 */
export const SHIP_DEFAULT: MathBlockMode = "contain";

/*
 * THE FIND-IN-PAGE GATE, and why it is a proxy rather than a direct test.
 *
 * WebKit below Safari 26 cannot find SKIPPED `content-visibility` content with native find-in-page
 * (webkit.org/b/283846). `index.css` already refuses to use `content-visibility: auto` on code
 * blocks for exactly this reason and says so in a comment, noting that Studio has no in-thread
 * search to fall back on. Containment on maths-bearing blocks reintroduces the same hazard, and it
 * reaches further: the marked element is a whole paragraph or heading, so ordinary PROSE in a
 * maths-bearing block would stop being findable too, not just the formula.
 *
 * Nothing on the platform exposes "can find-in-page reach skipped content", so this gates on a CSS
 * feature that shipped in the SAME release as the fix and is therefore absent on exactly the builds
 * that are affected. Anchor positioning is that feature. This is a PROXY: it asserts a release
 * train, not the bug, and if some engine ever ships one without the other this gate is wrong in
 * whichever direction that engine chose. That is a deliberate trade against the alternative, which
 * is parsing a user-agent string that WebKitGTK freezes at `AppleWebKit/605.1.15` on every version
 * it has ever shipped.
 *
 * Measured on the venue the performance numbers came from, WebKitGTK 2.50.4: `anchor-name` is
 * supported, and that build does find skipped content. The point of the gate is the builds that are
 * not that one, which Studio absolutely ships against, since Linux uses whatever WebKitGTK the host
 * provides and macOS uses whatever the OS provides.
 */
export const FIND_IN_PAGE_PROBE = "anchor-name: --unsloth-probe";

/**
 * Whether the engine may take containment at all, given the outcome of the probe above.
 *
 * An EXPLICIT RUNTIME override wins, because that global exists so a measurement or a bug report
 * can force an arm from the devtools console, and a gate that silently refused would make the
 * console flip look like it had worked while measuring the other arm. A build flag does NOT win:
 * a build is shipped to machines whose engines the builder cannot see.
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
 * `html[data-panel-resizing]` precedent already in `index.css`. An attribute rather than a class
 * on the thread root because it has to be reachable before any thread has mounted, and because a
 * measurement can flip it without provoking a React render, which keeps the DOM identical between
 * a measured window with the feature on and one with it off.
 */
export const MATH_BLOCK_CONTAINMENT_ATTRIBUTE = "data-math-block-containment";
export const MATH_BLOCK_CONTAINMENT_ON = "on";

/*
 * REAPPLY WHEN THE CONSOLE FLIPS THE GLOBAL.
 *
 * `applyMathBlockContainment()` is called once, before the first render. Without this, a tester who
 * set `__UNSLOTH_MATH_BLOCK_CONTAINMENT__` from devtools AFTER load changed nothing: the attribute
 * kept its old value and the session went on measuring the arm it was already in, silently. That is
 * the worst failure mode an escape hatch can have, because the number it produces looks like an
 * answer.
 *
 * The global is redefined as an accessor so that ASSIGNING it reapplies. The value is held in a
 * closure rather than on the scope object, so reading the property still returns what was written
 * and nothing else on the page sees a different shape than before.
 *
 * It lives HERE, in the module with no `import.meta` and no `document`, for the reason this file's
 * header gives: the frontend's `node --experimental-strip-types` runner cannot load the other one,
 * and an escape hatch whose failure mode is silence is exactly the thing that needs rows run
 * against it rather than a comment.
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
