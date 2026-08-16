// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * When the thread may let a code block skip its own rendering, and when it may not.
 *
 * Streamdown puts `content-visibility: auto` with `contain-intrinsic-size: auto 200px` inline on
 * every code-block wrapper, which lets the engine skip style, layout and paint for blocks that
 * are off screen. That is worth having on a long thread: measured over the same fixture, a
 * 300K-character thread pays for every code block on every scroll frame without it.
 *
 * It is only safe once the engine knows how tall the block really is. `contain-intrinsic-size`
 * with the `auto` keyword makes the engine record an element's rendered size and reuse it while
 * the element is skipped -- the LAST REMEMBERED SIZE of CSS Sizing 4 -- but an element that has
 * never been rendered has nothing recorded and falls back to the 200px, so it lays out 200px
 * tall until the engine decides it is close enough to the viewport to render it.
 *
 * Two things make that bite in the chat thread:
 *
 *   1. A code block's DOM node is REPLACED when its fence closes. StreamdownBlockContent returns
 *      a bare <Block> while the fence is open and a <div className="relative isolate"> wrapping
 *      the same <Block> once `getCodeFence` matches, and React reconciles a changed element type
 *      by unmounting the old subtree and mounting a new one. The replacement is a brand new
 *      element, so whatever size the old one had recorded is gone. Measured on this tree, that
 *      is a single frame at 226px on a block that was 1722px, per fence, which is the
 *      "reload"-style flicker index.css used to disable the optimization outright to avoid.
 *   2. Every block in a thread is a new element again each time the thread is mounted, so a
 *      freshly opened thread would place every off-screen code block at 200px and then correct
 *      it as the user scrolled into it, moving the content under them.
 *
 * So the rule this implements is: hold every code block at `content-visibility: visible` until
 * the thread is quiet, which guarantees each one is laid out at its real height at least once
 * and therefore has a last remembered size, and only then let the engine skip them. Releasing
 * after that is free of any height change, because the size the engine skips at is the size it
 * measured.
 *
 * The CSS half of this lives in src/index.css and keys off the `data-code-block-layout`
 * attribute that ./code-block-layout-signal.ts writes on `.aui-thread-root`. The attribute being
 * ABSENT means held, so a tree where the signal never runs behaves the way the unconditional
 * override did.
 *
 * This file holds the decision and nothing else, with no React and no DOM, so it can be driven
 * deterministically in a test.
 */

export type CodeBlockLayout = "building" | "settled";

export const CODE_BLOCK_LAYOUT_ATTRIBUTE = "data-code-block-layout";

/**
 * How long the thread has to be quiet before code blocks are allowed to skip.
 *
 * It has to outlast the render that finalizes a message, because the node replacement in (1)
 * above can land in the SAME commit that flips the message out of its running status: a release
 * keyed on "no longer running" alone therefore lets a block be skipped on the very frame it was
 * re-created, which is the flicker again. Measured on this tree, that release is worth 3
 * collapses per reply. It does not need to outlast highlighting: Shiki replaces the text of a
 * line with spans for the same line, and block heights were measured to be identical from the
 * frame the block first exists (816 highlighted tokens) to the frame highlighting finishes
 * (5,544), so a block skipped before it is highlighted was already recorded at the right height.
 */
export const CODE_BLOCK_SETTLE_MS = 900;

type Timers = {
  setTimeout: (callback: () => void, ms: number) => number;
  clearTimeout: (handle: number) => void;
  requestAnimationFrame: (callback: () => void) => number;
  cancelAnimationFrame: (handle: number) => void;
};

const REAL_TIMERS: Timers = {
  setTimeout: (callback, ms) => window.setTimeout(callback, ms),
  clearTimeout: (handle) => {
    window.clearTimeout(handle);
  },
  requestAnimationFrame: (callback) =>
    window.requestAnimationFrame(() => {
      callback();
    }),
  cancelAnimationFrame: (handle) => {
    window.cancelAnimationFrame(handle);
  },
};

export type CodeBlockLayoutController = {
  /** Feed the thread's running state. Idempotent: repeating a value does not restart anything. */
  setRunning(running: boolean): void;
  layout(): CodeBlockLayout;
  dispose(): void;
};

/**
 * The state machine, with its clock injected so it can be driven deterministically in a test.
 *
 * It starts HELD rather than settled. A controller that started settled would let the blocks of
 * a thread that is mid-render skip before any of them had been measured, which is the failure
 * this exists to prevent, and a mount is exactly when that is most likely.
 */
export function createCodeBlockLayoutController(options: {
  onChange: (layout: CodeBlockLayout) => void;
  settleMs?: number;
  timers?: Partial<Timers>;
}): CodeBlockLayoutController {
  const timers: Timers = { ...REAL_TIMERS, ...options.timers };
  const settleMs = options.settleMs ?? CODE_BLOCK_SETTLE_MS;

  let layout: CodeBlockLayout = "building";
  let running: boolean | null = null;
  let frameHandle: number | null = null;
  let timerHandle: number | null = null;

  const cancelPending = (): void => {
    if (frameHandle !== null) {
      timers.cancelAnimationFrame(frameHandle);
      frameHandle = null;
    }
    if (timerHandle !== null) {
      timers.clearTimeout(timerHandle);
      timerHandle = null;
    }
  };

  const set = (next: CodeBlockLayout): void => {
    if (layout === next) return;
    layout = next;
    options.onChange(next);
  };

  const armRelease = (): void => {
    cancelPending();
    // Two frames before the clock starts. The release must not be scheduled from inside the
    // same rendering update that created a block, or the timer can expire before that block has
    // ever been through layout, and a block that has never been laid out is exactly the one
    // with nothing recorded to be skipped at.
    frameHandle = timers.requestAnimationFrame(() => {
      frameHandle = timers.requestAnimationFrame(() => {
        frameHandle = null;
        timerHandle = timers.setTimeout(() => {
          timerHandle = null;
          set("settled");
        }, settleMs);
      });
    });
  };

  return {
    setRunning(next: boolean): void {
      if (running === next) return;
      running = next;
      if (next) {
        cancelPending();
        set("building");
        return;
      }
      armRelease();
    },
    layout: () => layout,
    dispose: cancelPending,
  };
}
