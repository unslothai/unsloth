// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// WebKitGTK parks settled offscreen fences as plain text. These run the real hook and listener
// code from code-fence-defer.tsx against small fakes, so they check behaviour, not spelling.

import assert from "node:assert/strict";
import test from "node:test";
import { runInNewContext } from "node:vm";
import ts from "@typescript/typescript6";

import {
  advanceHighlightProgress,
  EMPTY_HIGHLIGHT_PROGRESS,
  HIGHLIGHT_BATCH_TOKENS,
  type HighlightProgress,
  initialHighlightProgress,
  needsHighlightBatches,
  retainHighlightProgress,
} from "../src/components/assistant-ui/fence-highlight-progress.ts";
import { WINDOW_CAP_LINES } from "../src/components/assistant-ui/code-fence-window.ts";

import { readSrc } from "./helpers/kit.ts";

const DEFER = readSrc("components/assistant-ui/code-fence-defer.tsx");
const PARSED = ts.createSourceFile("code-fence-defer.tsx", DEFER, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

function declarations(names: readonly string[]): string {
  const wanted = new Set(names);
  const found: string[] = [];
  for (const statement of PARSED.statements) {
    const named = ts.isFunctionDeclaration(statement)
      ? [statement.name?.text]
      : ts.isVariableStatement(statement)
        ? statement.declarationList.declarations.map((node) => ts.isIdentifier(node.name) ? node.name.text : undefined)
        : [];
    if (!named.some((name) => name !== undefined && wanted.has(name))) continue;
    for (const name of named) if (name !== undefined) wanted.delete(name);
    found.push(statement.getText(PARSED).replace(/^export\s+/, ""));
  }
  assert.deepEqual([...wanted], [], "every extracted declaration exists in code-fence-defer.tsx");
  return ts.transpileModule(found.join("\n"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.React },
  }).outputText;
}

// Just enough of React's hook semantics: render-phase setState re-renders at once, layout effects
// run after a settled render when their deps change.
function hookRunner<P, R>(hook: (props: P) => R) {
  const slots: unknown[] = [];
  const effects: { deps: unknown[]; cleanup?: (() => void) | void }[] = [];
  let pending: { index: number; run: () => (() => void) | void; deps: unknown[] }[] = [];
  let slot = 0;
  let effect = 0;
  let dirty = false;
  let props: P;
  let result: R;
  const same = (a: unknown[] | undefined, b: unknown[]) =>
    a !== undefined && a.length === b.length && a.every((value, i) => Object.is(value, b[i]));
  const react = {
    useState<T>(init: T | (() => T)): [T, (next: T) => void] {
      const index = slot++;
      if (!(index in slots)) slots[index] = typeof init === "function" ? (init as () => T)() : init;
      return [slots[index] as T, (next: T) => {
        if (Object.is(slots[index], next)) return;
        slots[index] = next;
        dirty = true;
      }];
    },
    useMemo<T>(make: () => T, deps: unknown[]): T {
      const index = slot++;
      const known = slots[index] as { deps: unknown[]; value: T } | undefined;
      if (known !== undefined && same(known.deps, deps)) return known.value;
      const value = make();
      slots[index] = { deps, value };
      return value;
    },
    useLayoutEffect(run: () => (() => void) | void, deps: unknown[]) {
      const index = effect++;
      if (!same(effects[index]?.deps, deps)) pending.push({ index, run, deps });
    },
  };
  const render = (next: P): R => {
    props = next;
    for (let rounds = 0; ; rounds += 1) {
      assert.ok(rounds < 100, "renders settle");
      do {
        dirty = false;
        slot = 0;
        effect = 0;
        pending = [];
        result = hook(props);
      } while (dirty);
      const ready = pending;
      pending = [];
      for (const { index, run, deps } of ready) {
        effects[index]?.cleanup?.();
        effects[index] = { deps, cleanup: run() };
      }
      if (!dirty) return result;
    }
  };
  return { react, render, rerender: () => render(props) };
}

test("a fully highlighted fence parked as plain text comes back in quiet-work batches", () => {
  type Tokens = { content: string }[][];
  type Hook = (
    tokens: Tokens,
    isIncomplete: boolean,
    windowing: boolean,
    language: string,
    source: string,
    code: { current: null },
    frame: { current: null },
    parked: boolean,
  ) => { progress: HighlightProgress | null; staged: boolean };
  const queue: { run: () => boolean; flush: () => void; cancelled: boolean }[] = [];
  const context: Record<string, unknown> = {
    exports: {} as { hook?: Hook },
    window: {},
    printing: false,
    findBarOpen: () => false,
    fenceMode: () => "defer",
    justStreamed: () => false,
    FENCE_HEAD_CHARS: 512,
    trimmedLength: (source: string) => source.trim().length,
    WINDOW_CAP_LINES,
    needsHighlightBatches,
    initialHighlightProgress,
    advanceHighlightProgress,
    retainHighlightProgress,
    EMPTY_HIGHLIGHT_PROGRESS,
    watchHighlighting: () => {},
    unwatchHighlighting: () => {},
    visibleHighlightLine: () => 0,
    highlightQueue: {
      add(run: () => boolean, flush: () => void) {
        const entry = { run, flush, cancelled: false };
        queue.push(entry);
        return () => {
          entry.cancelled = true;
        };
      },
    },
  };
  const runner = hookRunner((props: Parameters<Hook>) => (context.exports as { hook: Hook }).hook(...props));
  Object.assign(context, runner.react);
  runInNewContext(`${declarations(["useFenceHighlightLimit"])}\nexports.hook = useFenceHighlightLimit;`, context);

  // Dense and under the line cap: the shape WebKitGTK leaves on the batched path.
  const lines = 120;
  assert.ok(lines < WINDOW_CAP_LINES);
  const tokens: Tokens = Array.from({ length: lines }, (_, line) =>
    Array.from({ length: 64 }, (_, at) => ({ content: `t${line}_${at} ` })));
  assert.ok(needsHighlightBatches(tokens, WINDOW_CAP_LINES));
  const refs = { current: null };
  const props = (parked: boolean): Parameters<Hook> =>
    [tokens, false, false, "text", "source", refs, refs, parked];
  const live = () => queue.filter((entry) => !entry.cancelled);
  const highlighted = (progress: HighlightProgress | null) =>
    progress === null ? Number.POSITIVE_INFINITY
      : progress.counts.flat().reduce((sum, count) => sum + count, 0);

  let state = runner.render(props(false));
  assert.ok(state.staged);
  for (let steps = 0; state.progress !== null; steps += 1) {
    assert.ok(steps < 100, "highlighting completes");
    const [next] = live();
    assert.ok(next, "pending batches are queued");
    next.cancelled = true;
    next.run();
    state = runner.rerender();
  }
  assert.equal(state.progress, null, "fully highlighted before scrolling away");

  state = runner.render(props(true));
  assert.equal(live().length, 0, "no highlight work runs while the fence is plain text");

  state = runner.render(props(false));
  assert.ok(
    highlighted(state.progress) <= HIGHLIGHT_BATCH_TOKENS,
    `returning remounts at most one batch, got ${highlighted(state.progress)} of ${lines * 64} tokens`,
  );
  assert.equal(live().length, 1, "the rest is queued as quiet work again");

  // Idempotent: another render while still on screen does not restart the batches.
  const before = state.progress;
  state = runner.rerender();
  assert.equal(state.progress, before);
});

type Listener = (event: Record<string, unknown>) => void;
class Target {
  listeners = new Map<string, Set<Listener>>();
  addEventListener(type: string, listener: Listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(listener);
  }
  removeEventListener(type: string, listener: Listener) {
    this.listeners.get(type)?.delete(listener);
  }
  fire(type: string, event: Record<string, unknown> = {}) {
    for (const listener of [...(this.listeners.get(type) ?? [])]) listener(event);
  }
}

for (const [pointerType, pressedEarlier] of [["touch", false], ["mouse", false], ["mouse", true]] as const) {
  const when = pressedEarlier ? " pressed before the listeners attached" : "";
  test(`a plain fence skipped under a held ${pointerType} pointer${when} is restored when it lifts`, () => {
    const window = new Target();
    const document = Object.assign(new Target(), {
      documentElement: { hasAttribute: () => false },
      getSelection: () => null,
    });
    const frames: (() => void)[] = [];
    type Observed = (entries: { isIntersecting: boolean }[]) => void;
    // Assigned inside the fake observer, so annotate rather than let TS narrow the initial null.
    let observed = null as Observed | null;
    let held = false;
    const plain: boolean[] = [];
    const node = { isConnected: true, contains: () => false, closest: () => null };
    const context: Record<string, unknown> = {
      exports: {},
      window,
      document,
      requestAnimationFrame: (run: () => void) => frames.push(run),
      cancelAnimationFrame: () => {},
      forgetScrollable: () => {},
      inputPointerHeld: () => held,
      Node: Object,
      MutationObserver: class {
        observe() {}
        disconnect() {}
      },
      IntersectionObserver: class {
        constructor(callback: (entries: { isIntersecting: boolean }[]) => void) {
          observed = callback;
        }
        observe() {}
        disconnect() {}
      },
      useState: (init: boolean) => [init, (value: boolean) => plain.push(value)],
      useLayoutEffect: (run: () => void) => run(),
    };
    runInNewContext(`${declarations([
      "windowedFences", "windowFrame", "windowWatched", "findBarObserver", "frameRects",
      "frameFlying", "frameSettle", "pointerHeld", "heldLastFrame", "frameSelection", "printing",
      "liveSelection", "remeasureWindows", "FIND_BAR_FLAG", "findBarOpen", "scheduleRemeasure",
      "onPointerDown", "releasePointer", "onPointerMove", "onSelectionChange", "watchWindows",
      "unwatchWindows", "useOffscreenFenceText",
    ])}\nexports.hook = useOffscreenFenceText;`, context);
    const hook = (context.exports as { hook: (frame: { current: unknown }, enabled: boolean) => boolean }).hook;
    hook({ current: node }, true);
    assert.ok(observed);
    const intersect = observed as Observed;

    intersect([{ isIntersecting: false }]);
    assert.deepEqual(plain, [true], "parked as plain text while far off screen");

    // The finger (or button) goes down, the fence scrolls into view, and the pointer pauses.
    held = true;
    if (!pressedEarlier) document.fire("pointerdown", { pointerType, button: 0, target: {} });
    intersect([{ isIntersecting: true }]);
    assert.deepEqual(plain, [true], "a held pointer defers the restore");

    // Lift without any further scroll, resize or selection change.
    held = false;
    window.fire("pointerup", { pointerType, button: 0 });
    for (let steps = 0; frames.length > 0; steps += 1) {
      assert.ok(steps < 10);
      frames.shift()!();
    }
    assert.deepEqual(plain, [true, false], "the release reaches the plain fence and restores its highlighting");
  });
}
