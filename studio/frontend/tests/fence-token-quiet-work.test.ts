// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";

import { readSrc } from "./helpers/kit.ts";

type Result = { source: string; language: string | null; revision?: number };
type Props = { source: string; language: string | null; enabled: boolean; streaming: boolean };
type Effect = { deps: unknown[]; cleanup?: () => void };
const source = readSrc("components/assistant-ui/markdown-text.tsx");
const hookSource = source.slice(source.indexOf("function useFenceTokens("), source.indexOf("\n/**", source.indexOf("function useFenceTokens(")));
const compiled = ts.transpileModule(hookSource, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText;

function harness(initial: Partial<Props> = {}) {
  const slots: unknown[] = [];
  const effects = new Map<number, Effect>();
  let pending: { index: number; deps: unknown[]; effect: () => (() => void) | void }[] = [];
  let cursor = 0;
  let dirty = false;
  let mounted = true;
  let props: Props = { source: "old source", language: "javascript", enabled: true, streaming: false, ...initial };
  const jobs = new Set<() => void>();
  const workers: { callback: (result: Result | null) => void; cancelled: boolean; patient: boolean }[] = [];
  const printing = new Set<() => boolean>();
  const calls: string[] = [];
  const late: ((result: Result) => void)[] = [];
  const cache = new Map<string, Result>();
  const options = { mode: "defer", force: false, recent: false, offThread: false, long: false, grammar: true, workerAvailable: true };
  const same = (a: unknown[], b: unknown[]) => a.length === b.length && a.every((value, index) => Object.is(value, b[index]));
  const runHighlight = ({ code, language }: { code: string; language: string | null }, callback: (result: Result) => void): Result | null => {
    calls.push(code);
    if (!options.grammar) { late.push(callback); return null; }
    const result = { source: code, language };
    cache.set(`${language}:${code}`, result);
    return result;
  };
  const context = {
    useState(initialValue: unknown) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = typeof initialValue === "function" ? initialValue() : initialValue;
      return [slots[index], (value: unknown) => {
        const next = typeof value === "function" ? value(slots[index]) : value;
        if (!Object.is(next, slots[index])) { slots[index] = next; dirty = true; }
      }];
    },
    useRef(initialValue: unknown) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = { current: initialValue };
      return slots[index];
    },
    useCallback(callback: unknown, deps: unknown[]) {
      const index = cursor++;
      const previous = slots[index] as { callback: unknown; deps: unknown[] } | undefined;
      if (!previous || !same(previous.deps, deps)) slots[index] = { callback, deps };
      return (slots[index] as { callback: unknown }).callback;
    },
    useLayoutEffect(effect: () => (() => void) | void, deps: unknown[]) {
      const index = cursor++;
      const previous = effects.get(index);
      if (!previous || !same(previous.deps, deps)) pending.push({ index, deps, effect });
    },
    fenceNeedsQuietWork: (_language: unknown, _source: unknown, streaming: boolean) =>
      !streaming && options.mode === "defer" && !options.force && !options.recent,
    queueFenceWork: (work: () => void) => {
      if (options.force) { work(); return () => {}; }
      jobs.add(work);
      return () => { jobs.delete(work); };
    },
    trimTrailingNewlines: (body: string) => body.replace(/\n+$/, ""),
    fenceHighlightOptions: (code: string, language: string | null) => ({ code, language }),
    settlesOffThread: () => options.offThread,
    tokenizesOffThread: () => options.long,
    code: {
      highlight: runHighlight,
      highlightExact: runHighlight,
      cached: ({ code, language }: { code: string; language: string | null }) => cache.get(`${language}:${code}`) ?? null,
      cover: () => ({ result: null }),
      seed: ({ code, language }: { code: string; language: string | null }, result: Result) => {
        cache.set(`${language}:${code}`, result);
      },
    },
    requestFullHighlight: (_body: unknown, _language: unknown, callback: (result: Result | null) => void, _speculative: unknown, patient: boolean) => {
      if (!options.workerAvailable) return null;
      const worker = { callback, cancelled: false, patient };
      workers.push(worker);
      return () => { worker.cancelled = true; };
    },
    awaitWorker: (_chars: unknown, callback: () => boolean) => {
      printing.add(callback);
      return () => { printing.delete(callback); };
    },
  };
  const hook = new Function(...Object.keys(context), compiled + "\nreturn useFenceTokens;")(...Object.values(context)) as
    (source: string, language: string | null, enabled: boolean, streaming: boolean) => Result | null;
  let result: Result | null = null;
  const render = (change: Partial<Props> = {}) => {
    assert.ok(mounted);
    props = { ...props, ...change };
    let attempts = 0;
    do {
      assert.ok(++attempts < 20, "hook settles without a render loop");
      dirty = false;
      cursor = 0;
      pending = [];
      result = hook(props.source, props.language, props.enabled, props.streaming);
      if (dirty) continue;
      for (const entry of pending) {
        effects.get(entry.index)?.cleanup?.();
        const cleanup = entry.effect();
        effects.set(entry.index, { deps: entry.deps, cleanup: cleanup || undefined });
      }
    } while (dirty);
    return result;
  };
  const flushOne = () => {
    const next = jobs.values().next().value;
    assert.ok(next);
    jobs.delete(next);
    next();
    return render();
  };
  const unmount = () => {
    for (const entry of effects.values()) entry.cleanup?.();
    mounted = false;
  };
  return { render, flushOne, unmount, jobs, workers, printing, calls, late, options, result: () => result };
}

test("cold completed fences perform no tokenization before their queued turn", () => {
  const h = harness();
  assert.equal(h.render(), null);
  assert.equal(h.calls.length, 0);
  assert.equal(h.jobs.size, 1);
  assert.equal(h.flushOne()?.source, "old source");
  assert.deepEqual(h.calls, ["old source"]);
  h.unmount();
});

test("streaming, window, off and recent completions retain immediate origin", () => {
  for (const mode of ["streaming", "window", "off", "recent"]) {
    const h = harness({ streaming: mode === "streaming" });
    h.options.mode = mode === "window" || mode === "off" ? mode : "defer";
    h.options.recent = mode === "recent";
    assert.equal(h.render()?.source, "old source");
    h.options.recent = false;
    h.options.mode = "defer";
    assert.equal(h.render({ source: "next source", streaming: false })?.source, "next source");
    assert.equal(h.jobs.size, 0);
    h.unmount();
  }
});

test("worker results wait for a queued commit and stale source or language callbacks are discarded", () => {
  const h = harness();
  h.options.offThread = true;
  h.options.long = true;
  h.render();
  h.flushOne();
  const first = h.workers[0];
  first.callback({ source: "old source", language: "javascript" });
  assert.equal(h.render(), null);
  assert.equal(h.jobs.size, 1);
  h.render({ language: "python" });
  assert.equal(first.cancelled, true);
  first.callback({ source: "old source", language: "javascript" });
  assert.equal(h.jobs.size, 1);
  h.flushOne();
  h.workers[1].callback({ source: "old source", language: "python" });
  assert.equal(h.flushOne()?.language, "python");
  h.unmount();
});

test("late grammar callbacks coalesce into the latest publication", () => {
  const h = harness();
  h.options.grammar = false;
  h.render();
  h.flushOne();
  const callback = h.late[0];
  callback({ source: "old source", language: "javascript", revision: 1 });
  callback({ source: "old source", language: "javascript", revision: 2 });
  assert.equal(h.jobs.size, 1);
  assert.equal(h.render(), null);
  assert.equal(h.flushOne()?.revision, 2);
  h.unmount();
});

test("a changed completed source clears old tokens before its new queued work runs", () => {
  const h = harness();
  h.render();
  assert.equal(h.flushOne()?.source, "old source");
  assert.equal(h.render({ source: "replacement source" }), null);
  assert.deepEqual(h.calls, ["old source"]);
  assert.equal(h.flushOne()?.source, "replacement source");
  h.unmount();
});

test("disable and unmount cancel queued starts and delayed publications", () => {
  const h = harness();
  h.render();
  h.render({ enabled: false });
  assert.equal(h.jobs.size, 0);
  h.options.offThread = true;
  h.render({ enabled: true });
  h.flushOne();
  const worker = h.workers[0];
  worker.callback({ source: "old source", language: "javascript" });
  assert.equal(h.jobs.size, 1);
  h.unmount();
  assert.equal(h.jobs.size, 0);
  assert.equal(h.printing.size, 0);
  worker.callback({ source: "old source", language: "javascript" });
  assert.equal(h.jobs.size, 0);
});

test("failed short worker requests queue fallback but long requests never tokenize on failure", () => {
  for (const long of [false, true]) {
    const h = harness();
    h.options.offThread = true;
    h.options.long = long;
    h.render();
    h.flushOne();
    const before = h.calls.length;
    h.workers[0].callback(null);
    assert.equal(h.calls.length, before);
    assert.equal(h.jobs.size, long ? 0 : 1);
    if (!long) assert.equal(h.flushOne()?.source, "old source");
    assert.equal(h.workers[0].patient, long);
    h.unmount();
  }
});

test("print fallback honors quiet work normally and completes synchronously when forced", () => {
  for (const force of [false, true]) {
    const h = harness();
    h.options.offThread = true;
    h.options.long = true;
    h.render();
    h.flushOne();
    h.options.force = force;
    const callback = h.printing.values().next().value;
    assert.ok(callback);
    const before = h.calls.length;
    assert.equal(callback(), force);
    if (force) {
      assert.equal(h.jobs.size, 0);
      assert.equal(h.render()?.source, "old source");
    } else {
      assert.equal(h.calls.length, before);
      assert.equal(h.flushOne()?.source, "old source");
    }
    assert.equal(h.workers[0].cancelled, true);
    h.unmount();
  }
});

test("forcing pending startup makes its print fallback available before the snapshot", () => {
  const h = harness();
  h.options.offThread = true;
  h.options.long = true;
  h.render();
  h.options.force = true;
  h.flushOne();
  for (const fallback of h.printing) fallback();
  assert.equal(h.render()?.source, "old source");
  assert.equal(h.jobs.size, 0);
  h.unmount();
});
