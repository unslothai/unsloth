// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";
import ts from "typescript";

type Load = { status: string; url?: string };

function imageHook() {
  const source = readFileSync(
    new URL(
      "../src/components/assistant-ui/use-sandbox-image.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const code = ts.transpileModule(source.replace(/^import .*$/gm, ""), {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;
  const slots: unknown[] = [];
  const effects: { deps: unknown[]; cleanup?: () => void }[] = [];
  const pending: (() => void)[] = [];
  const requests: {
    signal: AbortSignal;
    resolve: (response: unknown) => void;
  }[] = [];
  const live = new Set<string>();
  let cursor = 0;
  let serial = 0;
  const context = {
    exports: {} as { useSandboxImage: (url: string | null) => { state: Load } },
    AbortController,
    URL: {
      createObjectURL: () => {
        const url = `blob:${++serial}`;
        live.add(url);
        return url;
      },
      revokeObjectURL: (url: string) => {
        live.delete(url);
      },
    },
    useRef: () => {
      const i = cursor++;
      return slots[i] ?? (slots[i] = { current: null });
    },
    useState: (initial: unknown) => {
      const i = cursor++;
      if (!(i in slots))
        slots[i] = typeof initial === "function" ? initial() : initial;
      return [
        slots[i],
        (value: unknown) => {
          slots[i] = value;
        },
      ];
    },
    useEffect: (fn: () => (() => void) | undefined, deps: unknown[]) => {
      const i = cursor++;
      if (
        !effects[i] ||
        deps.some((value, j) => value !== effects[i].deps[j])
      ) {
        pending.push(() => {
          effects[i]?.cleanup?.();
          effects[i] = { deps, cleanup: fn() };
        });
      }
    },
    authFetch: (_url: string, init: { signal: AbortSignal }) =>
      new Promise((resolve) => {
        requests.push({ signal: init.signal, resolve });
      }),
  };
  runInNewContext(code, context);
  return {
    live,
    requests,
    render(url: string | null) {
      cursor = 0;
      return context.exports.useSandboxImage(url).state;
    },
    commit() {
      while (pending.length) pending.shift()!();
    },
    unmount() {
      effects.forEach((effect) => effect.cleanup?.());
    },
  };
}

const flush = async () => {
  for (let i = 0; i < 8; i++) await Promise.resolve();
};
const response = { ok: true, blob: async () => ({ type: "image/png" }) };

for (const previous of ["loaded", "failed"] as const) {
  test(`A to B to A does not reuse a cancelled ${previous} result`, async () => {
    const hook = imageHook();
    hook.render("A");
    hook.commit();
    hook.requests[0].resolve(previous === "loaded" ? response : { ok: false });
    await flush();
    const old = hook.render("A");
    assert.equal(old.status, previous);
    assert.equal(hook.render("B").status, "idle");
    hook.commit();
    assert.equal(hook.requests[0].signal.aborted, true);
    assert.equal(hook.live.size, 0);
    assert.equal(
      hook.render("A").status,
      "idle",
      "a cancelled generation must not render during a fresh retry",
    );
    hook.commit();
    hook.requests[1].resolve({ ok: false });
    await flush();
    assert.equal(
      hook.render("A").status,
      "idle",
      "late B must not replace pending A",
    );
    hook.requests[2].resolve(response);
    await flush();
    const current = hook.render("A");
    assert.equal(current.status, "loaded");
    assert.ok(current.url && hook.live.has(current.url));
    assert.notEqual(current.url, old.url);
    hook.unmount();
    assert.equal(hook.live.size, 0);
  });
}
