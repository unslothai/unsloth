// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const script = readFileSync(
  new URL("../public/reload-snapshot.js", import.meta.url),
  "utf8",
);
const indexHtml = readFileSync(
  new URL("../index.html", import.meta.url),
  "utf8",
);

type Listener = (event: Record<string, unknown>) => void;

function createEnvironment(options: {
  navigationType: "navigate" | "reload";
  storage?: Map<string, string>;
  rootHtml?: string;
}) {
  const storage = options.storage ?? new Map<string, string>();
  const listeners = new Map<string, Listener[]>();
  const animationFrames: Array<() => void> = [];
  const appended: Array<{ innerHTML: string; removed: boolean }> = [];
  const clone = {
    innerHTML: options.rootHtml ?? "<main>Chat is ready</main>",
    querySelectorAll: () => [],
  };
  const root = {
    firstElementChild: {},
    cloneNode: () => clone,
    querySelectorAll: () => [],
  };

  const window = {
    addEventListener(name: string, listener: Listener) {
      const current = listeners.get(name) ?? [];
      current.push(listener);
      listeners.set(name, current);
    },
  };
  const document = {
    body: {
      appendChild(element: { innerHTML: string; removed: boolean }) {
        appended.push(element);
      },
    },
    documentElement: {
      appendChild(element: { innerHTML: string; removed: boolean }) {
        appended.push(element);
      },
    },
    getElementById: () => root,
    createElement: () => ({
      className: "",
      inert: false,
      innerHTML: "",
      removed: false,
      setAttribute() {
        // DOM stub.
      },
      remove() {
        this.removed = true;
      },
    }),
  };

  vm.runInNewContext(script, {
    Array,
    Date,
    JSON,
    clearTimeout() {
      // Timer bookkeeping is outside this lifecycle test.
    },
    document,
    getComputedStyle: () => ({ display: "block", visibility: "visible" }),
    innerHeight: 900,
    innerWidth: 1440,
    location: { pathname: "/chat", search: "" },
    performance: {
      getEntriesByType: () => [{ type: options.navigationType }],
    },
    requestAnimationFrame(callback: () => void) {
      animationFrames.push(callback);
    },
    sessionStorage: {
      getItem: (key: string) => storage.get(key) ?? null,
      removeItem: (key: string) => storage.delete(key),
      setItem: (key: string, value: string) => storage.set(key, value),
    },
    setTimeout: () => 1,
    window,
  });

  return {
    storage,
    appended,
    dispatch(name: string, event: Record<string, unknown> = {}) {
      for (const listener of listeners.get(name) ?? []) {
        listener(event);
      }
    },
    runAnimationFrame() {
      const callbacks = animationFrames.splice(0);
      for (const callback of callbacks) {
        callback();
      }
    },
  };
}

test("carries the rendered shell through a reload until the new shell is ready", () => {
  const outgoing = createEnvironment({
    navigationType: "navigate",
    rootHtml: "<main>Existing chat</main>",
  });
  outgoing.dispatch("pageswap", {
    activation: { navigationType: "reload" },
  });

  const incoming = createEnvironment({
    navigationType: "reload",
    storage: outgoing.storage,
  });
  assert.equal(incoming.appended.length, 1);
  assert.equal(incoming.appended[0].innerHTML, "<main>Existing chat</main>");
  assert.equal(incoming.storage.size, 0);

  incoming.dispatch("unsloth:app-shell-ready");
  assert.equal(incoming.appended[0].removed, false);
  incoming.runAnimationFrame();
  assert.equal(incoming.appended[0].removed, false);
  incoming.runAnimationFrame();
  assert.equal(incoming.appended[0].removed, true);
});

test("restores from a parser-blocking head script before the first body paint", () => {
  const scriptPosition = indexHtml.indexOf(
    '<script src="/reload-snapshot.js">',
  );
  assert.ok(scriptPosition > 0);
  assert.ok(scriptPosition < indexHtml.indexOf("</head>"));
});

test("does not retain the shell for a non-reload navigation", () => {
  const environment = createEnvironment({ navigationType: "navigate" });
  environment.dispatch("pageswap", {
    activation: { navigationType: "push" },
  });
  assert.equal(environment.storage.size, 0);
});
