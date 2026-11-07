// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";
import ts from "@typescript/typescript6";

const source = readFileSync(new URL("../src/components/assistant-ui/reply-paint.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } });

class Reply {
  attributes = new Set<string>();
  scrollWidth = 600;
  clientWidth = 600;
  viewport: object;
  constructor(viewport: object) { this.viewport = viewport; }
  closest(selector: string) { return selector === ".aui-thread-viewport" ? this.viewport : null; }
  hasAttribute(name: string) { return this.attributes.has(name); }
  removeAttribute(name: string) { this.attributes.delete(name); }
  toggleAttribute(name: string, on: boolean) { if (on) this.attributes.add(name); else this.attributes.delete(name); }
}

function world() {
  let intersect: (entries: unknown[]) => void = () => {};
  let resize: (entries: Array<{ target: Reply }>) => void = () => {};
  const resized = new Set<Reply>();
  const exports: { observeReplyPaint?: (reply: Reply) => (() => void) | undefined } = {};
  vm.runInNewContext(outputText, {
    exports,
    IntersectionObserver: class { constructor(cb: typeof intersect) { intersect = cb; } observe() {} unobserve() {} },
    ResizeObserver: class {
      constructor(cb: typeof resize) { resize = cb; }
      observe(target: Reply) { resized.add(target); }
      unobserve(target: Reply) { resized.delete(target); }
    },
  });
  const rootBounds = { top: 0, bottom: 800 };
  return {
    observe: exports.observeReplyPaint!,
    distant: (reply: Reply) => intersect([{ target: reply, rootBounds, boundingClientRect: { top: 5000, bottom: 5600 } }]),
    resize: (...targets: Reply[]) => resize(targets.map((target) => ({ target }))),
    resized,
  };
}

test("a bounded reply that becomes wide off screen gets its overflow back without an intersection", () => {
  const w = world(), viewport = {}, reply = new Reply(viewport), other = new Reply(viewport);
  const stop = w.observe(reply)!;
  w.observe(other);
  w.distant(reply);
  w.distant(other);
  assert.equal(reply.hasAttribute("data-reply-paint"), true, "a distant, fitting reply is bounded");
  // The chat column narrowed: `reply` now overflows, `other` still fits.
  reply.clientWidth = 400;
  w.resize(reply, other);
  assert.equal(reply.hasAttribute("data-reply-paint"), false, "the wide-reply exemption applies at once");
  assert.equal(other.hasAttribute("data-reply-paint"), true, "a reply that still fits stays bounded");
  stop();
  assert.equal(w.resized.has(reply), false, "cleanup stops watching the reply's size");
});
