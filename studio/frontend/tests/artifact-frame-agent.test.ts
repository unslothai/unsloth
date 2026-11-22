// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const backend = readFileSync(new URL("../../backend/routes/inference.py", import.meta.url), "utf8");
const shell = backend.match(/_ARTIFACT_PREVIEW_FRAME_HTML = """([\s\S]*?)"""/)![1]!.replace(/\\(["\\])/g, "$1");
const script = shell.match(/<script>([\s\S]*?)<\/script>/)![1]!;

function preview() {
  const listeners = new Map<string, Array<(event: unknown) => void>>();
  const reports: Array<Record<string, unknown>> = [];
  const parent = { postMessage: (value: Record<string, unknown>) => reports.push(value) };
  let context: vm.Context;
  class Node {
    appendChild(child: { source: string }) {
      vm.runInContext(child.source, context);
      return child;
    }
  }
  class Element extends Node {
    remove() {}
  }
  class HTMLScriptElement extends Element {
    source = "";
    set text(value: string) {
      this.source = value;
    }
  }
  class Document extends Node {
    get documentElement() {
      return new Element();
    }
    createElement() {
      return new HTMLScriptElement();
    }
    open() {
      listeners.clear();
    }
    write() {}
    close() {}
    addEventListener() {}
  }
  class Crypto {
    getRandomValues(words: Uint32Array) {
      return words.fill(7);
    }
  }
  context = vm.createContext(
    {
      parent, URLSearchParams, location: { search: "?v=test" }, Document, Node, Element, HTMLScriptElement, Crypto,
      crypto: new Crypto(), document: new Document(),
      console: Object.fromEntries(["error", "warn", "info", "log", "debug"].map((level) => [level, () => {}])),
      addEventListener: (type: string, listener: (event: unknown) => void) => {
        listeners.set(type, [...(listeners.get(type) ?? []), listener]);
      },
    },
    { codeGeneration: { strings: false } },
  );
  context.window = context;
  vm.runInContext(script.replace("/* ANNOTATE_JS */", "return {};"), context);
  const message = (data: unknown, source: unknown = parent) => {
    for (const listener of [...(listeners.get("message") ?? [])]) listener({ source, data });
  };
  message({ type: "unsloth:artifact-html", html: "<h1>Coffee</h1>" });
  return { context, message, reports };
}

const plain = (value: unknown) => JSON.parse(JSON.stringify(value));
const agentCode = "(() => ({ run: (op) => ({ ok: true, text: `ran ${op.op}` }) }))();";
const agent = (id: string, fields = {}) => ({ type: "unsloth:browser-command", command: "agent", id, op: { op: "snapshot" }, ...fields });

test("the canvas installs the browser agent from an inline script and answers its commands", () => {
  const frame = preview();
  frame.message(agent("a1"));
  assert.deepEqual(plain(frame.reports.at(-1)), { source: "unsloth-browser", type: "agent", id: "a1", installed: false, result: null });
  frame.message(agent("a2", { code: agentCode }));
  assert.deepEqual(plain(frame.reports.at(-1)), { source: "unsloth-browser", type: "agent", id: "a2", installed: true, result: { ok: true, text: "ran snapshot" } });
  frame.message(agent("a3"));
  assert.equal(frame.reports.at(-1)?.installed, true);
  assert.deepEqual(Object.keys(frame.context).filter((key) => key.startsWith("unslothAgent")), []);
});

test("only the embedder can install the agent", () => {
  const frame = preview();
  frame.message(agent("a1", { code: agentCode }), frame.context);
  assert.equal(frame.reports.some((report) => report.type === "agent"), false);
});
