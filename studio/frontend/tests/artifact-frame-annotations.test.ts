// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const backend = readFileSync(new URL("../../backend/routes/inference.py", import.meta.url), "utf8");
// Decode the Python string's escaped quotes/backslashes without importing the GPU backend.
const shell = backend.match(/_ARTIFACT_PREVIEW_FRAME_HTML = """([\s\S]*?)"""/)![1]!
  .replace(/\\(["\\])/g, "$1");
const script = shell.match(/<script>([\s\S]*?)<\/script>/)![1]!;

function preview() {
  const listeners = new Map<string, Array<(event: unknown) => void>>();
  const reports: Array<Record<string, unknown>> = [];
  const calls: unknown[] = [];
  let writes = 0;
  const parent = { postMessage: (value: Record<string, unknown>) => reports.push(value) };
  const context = vm.createContext({
    parent, URLSearchParams, location: { search: "?v=test" }, calls,
    console: Object.fromEntries(["error", "warn", "info", "log", "debug"].map((level) => [level, () => {}])),
    addEventListener: (type: string, listener: (event: unknown) => void) => {
      listeners.set(type, [...(listeners.get(type) ?? []), listener]);
    },
    document: {
      // A document meta CSP may prohibit new scripts after the HTML is written.
      createElement: () => { throw new Error("Inline scripts blocked by document CSP"); },
      open: () => listeners.clear(),
      write: () => writes++, close: () => {}, addEventListener: () => {},
    },
  });
  context.window = context;
  vm.runInContext(script.replace("/* ANNOTATE_JS */", code), context);
  const message = (data: unknown, source: unknown = parent) => {
    for (const listener of [...(listeners.get("message") ?? [])]) listener({ source, data });
  };
  message({ type: "unsloth:artifact-html", html: "<h1>Coffee</h1>" });
  return { context, message, reports, calls, writes: () => writes, installs: () => calls.filter((call) => Array.isArray(call) && call[0] === "install").length };
}

const code = `calls.push(["install"]); return {
  start: (color) => { calls.push(["start", color]); post({ type: "annotate", event: "mark", quote: "Coffee" }); },
  stop: () => calls.push(["stop"]),
  forget: (id) => calls.push(["forget", id]),
  number: (numbers) => calls.push(["numbers", numbers])
};`;
const command = (command: string, fields = {}) => ({ type: "unsloth:browser-command", command, ...fields });

test("the rendered artifact accepts annotations without rewriting its document", () => {
  const frame = preview();
  assert.ok(frame.reports.some((report) => report.event === "ready"));
  assert.equal(frame.installs(), 0, "the predeclared factory stays idle until requested");
  frame.message(command("annotateInstall", { code }));
  frame.message(command("annotate", { on: true, color: "#123456" }));
  frame.message(command("annotateForget", { id: 2 }));
  frame.message(command("annotateNumbers", { numbers: [[3, 1]] }));
  frame.message(command("annotate", { on: false }));
  assert.equal(frame.writes(), 1, "annotation must preserve the live preview");
  assert.equal(frame.installs(), 1);
  assert.deepEqual(JSON.parse(JSON.stringify(frame.calls)), [
    ["install"], ["start", "#123456"], ["forget", 2], ["numbers", [[3, 1]]], ["stop"],
  ]);
  assert.ok(frame.reports.some((report) => report.source === "unsloth-browser" && report.quote === "Coffee"));
  assert.equal(frame.context.__unslothInstallAnnotation, undefined);
});

test("a canvas cannot install or toggle annotations by messaging itself", () => {
  const frame = preview();
  frame.message(command("annotateInstall", { code }), frame.context);
  frame.message(command("annotate", { on: true }), frame.context);
  assert.equal(frame.installs(), 0);
  assert.deepEqual(frame.calls, []);
  frame.message(command("annotateInstall", { code }));
  frame.message(command("annotateInstall", { code }));
  assert.equal(frame.installs(), 1, "reentering annotation reuses the engine");
});

test("Request edits preserves the selected file view", () => {
  const panel = readFileSync(new URL("../src/features/browser/browser-panel.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(panel, /if \(!annotating && htmlPreview\) setView/);
  assert.doesNotMatch(panel, /if \(annotating && htmlPreview\).*setAnnotating\(null\)/);
  assert.match(panel, /htmlAnnotating \? \(\s*<WebAnnotateLayer/);
});

test("switching a file's view keeps its saved annotations", () => {
  const panel = readFileSync(new URL("../src/features/browser/browser-panel.tsx", import.meta.url), "utf8");
  // A view in the key remounts the layer, dropping saved comments on a Markdown Preview/Source switch.
  assert.match(panel, /<AnnotateLayer\s+key=\{`\$\{activeTab\.id\}:\$\{activeEntry\.fileId\}`\}/);
});
