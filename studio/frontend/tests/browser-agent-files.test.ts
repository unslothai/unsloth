// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { agentChat, answerQuestion, endBrowserAgentRun, stopBrowserAgent } = await import("../src/features/browser/agent/agent-store.ts");
const { applyFileEdit, numberedLines } = await import("../src/features/browser/agent/file-tab.ts");
const { browserFile, currentEntry, useBrowserStore } = await import("../src/features/browser/store.ts");

const globals = globalThis as Record<string, unknown>;
const policy = { allowedDomains: [], blockedDomains: [] };
const RUN = "r";
type Result = { ok: boolean; error?: string; text?: string; note?: string; file?: string };

let next = 0;
async function act(op: string, fields: Record<string, unknown> = {}, runKey = RUN): Promise<Result> {
  const posted: { result: Result }[] = [];
  globals.__authFetch = async (_url: string, init: { body: string }) => {
    posted.push(JSON.parse(init.body));
    return new Response(null, { status: 200 });
  };
  await runBrowserAction({ id: `f${next++}`, sessionId: "s", op, direction: "down", policy, ...fields }, { runKey, userText: "" });
  assert.equal(posted.length, 1);
  return posted[0].result;
}

function shownText(): Promise<string> {
  const store = useBrowserStore.getState();
  const tab = store.tabs.find((candidate) => candidate.id === store.activeTabId);
  const entry = tab && currentEntry(tab);
  assert.equal(entry?.kind, "file");
  return browserFile(entry.kind === "file" ? entry.fileId : "")?.text() ?? Promise.resolve("");
}

test("numbered lines page through a file and say where to go on", () => {
  const text = Array.from({ length: 3000 }, (_, i) => `line ${i + 1}`).join("\n");
  const first = numberedLines(text, 1);
  assert.match(first.text, /^1\tline 1\n2\tline 2/);
  const end = Number(/Lines 1-(\d+) of 3000/.exec(first.note)?.[1]);
  assert.ok(end > 100 && end < 3000);
  assert.match(first.note, new RegExp(`start_line ${end + 1}`));
  assert.equal(numberedLines(text, 2999).note, "Lines 2999-3000 of 3000.");
  assert.equal(numberedLines("a\n", 5).note, "The file has 1 line.");
});

test("an edit replaces text found once, refuses ambiguous or missing text, and keeps CRLF", () => {
  assert.deepEqual(applyFileEdit("a\nb\nc", { oldText: "b", newText: "B" }), { text: "a\nB\nc", line: 2 });
  assert.match((applyFileEdit("x x", { oldText: "x", newText: "y" }) as { error: string }).error, /appears 2 times/);
  assert.deepEqual(applyFileEdit("x x", { oldText: "x", newText: "y", replaceAll: true }), { text: "y y", line: 1 });
  assert.match((applyFileEdit("abc", { oldText: "z", newText: "y" }) as { error: string }).error, /not in the file/);
  assert.deepEqual(applyFileEdit("a\r\nb\r\nc", { oldText: "a\nb", newText: "a\nB" }), { text: "a\r\nB\r\nc", line: 1 });
  assert.deepEqual(applyFileEdit("price", { oldText: "price", newText: "$&5" }), { text: "$&5", line: 1 });
  assert.deepEqual(applyFileEdit("old", { content: "new" }), { text: "new", line: 1 });
});

test("the agent reads and edits the HTML file in front, and Back shows the version before", async () => {
  endBrowserAgentRun(RUN);
  const html = "<!doctype html>\n<h1>Hello</h1>\n<button>Go</button>\n";
  useBrowserStore.getState().openFile({ blob: new Blob([html], { type: "text/html" }), name: "page.html", contentType: "text/html", key: "html:a1" });
  const read = await act("read_file", { startLine: 1 });
  assert.equal(read.ok, true);
  assert.equal(read.file, "page.html");
  assert.match(read.text ?? "", /^1\t<!doctype html>\n2\t<h1>Hello<\/h1>/);

  const edited = await act("edit_file", { oldText: "<h1>Hello</h1>", newText: "<h1>Hi there</h1>" });
  assert.equal(edited.ok, true, edited.error);
  assert.match(edited.text ?? "", /2\t<h1>Hi there<\/h1>/);
  assert.equal(await shownText(), html.replace("Hello", "Hi there"));

  const missing = await act("edit_file", { oldText: "<h1>Hello</h1>", newText: "x" });
  assert.equal(missing.ok, false);
  assert.match(missing.error ?? "", /not in the file/);

  const store = useBrowserStore.getState();
  store.goBack(store.activeTabId ?? "");
  assert.equal(await shownText(), html);
});

test("a file's preview is read through its frame, and its links are not taken as the user's", async () => {
  endBrowserAgentRun(RUN);
  useBrowserStore.getState().openFile({ blob: new Blob(["<a href='https://example.com/'>x</a>"], { type: "text/html" }), name: "links.html", contentType: "text/html", key: "html:a2" });
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000 } };
    return { installed: true, result: { ok: true, text: "link r1 x", links: ["https://example.com/"] } };
  };
  const snapshot = await act("snapshot");
  assert.equal(snapshot.ok, true, snapshot.error);
  assert.equal(snapshot.file, "links.html");
  assert.equal(snapshot.text, "link r1 x");
  assert.match(snapshot.note ?? "", /browser_edit_file/);

  globals.__native = true;
  try {
    const desktop = await act("snapshot");
    assert.equal(desktop.text, "link r1 x", desktop.error);
  } finally {
    globals.__native = false;
  }

  let navigate = "https://example.com/";
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000 } };
    if (op.op === "describe") return { installed: true, result: { ok: true, info: { role: "link", name: "x", tag: "a", href: navigate } } };
    return { installed: true, result: { ok: true, navigate } };
  };
  const pending = act("click", { ref: "r1" });
  let question = agentChat("new").question;
  for (let i = 0; !question && i < 200; i++) {
    await new Promise((resolve) => setTimeout(resolve, 10));
    question = agentChat("new").question;
  }
  assert.equal(question?.ask, "open");
  assert.equal(question?.host, "example.com");
  answerQuestion(question.id, false);
  assert.match((await pending).error ?? "", /declined opening a page/);

  navigate = "http://localhost/api/inference/other.html";
  const beside = await act("click", { ref: "r1" });
  assert.match(beside.error ?? "", /beside this one/);
});

test("a preview that misses the first call, as one just switched on from Source does, still answers", async () => {
  endBrowserAgentRun(RUN);
  let calls = 0;
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>, _code: string, timeoutMs: number) => {
    if (calls++ === 0) return new Promise((resolve) => setTimeout(() => resolve(null), timeoutMs));
    if (op.op === "status") return { installed: true, result: { ok: true, quietMs: 1000 } };
    return { installed: true, result: { ok: true, text: "ready", links: [] } };
  };
  const snapshot = await act("snapshot");
  assert.equal(snapshot.text, "ready", snapshot.error);
});

test("a click that takes the preview a while is sent once, not again after a short wait", async () => {
  endBrowserAgentRun(RUN);
  let clicks = 0;
  globals.__frameAgent = (_tabId: string, op: Record<string, unknown>, _code: string, timeoutMs: number) => {
    const answer = (value: unknown, after = 0) =>
      Promise.race([new Promise((resolve) => setTimeout(() => resolve(value), after)), new Promise((resolve) => setTimeout(() => resolve(null), timeoutMs))]);
    if (op.op === "click") {
      clicks++;
      return answer({ installed: true, result: { ok: true } }, 2000);
    }
    if (op.op === "describe") return answer({ installed: true, result: { ok: true, info: { role: "button", name: "Go", tag: "button" } } });
    if (op.op === "status") return answer({ installed: true, result: { ok: true, quietMs: 1000 } });
    return answer({ installed: true, result: { ok: true, text: "page", links: [] } });
  };
  const click = await act("click", { ref: "r1" });
  assert.equal(click.ok, true, click.error);
  assert.equal(clicks, 1);
});

test("a screenshot of a file in a chat that is off screen is refused, not taken of the panel shown", async () => {
  endBrowserAgentRun(RUN);
  useBrowserStore.getState().openFile({ blob: new Blob(["notes"], { type: "text/plain" }), name: "shot.txt", contentType: "text/plain", key: "f:shot" });
  globals.__native = true;
  globals.__panel = false;
  try {
    const shot = await act("screenshot");
    assert.equal(shot.ok, false);
    assert.match(shot.error ?? "", /only runs while the chat is open/);
  } finally {
    globals.__native = false;
    globals.__panel = true;
  }
});

test("reloading a file's preview in the desktop app reloads its frame", async () => {
  endBrowserAgentRun(RUN);
  useBrowserStore.getState().openFile({ blob: new Blob(["<p>x</p>"], { type: "text/html" }), name: "reload.html", contentType: "text/html", key: "html:reload" });
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) =>
    op.op === "status" ? { installed: true, result: { ok: true, quietMs: 1000 } } : { installed: true, result: { ok: true, text: "x", links: [] } };
  const store = useBrowserStore.getState();
  const before = store.tabs.find((tab) => tab.id === store.activeTabId)?.reloadKey;
  globals.__native = true;
  try {
    const reloaded = await act("reload");
    assert.equal(reloaded.ok, true, reloaded.error);
  } finally {
    globals.__native = false;
  }
  const after = useBrowserStore.getState().tabs.find((tab) => tab.id === store.activeTabId)?.reloadKey;
  assert.equal(after, (before ?? 0) + 1);
});

test("an edit stopped while the file is read leaves the file as it was", async () => {
  endBrowserAgentRun(RUN);
  const source = "<h1>Keep</h1>";
  let release = () => {};
  const released = new Promise<void>((resolve) => { release = resolve; });
  const slow = Object.assign(new Blob([source], { type: "text/html" }), { text: () => released.then(() => source) });
  useBrowserStore.getState().openFile({ blob: slow, name: "stop.html", contentType: "text/html", key: "html:stop" });
  const pending = act("edit_file", { oldText: "Keep", newText: "Changed" }, "stopped-run");
  for (let i = 0; i < 100 && agentChat("new").runKey !== "stopped-run"; i++) await new Promise((resolve) => setTimeout(resolve, 5));
  await new Promise((resolve) => setTimeout(resolve, 20));
  stopBrowserAgent("new");
  release();
  const result = await pending;
  assert.equal(result.ok, false);
  assert.equal(await shownText(), source);
  endBrowserAgentRun("stopped-run");
});

test("a PDF is read as text but not edited, and a text file has nothing to click", async () => {
  endBrowserAgentRun(RUN);
  useBrowserStore.getState().openFile({ blob: new Blob(["report body"], { type: "application/pdf" }), name: "report.pdf", contentType: "application/pdf", key: "f:pdf" });
  const read = await act("read_file");
  assert.equal(read.text, "1\treport body");
  const edit = await act("edit_file", { content: "x" });
  assert.equal(edit.ok, false);
  assert.match(edit.error ?? "", /not PDFs/);

  useBrowserStore.getState().openFile({ blob: new Blob(["alpha\nbeta\n"], { type: "text/plain" }), name: "notes.txt", contentType: "text/plain", key: "f:txt" });
  await act("tabs", { tab: useBrowserStore.getState().tabs.filter((tab) => tab.chat === useBrowserStore.getState().chat).length });
  const found = await act("find", { query: "BETA" });
  assert.equal(found.text, "2\tbeta");
  const click = await act("click", { ref: "r1" });
  assert.equal(click.ok, false);
  assert.match(click.error ?? "", /no page to act on/);
});

test("browser_tabs lists files by name and web pages by site, and switches by number", async () => {
  endBrowserAgentRun(RUN);
  const listed = await act("tabs");
  assert.equal(listed.ok, true);
  assert.match(listed.text ?? "", /File: page\.html \(HTML or React, with a preview\)/);
  assert.match(listed.text ?? "", /File: report\.pdf \(document\)/);
  assert.match(listed.text ?? "", /\(the agent's tab, in front\)/);
  const missing = await act("tabs", { tab: 99 });
  assert.equal(missing.ok, false);
  const switched = await act("tabs", { tab: 2 });
  assert.equal(switched.ok, true);
  assert.equal(switched.file, "links.html");
});
