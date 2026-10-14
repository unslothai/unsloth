// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import { test } from "node:test";
import type { ActionContext, BrowserAction } from "../src/features/browser/agent/executor.ts";
import type { ElementInfo } from "../src/features/browser/agent/page-agent.ts";

register("./helpers/browser-executor-resolver.mjs", import.meta.url);
const { parseBrowserAction, runBrowserAction } = await import("../src/features/browser/agent/executor.ts");
const { answerQuestion, stopBrowserAgent, useBrowserAgentStore } = await import("../src/features/browser/agent/agent-store.ts");
const { DEFAULT_PERMISSIONS } = await import("../src/features/browser/agent/permissions.ts");
const { useBrowserStore } = await import("../src/features/browser/store.ts");

const globals = globalThis as Record<string, unknown>;
const url = "https://shop.example.com/";
const policy = { allowedDomains: [], blockedDomains: [] };
const button: ElementInfo = {
  role: "button", name: "Buy now", tag: "button", type: "", href: "", autocomplete: "",
  formMethod: "", formAction: "", formHasPassword: false, formPasswordFilled: false, submits: false,
};
let sequence = 0;
type Result = { ok: boolean; error?: string; text?: string };

function fixture(info: ElementInfo = button, onDescribe?: () => void, allowQuestions = false) {
  const key = `full-access-${++sequence}`;
  useBrowserStore.setState({ tabs: [], activeTabId: null });
  useBrowserStore.getState().openUrl(url, { newTab: true });
  useBrowserAgentStore.setState({
    tabId: useBrowserStore.getState().activeTabId, runKey: null, stoppedRun: null, question: null,
    permissions: DEFAULT_PERMISSIONS, scriptAccess: "ask", historyAccess: "ask",
  });
  const posted: Result[] = [];
  const sent: string[] = [];
  const questions: string[] = [];
  globals.__browserHistory = [{ url, title: "Shop", visitedAt: Date.UTC(2026, 0, 1) }];
  globals.__authFetch = async (_url: string, init: { body: string }) => {
    posted.push(JSON.parse(init.body).result);
    return new Response(null, { status: 200 });
  };
  globals.__frameAgent = async (_tabId: string, op: Record<string, unknown>) => {
    sent.push(String(op.op));
    if (op.op === "describe") onDescribe?.();
    return { installed: true, result: {
      ok: true, quietMs: 1000, requests: 0, doc: "d1", hit: "r1", sig: "control", info,
      x: 20, y: 20, text: op.op === "evaluated" ? "script result" : "page", links: [],
    } };
  };
  // Model the panel finishing a navigation; no mounted iframe delivers onLoad in this Node harness.
  const unsubscribeLoad = useBrowserStore.subscribe((state) => {
    const tab = state.tabs.find((candidate) => candidate.id === state.activeTabId);
    if (tab?.loading) queueMicrotask(() => useBrowserStore.getState().updateTab(tab.id, { loading: false }));
  });
  // Deny unexpected prompts immediately, so the regression fails rather than hanging.
  const unsubscribeQuestion = useBrowserAgentStore.subscribe((state) => {
    if (!state.question) return;
    questions.push(state.question.ask);
    answerQuestion(state.question.id, allowQuestions);
  });
  async function act(patch: Partial<BrowserAction>, bypassPermissions?: boolean) {
    const context: ActionContext & { bypassPermissions?: boolean } = {
      runKey: key, scope: key, userText: `open ${url}`, bypassPermissions,
    };
    await runBrowserAction({ id: `${key}-${++sequence}`, sessionId: "s", op: "click", ref: "r1", policy, ...patch }, context);
    assert.ok(posted.length);
    return posted.at(-1)!;
  }
  const unsubscribe = () => { unsubscribeLoad(); unsubscribeQuestion(); };
  return { act, sent, questions, unsubscribe };
}

const approvals: { name: string; action: Partial<BrowserAction>; info?: ElementInfo; ask: string; sent: string }[] = [
  { name: "opening an unnamed site", action: { op: "navigate", url: "https://new.example.org/" }, ask: "open", sent: "snapshot" },
  { name: "consequential clicks", action: { op: "click" }, ask: "click", sent: "click" },
  { name: "signing in", action: { op: "click" }, info: { ...button, submits: true, formHasPassword: true, formPasswordFilled: true, formAction: url }, ask: "signIn", sent: "click" },
  { name: "posting a form", action: { op: "type", text: "hello", submit: true }, info: { ...button, role: "textbox", tag: "input", name: "Message", formMethod: "post", formAction: url }, ask: "submit", sent: "type" },
  { name: "selecting a consequential option", action: { op: "select", value: "delete" }, info: { ...button, role: "combobox", tag: "select", name: "Action", option: "Delete" }, ask: "click", sent: "select" },
  { name: "activating a button with Enter", action: { op: "press", key: "Enter" }, ask: "click", sent: "press" },
  { name: "clicking an unlabelled point", action: { op: "click", x: 20, y: 20 }, info: { ...button, role: "element", name: "", tag: "canvas" }, ask: "pointer", sent: "click" },
  { name: "dragging unlabelled points", action: { op: "drag", x: 20, y: 20, toX: 30, toY: 30 }, info: { ...button, role: "element", name: "", tag: "canvas" }, ask: "pointer", sent: "pointer" },
  { name: "running a page script", action: { op: "evaluate", script: "return 1;" }, ask: "script", sent: "evaluate" },
  { name: "reading browsing history", action: { op: "history" }, ask: "history", sent: "" },
];

for (const approval of approvals) {
  test(`Full access skips the panel approval for ${approval.name}`, async (t) => {
    const f = fixture(approval.info);
    t.after(f.unsubscribe);
    const result = await f.act(approval.action, true);
    assert.deepEqual(f.questions, []);
    assert.equal(result.ok, true, result.error);
    if (approval.sent) assert.ok(f.sent.includes(approval.sent), f.sent.join(", "));
    else assert.match(result.text ?? "", /Shop/);
  });
  test(`without Full access the panel still asks before ${approval.name}`, async (t) => {
    const f = fixture(approval.info);
    t.after(f.unsubscribe);
    const result = await f.act(approval.action);
    assert.equal(result.ok, false);
    assert.ok(f.questions.includes(approval.ask), f.questions.join(", "));
    if (approval.sent) assert.ok(!f.sent.includes(approval.sent), f.sent.join(", "));
  });
}

test("Full access does not persist into a later action's permission context", async (t) => {
  const f = fixture();
  t.after(f.unsubscribe);
  assert.equal((await f.act({ op: "click" }, true)).ok, true);
  assert.equal((await f.act({ op: "click" }, false)).ok, false);
  assert.deepEqual(f.questions, ["click"]);
});

test("a site opened under Full access needs approval again after leaving Full access", async (t) => {
  const f = fixture();
  t.after(f.unsubscribe);
  const other = "https://new.example.org/";
  assert.equal((await f.act({ op: "navigate", url: other }, true)).ok, true);
  assert.equal((await f.act({ op: "navigate", url: `${other}another-page` }, false)).ok, false);
  assert.deepEqual(f.questions, ["open"]);
});

test("a Full access arrival does not approve later reads of that site", async (t) => {
  const f = fixture();
  t.after(f.unsubscribe);
  assert.equal((await f.act({ op: "navigate", url: "https://new.example.org/" }, true)).ok, true);
  assert.equal((await f.act({ op: "snapshot" }, false)).ok, false);
  assert.deepEqual(f.questions, ["open"]);
});

for (const op of ["click", "drag"] as const) {
  test(`a bypassed unlabelled ${op} is not cached as user approval`, async (t) => {
    const f = fixture({ ...button, role: "element", name: "", tag: "canvas" });
    t.after(f.unsubscribe);
    const action = { op, x: 20, y: 20, toX: 30, toY: 30 };
    assert.equal((await f.act(action, true)).ok, true);
    assert.equal((await f.act(action, false)).ok, false);
    assert.deepEqual(f.questions, ["pointer"]);
  });
}

test("entering Full access preserves approvals the user actually granted", async (t) => {
  const f = fixture(button, undefined, true);
  t.after(f.unsubscribe);
  const approved = "https://approved.example.org/";
  assert.equal((await f.act({ op: "navigate", url: approved }, false)).ok, true);
  assert.deepEqual(f.questions, ["open"]);
  assert.equal((await f.act({ op: "navigate", url: "https://full-only.example.org/" }, true)).ok, true);
  assert.equal((await f.act({ op: "navigate", url: approved }, false)).ok, true);
  assert.deepEqual(f.questions, ["open"]);
});

test("Full access preserves website policy and explicit browsing blocks", async (t) => {
  const f = fixture();
  t.after(f.unsubscribe);
  const blocked = await f.act({ op: "navigate", url, policy: { allowedDomains: [], blockedDomains: ["example.com"] } }, true);
  assert.equal(blocked.ok, false);
  assert.match(blocked.error ?? "", /website policy blocks/);
  useBrowserAgentStore.getState().setDefaultRule({ browsing: "block" });
  assert.equal((await f.act({ op: "navigate", url }, true)).ok, false);
  assert.equal((await f.act({ op: "navigate", url: "file:///etc/passwd" }, true)).ok, false);
  assert.deepEqual(f.questions, []);
  assert.deepEqual(f.sent, []);
});

for (const access of ["scriptAccess", "historyAccess"] as const) {
  test(`Full access preserves ${access} Never`, async (t) => {
    const f = fixture();
    t.after(f.unsubscribe);
    useBrowserAgentStore.setState({ [access]: "never" });
    const result = await f.act({ op: access === "scriptAccess" ? "evaluate" : "history", script: "return 1;" }, true);
    assert.equal(result.ok, false);
    assert.match(result.error ?? "", /doesn't let the agent/);
    assert.deepEqual(f.questions, []);
    assert.deepEqual(f.sent, []);
  });
}

for (const secret of [
  { ...button, role: "password", tag: "input", type: "password" },
  { ...button, role: "textbox", tag: "input", autocomplete: "cc-number" },
  { ...button, role: "file-input", tag: "input", type: "file" },
]) {
  test(`Full access preserves the ${secret.type || secret.autocomplete} field block`, async (t) => {
    const f = fixture(secret);
    t.after(f.unsubscribe);
    assert.equal((await f.act({ op: "type", text: "secret" }, true)).ok, false);
    assert.ok(!f.sent.includes("type"));
    assert.deepEqual(f.questions, []);
  });
}

test("Stop still prevents a Full access click from reaching the page", async (t) => {
  const f = fixture(button, stopBrowserAgent);
  t.after(f.unsubscribe);
  const result = await f.act({ op: "click" }, true);
  assert.equal(result.ok, false);
  assert.match(result.error ?? "", /stopped/);
  assert.ok(!f.sent.includes("click"));
});

test("a browser action payload cannot grant itself Full access", () => {
  const action = parseBrowserAction({ id: "untrusted", op: "click", bypassPermissions: true, bypass_permissions: true });
  assert.ok(action);
  assert.equal("bypassPermissions" in action, false);
  assert.equal("bypass_permissions" in action, false);
});

test("script settings disclose that Full access skips script approval", async () => {
  const { en } = await import("../src/i18n/locales/en.ts");
  assert.match(en.browser.agentSettings.scriptsDescription, /Full access/);
});

test("the chat adapter forwards its captured Full access flag to the browser action context", () => {
  const adapter = readFileSync(new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url), "utf8");
  const start = adapter.indexOf("void runBrowserAction(action, {");
  assert.ok(start >= 0);
  const handoff = adapter.slice(start, adapter.indexOf("});", start));
  assert.match(handoff, /\bbypassPermissions,/);
  assert.doesNotMatch(handoff, /getState\(/);
});
