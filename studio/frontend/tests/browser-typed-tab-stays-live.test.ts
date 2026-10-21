// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import { test } from "node:test";
import ts from "@typescript/typescript6";
import { userActionGate } from "../src/features/browser/frame-message.ts";
import { readSrc } from "./helpers/kit.ts";

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { chatTabs, currentEntry, useBrowserStore } = await import("../src/features/browser/store.ts");
type Tab = ReturnType<typeof chatTabs>[number];

function sourceCode(file: string, names: string[], globals: Record<string, unknown>): Record<string, unknown> {
  const source = ts.createSourceFile(file, readSrc(file), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const parts = names.map((name) => {
    const node = source.statements.find(
      (statement) =>
        (ts.isFunctionDeclaration(statement) && statement.name?.text === name) ||
        (ts.isVariableStatement(statement) &&
          statement.declarationList.declarations.some((declaration) => declaration.name.getText(source) === name)),
    );
    assert.ok(node, `${name} in ${file}`);
    return node.getText(source);
  });
  const code = ts.transpileModule(`${parts.join("\n")}\nreturn { ${names.join(", ")} };`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
  }).outputText;
  return new Function(...Object.keys(globals), code)(...Object.values(globals));
}

const { liveTabIds, mountedTabIds, retainedTabs } = sourceCode(
  "features/browser/browser-panel.tsx",
  ["MAX_MOUNTED_TABS", "MAX_HIDDEN_WEB_PAGES", "MAX_TYPED_PAGES", "typedTabIds", "mountedTabIds", "liveTabIds", "retainedTabs"],
  { currentEntry },
) as {
  liveTabIds: (mounted: readonly string[], tabs: Tab[], activeTabId: string | null) => Set<string>;
  mountedTabIds: (mounted: readonly string[], tabs: Tab[], activeTabId: string) => string[];
  retainedTabs: (mounted: readonly string[], tabs: Tab[], activeTabId: string | null) => { live: Set<string>; pendingDrafts: Tab[] };
};

function openTabs(urls: string[]): string[] {
  const store = useBrowserStore.getState();
  for (const tab of chatTabs(store)) store.closeTab(tab.id);
  return urls.map((url) => {
    useBrowserStore.getState().openUrl(url, { newTab: true });
    return useBrowserStore.getState().activeTabId ?? "";
  });
}

const tabs = () => chatTabs(useBrowserStore.getState());
let typedAt = 0;
const type = (id: string) => useBrowserStore.getState().updateTab(id, { typedAt: ++typedAt });

function visit(ids: string[]): readonly string[] {
  let mounted: readonly string[] = [];
  for (const id of ids) mounted = mountedTabIds(mounted, tabs(), id);
  return mounted;
}

test("a page typed into stays live when two more tabs are opened", () => {
  const [form, example, banana] = openTabs([
    "https://httpbin.org/forms/post",
    "https://example.com/",
    "https://en.wikipedia.org/wiki/Banana",
  ]);
  type(form);
  const live = liveTabIds([form, example, banana], tabs(), banana);
  assert.ok(live.has(form));
  assert.ok(live.has(example));
});

test("pages not typed into still leave only the newest hidden one live", () => {
  const [a, b, c] = openTabs(["https://a.example/", "https://b.example/", "https://c.example/"]);
  assert.deepEqual([...liveTabIds([a, b, c], tabs(), c)], [c, b]);
});

test("a page typed into stays mounted past the mounted tab limit", () => {
  const ids = openTabs(["https://httpbin.org/forms/post", "https://a.example/", "https://b.example/", "https://c.example/", "https://d.example/"]);
  const [form] = ids;
  type(form);
  const mounted = visit(ids);
  assert.ok(mounted.includes(form));
  assert.ok(liveTabIds(mounted, tabs(), ids[ids.length - 1]).has(form));
});

test("the fourth most recent draft leaves the live set only after consent", () => {
  const ids = openTabs(["https://a.example/", "https://b.example/", "https://c.example/", "https://d.example/", "https://e.example/", "https://f.example/"]);
  const [a, b, c, d, e, f] = ids;
  for (const id of [a, b, c, d]) type(id);
  const mounted = visit(ids);
  assert.deepEqual(mounted, [a, b, c, d, e, f]);
  assert.deepEqual([...liveTabIds(mounted, tabs(), f)], [f, e, d, c, b]);
  const pending = retainedTabs(mounted, tabs(), f);
  assert.deepEqual(pending.pendingDrafts.map((tab) => tab.id), [a]);
  assert.deepEqual([...pending.live], [f, d, c, b, a]);
  useBrowserStore.getState().updateTab(a, { typedAt: undefined });
  const confirmed = retainedTabs(mounted, tabs(), f);
  assert.deepEqual(confirmed.pendingDrafts, []);
  assert.deepEqual([...confirmed.live], [f, e, d, c, b]);
  assert.ok(!mountedTabIds(mounted, tabs(), f).includes(a));
});

test("the active draft is separate from the three hidden draft slots", () => {
  const ids = openTabs(["https://a.example/", "https://b.example/", "https://c.example/", "https://d.example/", "https://e.example/"]);
  for (const id of ids.slice(0, 4)) type(id);
  for (const active of [ids[3], ids[1]]) {
    const result = retainedTabs(visit([...ids, active]), tabs(), active);
    assert.equal(result.live.size, 5);
    assert.deepEqual(result.pendingDrafts, []);
  }
});

test("re-editing an older draft changes eviction order", () => {
  const ids = openTabs(["https://a.example/", "https://b.example/", "https://c.example/", "https://d.example/", "https://e.example/", "https://f.example/"]);
  let now = 0;
  const { useFrameMessages } = sourceCode("features/browser/tab-view.tsx", ["useFrameMessages"], {
    useBrowserStore, useCallback: (fn: unknown) => fn, useT: () => () => "", Date: { now: () => ++now },
  }) as { useFrameMessages: (id: string, origin: string) => (message: { type: "typed" }) => void };
  const edit = (id: string) => useFrameMessages(id, "https://a.example")({ type: "typed" });
  for (const id of ids.slice(0, 4)) edit(id);
  const mounted = visit(ids);
  edit(ids[0]);
  assert.deepEqual(retainedTabs(mounted, tabs(), ids[5]).pendingDrafts.map((tab) => tab.id), [ids[1]]);
  edit(ids[1]);
  assert.deepEqual(retainedTabs(mounted, tabs(), ids[5]).pendingDrafts.map((tab) => tab.id), [ids[2]]);
});

test("pending drafts defer a new page rather than exceed five live web pages", () => {
  const ids = openTabs(["https://a.example/", "https://b.example/", "https://c.example/", "https://d.example/", "https://e.example/", "https://f.example/"]);
  for (const id of ids.slice(0, 5)) type(id);
  const result = retainedTabs(visit(ids), tabs(), ids[5]);
  assert.deepEqual(result.pendingDrafts.map((tab) => tab.id), [ids[0]]);
  assert.deepEqual([...result.live], [ids[4], ids[3], ids[2], ids[1], ids[0]]);
});

test("leaving or reloading the page forgets that it was typed into", () => {
  const [form] = openTabs(["https://httpbin.org/forms/post"]);
  const typed = () => tabs().find((tab) => tab.id === form)?.typedAt !== undefined;
  const store = useBrowserStore.getState();
  type(form);
  store.reload(form);
  assert.equal(typed(), false);
  type(form);
  store.navigate(form, { url: "https://httpbin.org/post", method: "POST", body: "custname=a" });
  assert.equal(typed(), false);
  type(form);
  store.goBack(form);
  assert.equal(typed(), false);
  type(form);
  store.goForward(form);
  assert.equal(typed(), false);
  store.navigate(form, { url: "https://httpbin.org/forms/post" });
  store.updateTab(form, { displayUrl: "https://httpbin.org/other", typedAt: ++typedAt });
  store.reload(form);
  assert.equal(typed(), false);
  assert.equal(currentEntry(tabs()[0]).kind === "web" && (currentEntry(tabs()[0]) as { url: string }).url, "https://httpbin.org/other");
});

test("a typed notice sent after the page left does not mark the next page", () => {
  const source = readSrc("features/browser/page-frame.tsx");
  assert.ok(source.includes('if (message.type === "typed" && left) return;'));
});

test("discarding a frame clears only its own draft mark", () => {
  const ids = openTabs(["https://a.example/", "https://b.example/", "https://c.example/"]);
  for (const id of ids) type(id);
  const { useFrameMessages } = sourceCode("features/browser/tab-view.tsx", ["useFrameMessages"], {
    useBrowserStore, useCallback: (fn: unknown) => fn, useT: () => () => "",
  }) as { useFrameMessages: (id: string, origin: string) => (message: unknown) => void };
  const file = "features/browser/page-frame.tsx";
  const source = ts.createSourceFile(file, readSrc(file), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const component = source.statements.find((node) => ts.isFunctionDeclaration(node) && node.name?.text === "PageFrame");
  assert.ok(component && ts.isFunctionDeclaration(component));
  const effect = component.body?.statements.find((node) => ts.isExpressionStatement(node) && ts.isCallExpression(node.expression)
    && node.expression.expression.getText(source) === "useEffect" && node.getText(source).includes("requests.stop()"));
  assert.ok(effect && ts.isExpressionStatement(effect) && ts.isCallExpression(effect.expression));
  const code = ts.transpileModule(`return (${effect.expression.arguments[0].getText(source)});`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
  }).outputText;
  for (const id of [ids[1], ids[0]]) {
    const noop = () => {};
    const globals = {
      pageRequests: () => ({ stop: noop }), requestForPage: noop, userActionGate, urlRef: { current: null }, frameRef: { current: null },
      window: { addEventListener: noop, removeEventListener: noop }, onPageStorageClear: () => noop,
      onPageStorageWrite: () => noop, origin: "https://a.example", tabIdRef: { current: id },
      onMessageRef: { current: useFrameMessages(id, "https://a.example") },
    };
    const mount = new Function(...Object.keys(globals), code)(...Object.values(globals)) as () => () => void;
    const unmount = mount();
    assert.notEqual(tabs().find((tab) => tab.id === id)?.typedAt, undefined);
    unmount();
    assert.equal(tabs().find((tab) => tab.id === id)?.typedAt, undefined);
    assert.notEqual(tabs().find((tab) => tab.id === ids[2])?.typedAt, undefined);
  }
});

test("only the first private input channel from this page can report edits", () => {
  const backend = readFileSync(new URL("../../backend/routes/browser.py", import.meta.url), "utf8");
  const start = backend.indexOf("      const FOLLOW_CLICK_MS = ");
  const relay = backend.slice(start, backend.indexOf("    </script>", start));
  type Port = { onmessage?: (event: { data: unknown }) => void };
  const child = {};
  const forwarded: unknown[] = [];
  const parent = { postMessage: (data: unknown) => forwarded.push(data) };
  let listener: (event: { source: unknown; data: unknown; ports: Port[] }) => void;
  let now = 0;
  let violation: (event: { effectiveDirective: string; blockedURI: string }) => void;
  const document = {
    createElement: () => ({ contentWindow: child, setAttribute() {} }),
    body: { appendChild() {} }, head: { appendChild() {} },
    addEventListener: (_: string, callback: typeof violation) => { violation = callback; },
  };
  new Function("window", "document", "parent", "performance", "boot", "escapeAttr", "inject", relay)(
    { addEventListener: (_: string, callback: typeof listener) => { listener = callback; } },
    document, parent, { now: () => now }, () => {}, (text: string) => text, () => "",
  );
  const send = (source: unknown, data: unknown, ports: Port[] = []) => listener({ source, data, ports });
  send(parent, { type: "unsloth:browser-html", html: "" });
  const foreign: Port = {};
  send({}, { source: "unsloth-browser-input" }, [foreign]);
  assert.equal(foreign.onmessage, undefined);
  send(child, { source: "unsloth-browser", type: "typed", tabId: "another-tab" });
  assert.deepEqual(forwarded, []);
  const port: Port = {};
  send(child, { source: "unsloth-browser-input" }, [port]);
  assert.ok(port.onmessage);
  const replacement: Port = {};
  send(child, { source: "unsloth-browser-input" }, [replacement]);
  assert.equal(replacement.onmessage, undefined);
  port.onmessage({ data: "true" });
  assert.deepEqual(forwarded, []);
  port.onmessage({ data: true });
  now = 500;
  port.onmessage({ data: true });
  assert.deepEqual(forwarded, [{ source: "unsloth-browser", type: "typed", reset: false }]);
  port.onmessage({ data: false });
  port.onmessage({ data: true });
  now = 1500;
  port.onmessage({ data: true });
  assert.deepEqual(forwarded.slice(1), [
    { source: "unsloth-browser", type: "typed", reset: true },
    { source: "unsloth-browser", type: "typed", reset: false },
    { source: "unsloth-browser", type: "typed", reset: false },
  ]);
  forwarded.length = 0;
  send(child, { source: "unsloth-browser", type: "linkClick", url: "https://docs.example/next" });
  assert.deepEqual(forwarded, []);
  violation!({ effectiveDirective: "frame-src", blockedURI: "https://docs.example/" });
  assert.deepEqual(forwarded, [{ source: "unsloth-browser", type: "navigate", url: "https://docs.example/next" }]);
});

test("bootstrap precedes page scripts without moving a leading doctype", () => {
  const backend = readFileSync(new URL("../../backend/routes/browser.py", import.meta.url), "utf8");
  const injection = backend.slice(backend.indexOf("      const inject = "), backend.indexOf("      let page = null;"));
  const inject = new Function(`${injection}\nreturn inject;`)() as (html: string, tags: string) => string;
  for (const prefix of ["<!doctype html>", "<!-- page -->\n<!doctype html>", ""]) {
    const result = inject(`${prefix}<script>hostile()</script><html lang="en"><head></head><body></body>`, "<script>boot()</script>");
    assert.ok(result.startsWith(`${prefix}<html><head><script>boot()</script></head>`));
    assert.ok(result.indexOf("boot()") < result.indexOf("hostile()"));
  }
});

test("scripted editing cannot claim typing but keyboard editors and paste can", () => {
  const backend = readFileSync(new URL("../../backend/routes/browser.py", import.meta.url), "utf8");
  const start = backend.indexOf("        let scriptedEdit = 0;");
  const code = backend.slice(start, backend.indexOf("        const report = ", start));
  const listeners = new Map<string, (event: unknown) => void>();
  const notices: boolean[] = [];
  const timers: Array<() => void> = [];
  class Element { matches() { return true; } }
  class Document {
    execCommand() { listeners.get("input")?.({ isTrusted: true }); }
    addEventListener(type: string, listener: (event: unknown) => void) { listeners.set(type, listener); }
  }
  const document = new Document();
  new Function("Document", "document", "Element", "edited", "setTimeout", code)(
    Document, document, Element, (value: boolean) => notices.push(value), (callback: () => void) => timers.push(callback),
  );
  document.execCommand();
  listeners.get("input")?.({ isTrusted: false });
  assert.deepEqual(notices, []);
  const apply = Function.prototype.apply;
  let escaped: ((this: Document) => void) | undefined;
  try {
    Function.prototype.apply = function () { escaped = this as typeof escaped; };
    document.execCommand();
  } finally {
    Function.prototype.apply = apply;
  }
  escaped?.call(document);
  assert.deepEqual(notices, []);
  listeners.get("input")?.({ isTrusted: true });
  listeners.get("paste")?.({ isTrusted: true });
  listeners.get("cut")?.({ isTrusted: true });
  assert.deepEqual(notices, [true, true, true]);
  listeners.get("keydown")?.({ isTrusted: true, key: "a", composedPath: () => [new Element()] });
  document.execCommand();
  assert.equal(notices.length, 4);
  for (const timer of timers) timer();
  document.execCommand();
  assert.equal(notices.length, 4);
});
