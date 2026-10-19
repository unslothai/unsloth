// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";
import ts from "@typescript/typescript6";
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

const { liveTabIds, mountedTabIds } = sourceCode(
  "features/browser/browser-panel.tsx",
  ["MAX_MOUNTED_TABS", "MAX_HIDDEN_WEB_PAGES", "MAX_TYPED_PAGES", "typedTabIds", "mountedTabIds", "liveTabIds"],
  { currentEntry },
) as {
  liveTabIds: (mounted: readonly string[], tabs: Tab[], activeTabId: string | null) => Set<string>;
  mountedTabIds: (mounted: readonly string[], tabs: Tab[], activeTabId: string) => string[];
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

test("a page typed into keeps its text when two more tabs are opened", () => {
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

test("only the three pages typed into last are kept beyond the limits", () => {
  const ids = openTabs(["https://a.example/", "https://b.example/", "https://c.example/", "https://d.example/", "https://e.example/", "https://f.example/"]);
  const [a, b, c, d, e, f] = ids;
  for (const id of [a, b, c, d]) type(id);
  const mounted = visit(ids);
  assert.deepEqual(mounted, [b, c, d, e, f]);
  assert.deepEqual([...liveTabIds(mounted, tabs(), f)], [f, e, d, c, b]);
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
});

test("a page counts as typed into only while the reader is in it, at most once a second", () => {
  const frame = {};
  const page = { document: { activeElement: frame as unknown }, navigator: { userActivation: { isActive: true } }, now: 0 };
  const { typedGate } = sourceCode("features/browser/page-frame.tsx", ["userActive", "TYPED_INTERVAL_MS", "typedGate"], {
    document: page.document,
    navigator: page.navigator,
    performance: { now: () => page.now },
  }) as { typedGate: () => (frame: unknown, message: { type: string }) => boolean };
  const typed = typedGate();
  const message = { type: "typed" };
  assert.equal(typed(frame, message), true);
  page.now = 500;
  assert.equal(typed(frame, message), false);
  page.now = 1500;
  assert.equal(typed(frame, message), true);
  page.now = 3000;
  page.document.activeElement = {};
  assert.equal(typed(frame, message), false);
  page.document.activeElement = frame;
  page.navigator.userActivation.isActive = false;
  assert.equal(typed(frame, message), false);
});

test("a typed notice sent after the page left does not mark the next page", () => {
  const source = readSrc("features/browser/page-frame.tsx");
  assert.ok(source.includes('if (message.type === "typed" && (left || !typed(frame, message))) return;'));
});
