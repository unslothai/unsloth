// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { type ViteDevServer, createServer } from "vite";

let vite: ViteDevServer;
let ConnectionModelList: (props: Record<string, unknown>) => unknown;
let fadeScrollEdges: (list: unknown) => (() => void) | undefined;

before(async () => {
  vite = await createServer({
    appType: "custom",
    server: { middlewareMode: true },
  });
  const catalog = await vite.ssrLoadModule(
    "/src/features/chat/model-catalog.ts",
  );
  catalog.setModelsDevCatalog({
    fetched_at: Date.now(),
    providers: {
      openai: {
        "fresh-model": { released: new Date().toISOString().slice(0, 10) },
        "old-model": { released: "2020-01-01" },
        "old-ticked": { released: "2020-01-01" },
      },
    },
  });
  ({ ConnectionModelList, fadeScrollEdges } = await vite.ssrLoadModule(
    "/src/features/chat/connection-model-list.tsx",
  ));
});

after(async () => {
  await vite.close();
});

const noop = () => {};
const render = (props: Record<string, unknown>) =>
  renderToStaticMarkup(
    createElement(ConnectionModelList as never, {
      providerType: "openai",
      panelKey: "t",
      totalCount: 3,
      selectedIds: ["old-ticked"],
      query: "",
      onQueryChange: noop,
      olderCount: 1,
      onShowOlderChange: noop,
      onToggle: noop,
      onSelectAll: noop,
      onClear: noop,
      ...props,
    }),
  );

function listOrder(html: string): string[] {
  const list = html.slice(html.indexOf("<ul"), html.indexOf("</ul>"));
  return [
    ...list.matchAll(/(Older models)|data-model-name="true"[^>]*>([^<]+)/g),
  ].map((match) => (match[1] ? "HEADER" : match[2]));
}

test("closed fold: the Older models header sits inside the list after the ticked rows", () => {
  const html = render({
    models: ["fresh-model", "old-ticked"],
    showOlder: false,
  });
  assert.deepEqual(listOrder(html), ["fresh-model", "old-ticked", "HEADER"]);
  assert.match(html, /aria-expanded="false"[^>]*>Older models · 1</);
});

test("open fold: unticked older models follow the header, ticked ones stay above it", () => {
  const html = render({
    models: ["fresh-model", "old-model", "old-ticked"],
    showOlder: true,
  });
  assert.deepEqual(listOrder(html), [
    "fresh-model",
    "old-ticked",
    "HEADER",
    "old-model",
  ]);
  assert.match(html, /aria-expanded="true"[^>]*>Older models · 1</);
});

test("the header pins to both list edges and Tab scrolls rows clear of it", () => {
  const grouped = render({ models: ["fresh-model"], showOlder: false });
  assert.match(grouped, /<ul[^>]*scroll-py-12/);
  assert.match(grouped, /data-fold-header="true"[^>]*sticky top-0 bottom-0/);
  const flat = render({ models: ["old-model"], query: "old", showOlder: false });
  assert.doesNotMatch(flat, /scroll-py-12/);
});

test("a search shows matches flat, with no fold header", () => {
  const html = render({
    models: ["old-model"],
    query: "old",
    showOlder: false,
  });
  assert.deepEqual(listOrder(html), ["old-model"]);
  assert.doesNotMatch(html, /Older models/);
});

test("the header pins to either list edge with no fade over it", () => {
  (globalThis as { MutationObserver?: unknown }).MutationObserver ??= class {
    observe() {}
    disconnect() {}
  };
  // A 100px window onto 400px of rows; the header is 20px tall.
  const fade = (headerTop: number | null) => {
    const attrs = new Set<string>();
    const style: Record<string, string> = {};
    const header = {
      getBoundingClientRect: () => ({ top: headerTop!, bottom: headerTop! + 20 }),
      toggleAttribute: (name: string, on: boolean) =>
        on ? attrs.add(name) : attrs.delete(name),
    };
    fadeScrollEdges({
      scrollTop: 150,
      clientHeight: 100,
      scrollHeight: 400,
      getBoundingClientRect: () => ({ top: 0, bottom: 100 }),
      querySelector: () => (headerTop === null ? null : header),
      style: { setProperty: (name: string, value: string) => (style[name] = value) },
      addEventListener() {},
    });
    const [top, bottom] = [/^linear-gradient\(to bottom, (\S+) 0/, /, (\S+) 100%\)$/].map(
      (edge) => style["-webkit-mask-image"].match(edge)?.[1] === "transparent",
    );
    return { top, bottom, stuck: attrs.has("data-stuck") };
  };
  assert.deepEqual(fade(null), { top: true, bottom: true, stuck: false });
  assert.deepEqual(fade(40), { top: true, bottom: true, stuck: false });
  assert.deepEqual(fade(0), { top: false, bottom: true, stuck: true });
  assert.deepEqual(fade(80), { top: true, bottom: false, stuck: true });
});
