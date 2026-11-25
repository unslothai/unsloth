// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { type ViteDevServer, createServer } from "vite";
import { readSrc } from "./helpers/kit.ts";

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
  const list = html.slice(html.indexOf("<ul"));
  return [
    ...list.matchAll(/(Older models)|data-model-name="true"[^>]*>([^<]+)/g),
  ].map((match) => (match[1] ? "HEADER" : match[2]));
}

test("closed fold: the Older models header follows the ticked rows", () => {
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

test("closed fold: the header sits below the scroller, so no row can pass under it", () => {
  const html = render({
    models: ["fresh-model", "old-model"],
    showOlder: false,
  });
  const scroller = html.slice(html.indexOf("<ul"), html.indexOf("</ul>"));
  assert.doesNotMatch(scroller, /Older models/);
  assert.match(
    html.slice(html.indexOf("</ul>")),
    /data-fold-header="true"[\s\S]*Older models · 1/,
  );
  assert.doesNotMatch(html, /sticky/);
});

test("open fold: the header scrolls with the rows instead of pinning over them", () => {
  const html = render({
    models: ["fresh-model", "old-model"],
    showOlder: true,
  });
  const scroller = html.slice(html.indexOf("<ul"), html.indexOf("</ul>"));
  assert.match(scroller, /data-fold-header="true"/);
  assert.doesNotMatch(html, /sticky/);
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

test("the edge fade follows the scroll position", () => {
  (globalThis as { MutationObserver?: unknown }).MutationObserver ??= class {
    observe() {}
    disconnect() {}
  };
  const fade = (scrollTop: number) => {
    const style: Record<string, string> = {};
    fadeScrollEdges({
      scrollTop,
      clientHeight: 100,
      scrollHeight: 400,
      style: {
        setProperty: (name: string, value: string) => (style[name] = value),
      },
      addEventListener() {},
    });
    return [/^linear-gradient\(to bottom, (\S+) 0/, /, (\S+) 100%\)$/].map(
      (edge) => style["-webkit-mask-image"].match(edge)?.[1] === "transparent",
    );
  };
  assert.deepEqual(fade(0), [false, true]);
  assert.deepEqual(fade(150), [true, true]);
  assert.deepEqual(fade(300), [true, false]);
});

test("the fold toggle keeps keyboard focus when it moves in or out of the scroller", () => {
  const src = readSrc("features/chat/connection-model-list.tsx");
  assert.match(src, /<button\s+ref=\{foldToggleRef\}/);
  assert.match(
    src,
    /refocusFoldToggle\.current =\s+document\.activeElement === event\.currentTarget/,
  );
  assert.match(
    src,
    /useLayoutEffect\(\(\) => \{\s+if \(refocusFoldToggle\.current\) foldToggleRef\.current\?\.focus\(\);[\s\S]*?\}, \[showOlder\]\);/,
  );
});
