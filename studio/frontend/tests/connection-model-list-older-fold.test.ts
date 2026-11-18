// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { type ViteDevServer, createServer } from "vite";

let vite: ViteDevServer;
let ConnectionModelList: (props: Record<string, unknown>) => unknown;

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
  ({ ConnectionModelList } = await vite.ssrLoadModule(
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

test("a search shows matches flat, with no fold header", () => {
  const html = render({
    models: ["old-model"],
    query: "old",
    showOlder: false,
  });
  assert.deepEqual(listOrder(html), ["old-model"]);
  assert.doesNotMatch(html, /Older models/);
});
