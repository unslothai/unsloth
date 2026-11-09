// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Config handed in before mermaid loads must be the one in force at the first render.
import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Plugin = {
  name: string;
  type: string;
  language: string;
  getMermaid(config?: Record<string, unknown>): {
    initialize(config: Record<string, unknown>): void;
    render(id: string, source: string): Promise<{ svg: string }>;
  };
};

function harness(options: { failFirstLoad?: boolean } = {}) {
  const calls: unknown[][] = [];
  let loads = 0;
  const stock = {
    createMermaidPlugin(pluginOptions: unknown) {
      calls.push(["create", pluginOptions]);
      const instance = {
        initialize(config: unknown) {
          calls.push(["initialize", config]);
        },
        async render(id: string, source: string) {
          calls.push(["render", id, source]);
          return { svg: `<svg>${source}</svg>` };
        },
      };
      return {
        name: "mermaid",
        type: "diagram",
        language: "mermaid",
        getMermaid(config?: unknown) {
          if (config) {
            instance.initialize(config);
          }
          return instance;
        },
      };
    },
  };
  const { mermaid } = loadWithStubs<{ mermaid: Plugin }>(
    new URL("../src/components/assistant-ui/mermaid-plugin.ts", import.meta.url),
    {
      get "@streamdown/mermaid"() {
        loads += 1;
        if (options.failFirstLoad && loads === 1) throw new Error("chunk failed");
        return stock;
      },
      "./mermaid-guard.ts": { stripRemoteDiagramImages: (source: string) => `guarded:${source}` },
    },
  );
  return { mermaid, calls, loads: () => loads };
}

test("loading the plugin does not load mermaid", () => {
  const { mermaid, calls, loads } = harness();
  assert.deepEqual(
    { name: mermaid.name, type: mermaid.type, language: mermaid.language },
    { name: "mermaid", type: "diagram", language: "mermaid" },
  );
  mermaid.getMermaid({ theme: "dark" });
  assert.equal(loads(), 0);
  assert.deepEqual(calls, []);
});

test("the last config before the first render is applied before it, then the guarded source renders", async () => {
  const { mermaid, calls } = harness();
  mermaid.getMermaid({ theme: "dark" });
  const instance = mermaid.getMermaid({ theme: "neutral" });
  assert.deepEqual(await instance.render("m1", "graph TD; A-->B"), {
    svg: "<svg>guarded:graph TD; A-->B</svg>",
  });
  assert.equal(calls[0][0], "create");
  assert.ok((calls[0][1] as { config: { secure: string[] } }).config.secure.includes("themeCSS"));
  assert.deepEqual(calls.slice(1), [
    ["initialize", { theme: "neutral" }],
    ["render", "m1", "guarded:graph TD; A-->B"],
  ]);
});

test("once loaded, a config initializes at once, as the eager plugin did", async () => {
  const { mermaid, calls } = harness();
  await mermaid.getMermaid().render("m1", "x");
  calls.length = 0;
  mermaid.getMermaid({ theme: "forest" });
  assert.deepEqual(calls, [["initialize", { theme: "forest" }]]);
  mermaid.getMermaid().initialize({ theme: "base" });
  assert.deepEqual(calls[1], ["initialize", { theme: "base" }]);
});

test("a render with no config handed in initializes nothing itself", async () => {
  const { mermaid, calls } = harness();
  await mermaid.getMermaid().render("m1", "x");
  assert.deepEqual(calls.map((call) => call[0]), ["create", "render"]);
});

test("a failed load is retried by the next diagram", async () => {
  const { mermaid, loads } = harness({ failFirstLoad: true });
  await assert.rejects(mermaid.getMermaid().render("m1", "x"), /chunk failed/);
  assert.deepEqual(await mermaid.getMermaid().render("m2", "y"), { svg: "<svg>guarded:y</svg>" });
  assert.equal(loads(), 2);
});
