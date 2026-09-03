// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { LOCAL_MODEL_SOURCES } = await import(
  "../src/features/hub/inventory/constants.ts"
);
const { buildLocalInventoryRows, localSourceLabel } = await import(
  "../src/features/hub/inventory/view-models.ts"
);

function hermesModel(overrides: Record<string, unknown> = {}) {
  return {
    id: "/home/u/.hermes/models/Qwen3-8B-UD-Q4_K_M.gguf",
    load_id: "/home/u/.hermes/models/Qwen3-8B-UD-Q4_K_M.gguf",
    display_name: "Qwen3-8B-UD-Q4_K_M",
    path: "/home/u/.hermes/models/Qwen3-8B-UD-Q4_K_M.gguf",
    source: "hermes",
    model_format: "gguf",
    runtime: "llama_cpp",
    updated_at: 1_756_000_000,
    ...overrides,
  };
}

test("hermes is a known local source", () => {
  assert.ok(LOCAL_MODEL_SOURCES.includes("hermes"));
});

test("a Hermes download is labelled by the app that fetched it", () => {
  assert.equal(localSourceLabel("hermes"), "Hermes");
});

test("a Hermes GGUF becomes a loadable local row", () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const rows = buildLocalInventoryRows([hermesModel() as any]);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].source, "hermes");
  assert.equal(rows[0].sourceLabel, "Hermes");
  assert.equal(rows[0].isGguf, true);
  assert.equal(rows[0].title, "Qwen3-8B-UD-Q4_K_M");
});

test("the chat picker allowlist admits Hermes rows", async () => {
  // PICKER_LOCAL_SOURCES and CHAT_LOCAL_SOURCES are documented as the same set; a source
  // present in one and not the other shows a model in Chat that the picker cannot pick.
  const fs = await import("node:fs/promises");
  const picker = await fs.readFile(
    new URL(
      "../src/features/model-picker/inventory/use-chat-picker-inventory.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const chat = await fs.readFile(
    new URL("../src/features/chat/local-model-options.ts", import.meta.url),
    "utf8",
  );
  const listed = (src: string) =>
    [...src.matchAll(/new Set\(\[([^\]]*)\]\)/g)][0]?.[1].match(/"[a-z_]+"/g) ?? [];
  assert.deepEqual(listed(picker), listed(chat));
  assert.ok(listed(picker).includes('"hermes"'));
});

