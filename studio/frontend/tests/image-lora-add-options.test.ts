// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { DiffusionLoraInfo } from "../src/features/images/api.ts";
import {
  customLoraId,
  loraAddGroups,
} from "../src/features/images/lib/lora-add-options.ts";

function lora(
  id: string,
  extra: Partial<DiffusionLoraInfo> = {},
): DiffusionLoraInfo {
  return {
    id,
    display_name: id.replace(/-[0-9a-f]{8}$/, ""),
    source: "local",
    format: "safetensors",
    families: [],
    size_bytes: 1,
    weight_default: 1,
    ...extra,
  };
}

const CATALOG = [
  lora("sks-dog", { fine_tuned: true }),
  lora("jyzjk-0f53895d"),
  lora("NexBlend12-14b044d2"),
  lora("pixel_art-2c81fb07"),
  lora("krea/Krea-2-LoRA-neondrip", {
    source: "hub",
    display_name: "Krea 2 Neon Drip",
  }),
];

const headings = (groups: ReturnType<typeof loraAddGroups>) =>
  groups.map((g) => g.heading);
const ids = (groups: ReturnType<typeof loraAddGroups>) =>
  groups.flatMap((g) => g.options.map((o) => o.lora.id));

test("every catalog LoRA is listed, grouped as fine-tuned, on this device, then Hugging Face", () => {
  const groups = loraAddGroups(CATALOG, new Set(), "");
  assert.deepEqual(headings(groups), [
    "Fine-tuned in Unsloth",
    "On this device",
    "From Hugging Face",
  ]);
  assert.deepEqual(
    ids(groups),
    CATALOG.map((l) => l.id),
  );
});

test("a LoRA already in a row stays listed, marked added", () => {
  const groups = loraAddGroups(CATALOG, new Set(["NexBlend12-14b044d2"]), "");
  const added = groups.flatMap((g) => g.options).filter((o) => o.added);
  assert.deepEqual(
    added.map((o) => o.lora.id),
    ["NexBlend12-14b044d2"],
  );
  assert.equal(ids(groups).length, CATALOG.length);
});

test("search matches the display name or the id, case-insensitively, and drops empty groups", () => {
  assert.deepEqual(ids(loraAddGroups(CATALOG, new Set(), "nexblend")), [
    "NexBlend12-14b044d2",
  ]);
  assert.deepEqual(ids(loraAddGroups(CATALOG, new Set(), "  NEON drip ")), [
    "krea/Krea-2-LoRA-neondrip",
  ]);
  assert.deepEqual(ids(loraAddGroups(CATALOG, new Set(), "2c81fb07")), [
    "pixel_art-2c81fb07",
  ]);
  assert.deepEqual(headings(loraAddGroups(CATALOG, new Set(), "pixel")), [
    "On this device",
  ]);
  assert.deepEqual(loraAddGroups(CATALOG, new Set(), "no such lora"), []);
});

test("only a typed owner/name that is not listed is offered as a Hugging Face id", () => {
  const none = new Set<string>();
  assert.deepEqual(customLoraId("  someone/cool-lora ", CATALOG, none), {
    id: "someone/cool-lora",
    added: false,
  });
  assert.deepEqual(
    customLoraId("someone/cool-lora:style.safetensors", CATALOG, none),
    { id: "someone/cool-lora:style.safetensors", added: false },
  );
  assert.equal(customLoraId("pixel", CATALOG, none), null);
  assert.equal(customLoraId("", CATALOG, none), null);
  assert.equal(customLoraId("some one/lora", CATALOG, none), null);
  assert.equal(customLoraId("krea/Krea-2-LoRA-neondrip", CATALOG, none), null);
});

test("a typed id already in a row is offered as added, so it cannot be added twice", () => {
  const taken = new Set(["someone/custom"]);
  assert.deepEqual(customLoraId("someone/custom", CATALOG, taken), {
    id: "someone/custom",
    added: true,
  });
  assert.deepEqual(customLoraId("  someone/custom  ", CATALOG, taken), {
    id: "someone/custom",
    added: true,
  });
  assert.deepEqual(customLoraId("someone/other", CATALOG, taken), {
    id: "someone/other",
    added: false,
  });
});
