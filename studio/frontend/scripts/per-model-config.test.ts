// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_PER_MODEL_CONFIG,
  MAX_CHAT_TEMPLATE_BYTES,
  MAX_CHAT_TEMPLATE_LENGTH,
  MAX_PER_MODEL_CONFIG_STORAGE_BYTES,
  chatTemplateByteLength,
  clampChatTemplateToByteLimit,
  deletePerModelConfig,
  loadPerModelConfig,
  savePerModelConfig,
} from "../src/features/chat/model-config/per-model-config.ts";
import { modelStorageKey } from "../src/lib/model-identity.ts";

const STORAGE_KEY = "unsloth_model_configs";

class MemoryStorage implements Storage {
  private readonly map = new Map<string, string>();

  get length(): number {
    return this.map.size;
  }

  clear(): void {
    this.map.clear();
  }

  getItem(key: string): string | null {
    return this.map.get(key) ?? null;
  }

  key(index: number): string | null {
    return Array.from(this.map.keys())[index] ?? null;
  }

  removeItem(key: string): void {
    this.map.delete(key);
  }

  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }
}

function installStorage(): MemoryStorage {
  const storage = new MemoryStorage();
  Object.defineProperty(globalThis, "window", {
    value: {},
    configurable: true,
  });
  Object.defineProperty(globalThis, "localStorage", {
    value: storage,
    configurable: true,
  });
  return storage;
}

function byteLength(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}

function readStoredMap(storage: Storage): Record<string, unknown> {
  const raw = storage.getItem(STORAGE_KEY);
  return raw ? JSON.parse(raw) : {};
}

test("per-model config storage evicts by serialized bytes and keeps latest", () => {
  const storage = installStorage();
  const template = "x".repeat(MAX_CHAT_TEMPLATE_LENGTH);

  for (let i = 0; i < 40; i += 1) {
    assert.equal(
      savePerModelConfig(`Org/Model-${i}`, null, {
        ...DEFAULT_PER_MODEL_CONFIG,
        chatTemplateOverride: template,
      }),
      true,
    );
  }

  const raw = storage.getItem(STORAGE_KEY);
  assert.ok(raw);
  assert.ok(byteLength(raw) <= MAX_PER_MODEL_CONFIG_STORAGE_BYTES);
  assert.equal(
    loadPerModelConfig("Org/Model-39")?.chatTemplateOverride,
    template,
  );
  assert.equal(loadPerModelConfig("Org/Model-0"), null);
});

test("per-model config storage evicts safely across multiple GGUF variants", () => {
  const storage = installStorage();
  const template = "v".repeat(MAX_CHAT_TEMPLATE_LENGTH);
  const variants = ["Q4_K_M", "Q5_K_M", "Q8_0"] as const;

  for (let i = 0; i < 25; i += 1) {
    for (const variant of variants) {
      assert.equal(
        savePerModelConfig(`Org/Multi-${i}`, variant, {
          ...DEFAULT_PER_MODEL_CONFIG,
          chatTemplateOverride: template,
          kvCacheDtype: variant === "Q8_0" ? "q8_0" : "q4_0",
        }),
        true,
      );
    }
  }

  const raw = storage.getItem(STORAGE_KEY);
  assert.ok(raw);
  assert.ok(byteLength(raw) <= MAX_PER_MODEL_CONFIG_STORAGE_BYTES);
  for (const variant of variants) {
    assert.equal(
      loadPerModelConfig("Org/Multi-24", variant)?.chatTemplateOverride,
      template,
    );
  }
  assert.equal(loadPerModelConfig("Org/Multi-0", "Q4_K_M"), null);
  assert.equal(loadPerModelConfig("Org/Multi-0", "Q8_0"), null);
});

test("per-model config count eviction treats loaded entries as recently used", () => {
  installStorage();

  for (let i = 0; i < 500; i += 1) {
    assert.equal(
      savePerModelConfig(`Org/Model-${i}`, null, {
        ...DEFAULT_PER_MODEL_CONFIG,
        kvCacheDtype: "q4_0",
      }),
      true,
    );
  }

  assert.equal(loadPerModelConfig("Org/Model-0")?.kvCacheDtype, "q4_0");
  assert.equal(
    savePerModelConfig("Org/Model-500", null, {
      ...DEFAULT_PER_MODEL_CONFIG,
      kvCacheDtype: "q8_0",
    }),
    true,
  );

  assert.equal(loadPerModelConfig("Org/Model-0")?.kvCacheDtype, "q4_0");
  assert.equal(loadPerModelConfig("Org/Model-1"), null);
  assert.equal(loadPerModelConfig("Org/Model-500")?.kvCacheDtype, "q8_0");
});

test("chat template limit is enforced by UTF-8 bytes", () => {
  installStorage();
  const emojiTemplate = "😀".repeat(
    Math.floor(MAX_CHAT_TEMPLATE_BYTES / 4) + 1,
  );

  assert.ok(emojiTemplate.length <= MAX_CHAT_TEMPLATE_LENGTH);
  assert.ok(chatTemplateByteLength(emojiTemplate) > MAX_CHAT_TEMPLATE_BYTES);
  assert.equal(
    savePerModelConfig("Org/Emoji", null, {
      ...DEFAULT_PER_MODEL_CONFIG,
      chatTemplateOverride: emojiTemplate,
    }),
    false,
  );
  assert.equal(loadPerModelConfig("Org/Emoji"), null);
});

test("chat template byte clamp preserves whole code points", () => {
  assert.equal(clampChatTemplateToByteLimit("a😀b"), "a😀b");
  assert.equal(
    chatTemplateByteLength(clampChatTemplateToByteLimit("😀".repeat(20_000))),
    MAX_CHAT_TEMPLATE_BYTES,
  );
});

test("per-model config loads legacy GGUF variant keys case-insensitively", () => {
  const storage = installStorage();
  storage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      "Org/Foo::Q4_K_M": {
        ...DEFAULT_PER_MODEL_CONFIG,
        kvCacheDtype: "q4_0",
      },
    }),
  );

  assert.equal(loadPerModelConfig("org/foo", "q4_k_m")?.kvCacheDtype, "q4_0");
});

test("per-model config keeps model ids and variants with separators isolated", () => {
  const storage = installStorage();

  assert.equal(
    savePerModelConfig("/models/a::b", "Q4_K_M", {
      ...DEFAULT_PER_MODEL_CONFIG,
      kvCacheDtype: "q4_0",
    }),
    true,
  );
  assert.equal(
    savePerModelConfig("/models/a", "b::Q4_K_M", {
      ...DEFAULT_PER_MODEL_CONFIG,
      kvCacheDtype: "q8_0",
    }),
    true,
  );

  assert.equal(
    loadPerModelConfig("/models/a::b", "q4_k_m")?.kvCacheDtype,
    "q4_0",
  );
  assert.equal(
    loadPerModelConfig("/models/a", "b::q4_k_m")?.kvCacheDtype,
    "q8_0",
  );

  const keys = Object.keys(readStoredMap(storage));
  assert.equal(keys.length, 2);
  assert.equal(new Set(keys).size, 2);
  assert.equal(keys.every((key) => key.startsWith("v2:")), true);
});

test("per-model config loads legacy local path keys containing separators", () => {
  const storage = installStorage();
  storage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      "/models/a::b::Q4_K_M": {
        ...DEFAULT_PER_MODEL_CONFIG,
        kvCacheDtype: "q4_0",
      },
    }),
  );

  assert.equal(
    loadPerModelConfig("/models/a::b", "q4_k_m")?.kvCacheDtype,
    "q4_0",
  );
});

test("saving per-model config collapses legacy GGUF variant key casing", () => {
  const storage = installStorage();
  storage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      "Org/Foo::Q4_K_M": {
        ...DEFAULT_PER_MODEL_CONFIG,
        kvCacheDtype: "q4_0",
      },
    }),
  );

  assert.equal(
    savePerModelConfig("org/foo", "q4_k_m", {
      ...DEFAULT_PER_MODEL_CONFIG,
      kvCacheDtype: "q8_0",
    }),
    true,
  );

  const map = readStoredMap(storage);
  assert.deepEqual(Object.keys(map), [modelStorageKey("org/foo", "q4_k_m")]);
  assert.equal(loadPerModelConfig("Org/Foo", "Q4_K_M")?.kvCacheDtype, "q8_0");
});

test("deleting per-model config removes legacy GGUF variant key casing", () => {
  const storage = installStorage();
  storage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      "Org/Foo::Q4_K_M": {
        ...DEFAULT_PER_MODEL_CONFIG,
        kvCacheDtype: "q4_0",
      },
      "org/foo::q4_k_m": {
        ...DEFAULT_PER_MODEL_CONFIG,
        kvCacheDtype: "q8_0",
      },
      "Org/Foo::Q5_K_M": {
        ...DEFAULT_PER_MODEL_CONFIG,
        kvCacheDtype: "q5_1",
      },
    }),
  );

  deletePerModelConfig("org/foo", "q4_k_m");

  const map = readStoredMap(storage);
  assert.equal(Object.hasOwn(map, "Org/Foo::Q4_K_M"), false);
  assert.equal(Object.hasOwn(map, "org/foo::q4_k_m"), false);
  assert.equal(Object.hasOwn(map, "Org/Foo::Q5_K_M"), true);
});

test("saving per-model config refuses to overwrite newer schema entries", () => {
  const storage = installStorage();
  const key = modelStorageKey("Org/Future", null);
  storage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      [key]: {
        ...DEFAULT_PER_MODEL_CONFIG,
        version: 2,
        kvCacheDtype: "q4_0",
        futureField: "preserve",
      },
    }),
  );
  assert.equal(loadPerModelConfig("org/future")?.kvCacheDtype, "q4_0");
  const before = storage.getItem(STORAGE_KEY);

  assert.equal(
    savePerModelConfig("Org/Future", null, {
      ...DEFAULT_PER_MODEL_CONFIG,
      kvCacheDtype: "q8_0",
    }),
    false,
  );

  assert.equal(storage.getItem(STORAGE_KEY), before);
});
