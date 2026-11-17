// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  clearDefaultChatModel,
  isDefaultChatModel,
  readDefaultChatModel,
  readDefaultChatModelLoadOnStartup,
  setDefaultChatModel,
  setDefaultChatModelLoadOnStartup,
} from "../src/features/chat/utils/default-chat-model.ts";

class MemoryStorage implements Storage {
  private readonly values = new Map<string, string>();

  get length() {
    return this.values.size;
  }

  clear() {
    this.values.clear();
  }

  getItem(key: string) {
    return this.values.get(key) ?? null;
  }

  key(index: number) {
    return [...this.values.keys()][index] ?? null;
  }

  removeItem(key: string) {
    this.values.delete(key);
  }

  setItem(key: string, value: string) {
    this.values.set(key, value);
  }
}

const storage = new MemoryStorage();
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: storage,
});

test("stores a default model and clears only the selected record", () => {
  storage.clear();
  const model = {
    id: "org/model",
    kind: "gguf" as const,
    ggufVariant: "Q4_K_M",
  };
  setDefaultChatModel(model);
  assert.deepEqual(readDefaultChatModel(), model);
  assert.equal(isDefaultChatModel(model), true);

  clearDefaultChatModel({
    id: "other/model",
    kind: "model",
    ggufVariant: null,
  });
  assert.deepEqual(readDefaultChatModel(), model);
  clearDefaultChatModel(model);
  assert.equal(readDefaultChatModel(), null);
});

test("startup loading is opt-in and can be toggled independently", () => {
  storage.clear();
  assert.equal(readDefaultChatModelLoadOnStartup(), false);
  setDefaultChatModelLoadOnStartup(true);
  assert.equal(readDefaultChatModelLoadOnStartup(), true);
  setDefaultChatModelLoadOnStartup(false);
  assert.equal(readDefaultChatModelLoadOnStartup(), false);
});
