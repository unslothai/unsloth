// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import { readLastPrompt, saveLastPrompt } from "../src/lib/last-prompt.ts";
import { WORKFLOW_EXAMPLE_PROMPTS, WORKFLOW_TABS } from "../src/features/images/workflows.ts";

function withStorage(storage: unknown, run: () => void) {
  const prev = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", { value: storage, configurable: true });
  try {
    run();
  } finally {
    if (prev) Object.defineProperty(globalThis, "localStorage", prev);
    else delete (globalThis as { localStorage?: unknown }).localStorage;
  }
}

test("the last prompt survives a reload and falls back to the example", () => {
  const store = new Map<string, string>();
  const storage = {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
  };
  withStorage(storage, () => {
    assert.equal(readLastPrompt("images:edit", "example"), "example");
    saveLastPrompt("images:edit", "make it blue");
    assert.equal(readLastPrompt("images:edit", "example"), "make it blue");
    // Keys are separate per page and workflow.
    assert.equal(readLastPrompt("images:create", "example"), "example");
  });
});

test("unavailable storage still yields the example", () => {
  const broken = {
    getItem: () => {
      throw new Error("denied");
    },
    setItem: () => {
      throw new Error("denied");
    },
  };
  withStorage(broken, () => {
    saveLastPrompt("video", "x");
    assert.equal(readLastPrompt("video", "example"), "example");
  });
});

test("every image workflow has its own short example prompt", () => {
  const examples = WORKFLOW_TABS.map(({ id }) => WORKFLOW_EXAMPLE_PROMPTS[id]);
  assert.equal(new Set(examples).size, WORKFLOW_TABS.length);
  for (const example of examples) {
    assert.ok(example.length > 0 && example.length < 200);
    assert.ok(!example.includes("\u2014"));
  }
});
