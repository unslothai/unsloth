// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);

const read = (path: string) => readFile(new URL(path, import.meta.url), "utf8");
const WHITESPACE = /\s+/g;

test("tool activity keeps its existing visible default", () => {
  assert.equal(
    useChatPreferencesStore.getInitialState().collapseToolActivityByDefault,
    false,
  );
});

test("older saved preferences keep tool activity visible", async () => {
  const source = await read(
    "../src/features/chat/stores/chat-preferences-store.ts",
  );
  assert.ok(
    source
      .replace(WHITESPACE, " ")
      .includes(
        "collapseToolActivityByDefault: saved?.collapseToolActivityByDefault ?? false",
      ),
  );
});

test("every automatically opening text tool follows the preference", async () => {
  for (const file of [
    "../src/components/assistant-ui/tool-fallback.tsx",
    "../src/components/assistant-ui/tool-group.tsx",
    "../src/components/assistant-ui/tool-ui-python.tsx",
    "../src/components/assistant-ui/use-tool-activity-open.ts",
  ]) {
    const source = await read(file);
    assert.ok(
      source.includes("collapseToolActivityByDefault"),
      `${file} can still open tool text without reading the preference`,
    );
  }

  for (const file of [
    "../src/components/assistant-ui/tool-ui-code-execution.tsx",
    "../src/components/assistant-ui/tool-ui-knowledge-base.tsx",
    "../src/components/assistant-ui/tool-ui-web-search.tsx",
  ]) {
    const source = await read(file);
    assert.ok(
      source.includes("useToolActivityOpen(isRunning, hasText)"),
      `${file} bypasses the shared automatic visibility policy`,
    );
  }
});

test("approval prompts still force grouped tools open", async () => {
  const source = await read("../src/components/assistant-ui/tool-group.tsx");
  assert.ok(
    source
      .replace(WHITESPACE, " ")
      .includes(
        "const forceOpen = hasPendingConfirmation || (!collapseByDefault",
      ),
  );
});

test("collapsed Python activity hides the script until the row is expanded", async () => {
  const source = await read(
    "../src/components/assistant-ui/tool-ui-python.tsx",
  );
  assert.ok(source.includes("{!collapseByDefault && scriptCell}"));
  assert.ok(
    source
      .replace(WHITESPACE, " ")
      .includes("<ToolFallbackContent> {collapseByDefault && scriptCell}"),
  );
});
