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
const { resolveToolActivityOpen, syncToolActivityPreference } = await import(
  "../src/components/assistant-ui/tool-activity-open-state.ts"
);

const read = (path: string) => readFile(new URL(path, import.meta.url), "utf8");
const WHITESPACE = /\s+/g;

test("tool activity is collapsed by default", () => {
  assert.equal(
    useChatPreferencesStore.getInitialState().collapseToolActivityByDefault,
    true,
  );
});

test("older saved preferences inherit the collapsed default", async () => {
  const source = await read(
    "../src/features/chat/stores/chat-preferences-store.ts",
  );
  assert.ok(
    source
      .replace(WHITESPACE, " ")
      .includes(
        "collapseToolActivityByDefault: saved?.collapseToolActivityByDefault ?? true",
      ),
  );
});

test("manual expansion survives updates while activity is collapsed", () => {
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: true,
      collapseByDefault: true,
      previousCollapseByDefault: true,
      isRunning: false,
      hasText: true,
    }),
    true,
  );
});

test("enabling collapsed activity closes an already open card", () => {
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: true,
      collapseByDefault: true,
      previousCollapseByDefault: false,
      isRunning: true,
      hasText: false,
    }),
    false,
  );
});

test("disabling collapsed activity restores automatic visibility", () => {
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: false,
      collapseByDefault: false,
      previousCollapseByDefault: true,
      isRunning: true,
      hasText: false,
    }),
    true,
  );
  assert.equal(
    resolveToolActivityOpen({
      currentOpen: true,
      collapseByDefault: false,
      previousCollapseByDefault: false,
      isRunning: false,
      hasText: true,
    }),
    false,
  );
});

test("fallback cards react to live preference changes", () => {
  const manuallyOpen = {
    collapseByDefault: false,
    open: true,
  };
  const collapsed = syncToolActivityPreference(manuallyOpen, true, true);
  assert.deepEqual(collapsed, {
    collapseByDefault: true,
    open: false,
  });
  assert.deepEqual(syncToolActivityPreference(collapsed, false, true), {
    collapseByDefault: false,
    open: true,
  });
});

test("fallback cards preserve manual state until the preference changes", () => {
  const manuallyOpen = {
    collapseByDefault: true,
    open: true,
  };
  assert.equal(
    syncToolActivityPreference(manuallyOpen, true, true),
    manuallyOpen,
  );
});

test("disabling collapsed activity respects a closed fallback default", () => {
  assert.deepEqual(
    syncToolActivityPreference(
      { collapseByDefault: true, open: false },
      false,
      false,
    ),
    { collapseByDefault: false, open: false },
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
