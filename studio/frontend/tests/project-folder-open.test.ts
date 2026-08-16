// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function source(relative: string): string {
  return readFileSync(
    fileURLToPath(new URL(relative, import.meta.url)),
    "utf8",
  );
}

test("desktop folder picker reaches the signed folder-project endpoint", () => {
  const nativeApi = source("../src/features/native-intents/api.ts");
  const chatApi = source("../src/features/chat/api/chat-api.ts");
  const hook = source("../src/features/chat/hooks/use-open-project-folder.ts");

  assert.match(
    nativeApi,
    /invokeNative<[^>]+>\(\s*"pick_native_project_folder"/,
  );
  assert.match(chatApi, /authFetch\("\/api\/chat\/projects\/open-folder"/);
  assert.match(hook, /await pickNativeProjectFolder\(\)/);
  assert.match(hook, /await openChatProjectFromFolder\(selected\.token\)/);
});

test("all project entry points expose Open folder on desktop", () => {
  const menuSources = [
    source("../src/features/chat/shared-composer.tsx"),
    source("../src/components/assistant-ui/thread.tsx"),
    source("../src/features/chat/projects-page.tsx"),
  ];

  for (const menu of menuSources) {
    assert.match(menu, /isTauri[\s\S]*Open folder/);
    assert.match(menu, /openProjectFolder\(\)/);
  }
});

test("folder-backed projects never offer deleting the selected folder", () => {
  const sidebar = source("../src/components/app-sidebar.tsx");
  assert.match(sidebar, /target\.project\.workspaceKind !== "folder"/);
  assert.match(sidebar, /project\.workspaceKind !== "folder"/);
});
