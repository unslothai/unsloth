// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (relativePath: string): string =>
  readFileSync(new URL(relativePath, import.meta.url), "utf8");

test("Hub progress requests carry the request-scoped token", () => {
  const api = read("../src/features/chat/api/chat-api.ts");
  for (const name of [
    "getGgufDownloadProgress",
    "getDownloadProgress",
    "getDatasetDownloadProgress",
  ]) {
    const start = api.indexOf(`export async function ${name}`);
    assert.notEqual(start, -1, `${name} is missing`);
    const next = api.indexOf("\nexport ", start + 1);
    const body = api.slice(start, next === -1 ? undefined : next);
    assert.match(body, /hfToken\?: string \| null/);
    assert.match(body, /headers: hubTokenHeader\(hfToken\)/);
  }
});

test("chat and training progress retain the prepared token", () => {
  const chatRuntime = read(
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
  );
  assert.match(
    chatRuntime,
    /const preparedToken = await prepareHfTokenForUse\(hfToken\)/,
  );
  assert.match(
    chatRuntime,
    /getGgufDownloadProgress\([\s\S]*expectedBytes,[\s\S]*hfToken,/,
  );
  assert.match(chatRuntime, /getDownloadProgress\(modelId, hfToken\)/);

  const overlay = read("../src/features/studio/training-start-overlay.tsx");
  assert.match(overlay, /startHfToken/);
  assert.match(overlay, /getDownloadProgress\(repoId, hfToken\)/);
  assert.match(overlay, /getDatasetDownloadProgress\(repoId, hfToken\)/);
});
