// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("durable reload recovery owns captures and finishes live TPS", async () => {
  const source = await readFile(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  );
  const start = source.indexOf("function scheduleGenerationRecovery(");
  const end = source.indexOf("export async function ensureThreadRecord", start);
  const recovery = source.slice(start, end);

  const begin = recovery.indexOf(
    'runtime.beginThreadLiveTps(threadId, serverCancel, "")',
  );
  const follow = recovery.indexOf("followChatGenerationRun(runId", begin);
  const capture = recovery.indexOf("onMonitorId: captureMonitorId", follow);
  const finish = recovery.indexOf("store.finishThreadLiveTps(", capture);
  const clearOwner = recovery.indexOf("store.clearThreadServerCancel", finish);

  assert.ok(begin > 0, "recovery must create a request-owned TPS entry");
  assert.ok(
    follow > begin && capture > follow,
    "recovery must capture the durable stream monitor id",
  );
  assert.ok(
    finish > capture,
    "recovery must terminalize TPS in its finally path",
  );
  assert.ok(
    clearOwner > finish,
    "TPS must finish before the run owner is cleared",
  );
});
