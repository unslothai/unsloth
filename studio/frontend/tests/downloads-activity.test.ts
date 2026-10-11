// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as DownloadsActivity from "../src/lib/downloads-activity.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

test("the desktop quit warning sees a download from any source, sent once per change", async () => {
  const sent: boolean[] = [];
  const activity = loadWithStubs<typeof DownloadsActivity>(
    new URL("../src/lib/downloads-activity.ts", import.meta.url),
    {
      "@/lib/api-base": { isTauri: true },
      "@tauri-apps/api/core": {
        invoke: async (
          command: string,
          args: { kind: string; active: boolean },
        ) => {
          assert.equal(command, "set_renderer_activity");
          assert.equal(args.kind, "downloads");
          sent.push(args.active);
        },
      },
    },
  );
  const flush = () => new Promise((resolve) => setImmediate(resolve));

  activity.reportDownloadsActive("hub", false);
  activity.reportDownloadsActive("npu", true);
  activity.reportDownloadsActive("hub", true);
  // A Hub download ending while an NPU pull runs must not clear the warning.
  activity.reportDownloadsActive("hub", false);
  activity.reportDownloadsActive("npu", false);
  await flush();
  assert.deepEqual(sent, [false, true, false]);
});
