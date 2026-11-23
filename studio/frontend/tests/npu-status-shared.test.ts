// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as Hook from "../src/features/npu/use-npu-status.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

// Each "component" gets its own state setter; effects run at mount.
function fakeReact(seen: unknown[][]) {
  return {
    useState: (initial: unknown) => {
      const mine: unknown[] = [initial];
      seen.push(mine);
      return [initial, (next: unknown) => mine.push(next)];
    },
    useEffect: (effect: () => void) => {
      effect();
    },
  };
}

test("enabling the NPU in one view updates every other mounted view", async () => {
  const seen: unknown[][] = [];
  let calls = 0;
  const { useNpuStatus } = loadWithStubs<typeof Hook>(
    new URL("../src/features/npu/use-npu-status.ts", import.meta.url),
    {
      react: fakeReact(seen),
      "./api": {
        getNpuStatus: async () => {
          calls += 1;
          return { supported: true, ready: false };
        },
      },
    },
  );
  const [, setFromHub] = useNpuStatus();
  useNpuStatus(false);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(calls, 1);
  setFromHub({ supported: true, ready: true } as never);
  const [hub, monitor] = seen;
  assert.deepEqual(hub.at(-1), { supported: true, ready: true });
  assert.deepEqual(monitor.at(-1), { supported: true, ready: true });
});
