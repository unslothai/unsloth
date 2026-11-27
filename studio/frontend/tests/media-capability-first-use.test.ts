// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as SystemHooks from "../src/hooks/use-system.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import { readSrc } from "./helpers/kit.ts";

function harness() {
  const requests: { url: string; resolve: (response: Response) => void }[] = [];
  const system = loadWithStubs<typeof SystemHooks>(
    new URL("../src/hooks/use-system.ts", import.meta.url),
    {
      react: {},
      "@/config/env": { refreshDeviceTypeAfterSystemInfo: () => {} },
      "@/features/auth": {
        authFetch: (url: string) => new Promise<Response>((resolve) => requests.push({ url, resolve })),
      },
    },
    { relativePassthrough: true },
  );
  return { system, requests };
}

test("a media probe follows an ordinary in-flight system read and publishes its capabilities", async () => {
  const { system, requests } = harness();
  const ordinary = system.fetchSystemInfo();
  const media = system.fetchSystemInfo({ probeMedia: true });
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "/api/system");
  requests[0].resolve(Response.json({ dense_quant_supported: false }));
  await ordinary;
  await Promise.resolve();
  assert.equal(requests[1].url, "/api/system?probe_media=true");
  requests[1].resolve(Response.json({ dense_quant_supported: true, quantised_streaming: true }));
  assert.equal((await media)?.dense_quant_supported, true);
  assert.equal((await system.fetchSystemInfo())?.quantised_streaming, true);
  assert.equal(requests.length, 2);
});

test("media discovery bypasses the cold cache and concurrent media requests share its probe", async () => {
  const { system, requests } = harness();
  const initial = system.fetchSystemInfo();
  requests[0].resolve(Response.json({ dense_quant_supported: false }));
  await initial;
  const first = system.fetchSystemInfo({ probeMedia: true });
  const second = system.fetchSystemInfo({ probeMedia: true });
  assert.equal(requests.length, 2);
  assert.equal(requests[1].url, "/api/system?probe_media=true");
  requests[1].resolve(Response.json({ dense_quant_supported: true }));
  const [a, b] = await Promise.all([first, second]);
  assert.equal(a, b);
  assert.equal(a?.dense_quant_supported, true);
});

test("a failed media probe can retry without poisoning ordinary system reads", async () => {
  const { system, requests } = harness();
  const first = system.fetchSystemInfo({ probeMedia: true });
  requests[0].resolve(new Response(null, { status: 503 }));
  assert.equal(await first, null);
  const next = system.fetchSystemInfo();
  assert.equal(requests[1].url, "/api/system");
  requests[1].resolve(Response.json({ dense_quant_supported: false }));
  assert.equal((await next)?.dense_quant_supported, false);
});

test("the picker requests media capabilities only while open for image or video tasks", () => {
  const source = readSrc("features/model-picker/components/model-selector/pickers.tsx");
  const taskNames = (name: string): string[] => {
    const match = new RegExp("export const " + name + " = \\[([\\s\\S]*?)\\] as const;").exec(source);
    assert.ok(match, name);
    return [...match[1].matchAll(/"([^"]+)"/g)].map((entry) => entry[1]);
  };
  const diffusionTasks = new Set([...taskNames("IMAGE_GEN_TASKS"), ...taskNames("VIDEO_GEN_TASKS")]);
  const decision = /const diffusionLoad = useMemo\(\(\) => \{([\s\S]*?)\}, \[task\]\);/.exec(source);
  assert.ok(decision, "picker task decision moved");
  const isMedia = new Function("task", "DIFFUSION_TASKS", decision[1]);
  const effect = /useEffect\(\(\) => \{([^{}]*?if \(open && diffusionLoad\) void fetchSystemInfo\(\{ probeMedia: true \}\);)\s*\}, \[open, diffusionLoad\]\);/.exec(source);
  assert.ok(effect, "picker capability effect moved");
  const runEffect = new Function("open", "diffusionLoad", "fetchSystemInfo", effect[1]);
  const calls: unknown[] = [];
  const fetch = (options: unknown) => { calls.push(options); };

  for (const task of ["text-to-image", "text-to-video"]) {
    runEffect(false, isMedia(task, diffusionTasks), fetch);
  }
  for (const task of [undefined, "text-generation", "text-to-speech", "text-to-audio"]) {
    runEffect(true, isMedia(task, diffusionTasks), fetch);
  }
  assert.equal(calls.length, 0);
  runEffect(true, isMedia("text-to-image", diffusionTasks), fetch);
  runEffect(true, isMedia(["text-to-video", "image-to-video"], diffusionTasks), fetch);
  assert.deepEqual(calls, [{ probeMedia: true }, { probeMedia: true }]);
  assert.match(
    readSrc("features/model-picker/components/model-selector.tsx"),
    /<HubModelPicker\s+open=\{open\}/,
  );
});
