// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import type * as SystemHooks from "../src/hooks/use-system.ts";
import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

register("./helpers/vite-env-loader.mjs", import.meta.url);
registerBundlerResolver();
const { store } = installLocalStorageFake();
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: { platform: "Win32", userAgent: "Windows" },
});
const platform = await import("../src/config/env.ts");
const { usePlatformStore, forgetMeasuredVerdict, fetchDeviceType } = platform;
const DEFERRED = { chat_only: true, hardware_detecting: true, hardware_detection_deferred: true, version: "test" };
const CPU = { device_type: "linux", chat_only: true, chat_only_reason: "no_gpu", version: "test" };
const MLX = { device_type: "mac", apple_silicon: true, chat_only: false, chat_only_reason: null, version: "test" };

function systemApi(fetch: (url: string) => Promise<Response>) {
  return loadWithStubs<typeof SystemHooks>(
    new URL("../src/hooks/use-system.ts", import.meta.url),
    {
      react: {},
      "@/features/auth": { authFetch: fetch },
      "@/config/env": platform,
    },
    { relativePassthrough: true },
  );
}

function reset(chatOnly = true) {
  store.set("unsloth_auth_token", "test-token");
  forgetMeasuredVerdict();
  usePlatformStore.setState({
    deviceType: "windows", fetched: true, verdictStale: false,
    detectionDeferred: true, chatOnly, chatOnlyReason: null,
  });
}

async function settle() {
  for (let i = 0; i < 5; i++) await new Promise<void>((resolve) => setImmediate(resolve));
}

for (const [name, health, initialChatOnly, backend] of [
  ["CPU", CPU, false, "cpu"],
  ["MLX", MLX, true, "mlx"],
] as const) {
  test(`first successful system read replaces the deferred ${name} picker verdict`, async () => {
    reset(initialChatOnly);
    const urls: string[] = [];
    globalThis.fetch = (async (url: string) => {
      urls.push(url);
      return Response.json(health);
    }) as typeof fetch;
    const system = systemApi(async (url) => {
      urls.push(url);
      return Response.json({ device_backend: backend, gpu: { available: backend === "mlx", devices: [] } });
    });
    await settle();
    assert.deepEqual(urls, [], "a closed/unused picker must not start discovery");
    const info = await system.fetchSystemInfo();
    await settle();
    assert.equal(info?.device_backend, backend);
    assert.deepEqual(urls, ["/api/system", "/api/health"]);
    const state = usePlatformStore.getState();
    assert.equal(state.capabilitiesUnknown(), false);
    assert.equal(state.detectionDeferred, false);
    assert.equal(state.isChatOnly(), health.chat_only);
    assert.equal(state.deviceType, health.device_type);
    await system.fetchSystemInfo();
    assert.equal(urls.length, 2, "a measured cached verdict needs no extra health read");
  });
}

test("failed system reads do not request a platform refresh", async () => {
  reset();
  let healthCalls = 0;
  globalThis.fetch = (async () => { healthCalls++; return Response.json(CPU); }) as typeof fetch;
  const system = systemApi(async () => new Response(null, { status: 503 }));
  assert.equal(await system.fetchSystemInfo(), null);
  await settle();
  assert.equal(healthCalls, 0);
  assert.equal(usePlatformStore.getState().detectionDeferred, true);
});

test("system readers share one pending verdict refresh without delaying GPU results", async () => {
  reset();
  let healthCalls = 0;
  let release: ((response: Response) => void) | undefined;
  globalThis.fetch = (async () => {
    healthCalls++;
    return new Promise<Response>((resolve) => { release = resolve; });
  }) as typeof fetch;
  const system = systemApi(async () => Response.json({ device_backend: "cpu" }));
  assert.equal((await system.fetchSystemInfo())?.device_backend, "cpu");
  assert.equal((await system.fetchSystemInfo({ force: true }))?.device_backend, "cpu");
  assert.equal((await system.fetchSystemInfo())?.device_backend, "cpu");
  assert.equal(healthCalls, 1);
  release?.(Response.json(CPU));
  await settle();
  assert.equal(usePlatformStore.getState().capabilitiesUnknown(), false);
});

test("a failed verdict refresh leaves GPU data usable and retries on the next cached demand", async () => {
  reset();
  let healthCalls = 0;
  let systemCalls = 0;
  globalThis.fetch = (async () => {
    healthCalls++;
    return healthCalls === 1 ? new Response(null, { status: 503 }) : Response.json(CPU);
  }) as typeof fetch;
  const system = systemApi(async () => { systemCalls++; return Response.json({ device_backend: "cpu" }); });
  assert.equal((await system.fetchSystemInfo())?.device_backend, "cpu");
  await settle();
  assert.equal(usePlatformStore.getState().detectionDeferred, true);
  assert.equal((await system.fetchSystemInfo())?.device_backend, "cpu");
  await settle();
  assert.equal(systemCalls, 1);
  assert.equal(healthCalls, 2);
  assert.equal(usePlatformStore.getState().capabilitiesUnknown(), false);
});

test("an earlier deferred health response cannot undo a system-triggered settled verdict", async () => {
  reset();
  let healthCalls = 0;
  let release: ((response: Response) => void) | undefined;
  globalThis.fetch = (async () => {
    healthCalls++;
    if (healthCalls === 1) return new Promise<Response>((resolve) => { release = resolve; });
    return Response.json(MLX);
  }) as typeof fetch;
  const oldIdleRead = fetchDeviceType({ force: true });
  const system = systemApi(async () => Response.json({ device_backend: "mlx" }));
  await system.fetchSystemInfo();
  await settle();
  assert.equal(usePlatformStore.getState().isChatOnly(), false);
  release?.(Response.json(DEFERRED));
  await oldIdleRead;
  assert.equal(usePlatformStore.getState().detectionDeferred, false);
  assert.equal(usePlatformStore.getState().isChatOnly(), false);
});

test("a new backend may defer again after the previous backend was measured", async () => {
  reset();
  globalThis.fetch = (async () => Response.json(MLX)) as typeof fetch;
  const system = systemApi(async () => Response.json({ device_backend: "mlx" }));
  await system.fetchSystemInfo();
  await settle();
  assert.equal(usePlatformStore.getState().capabilitiesUnknown(), false);
  forgetMeasuredVerdict();
  globalThis.fetch = (async () => Response.json(DEFERRED)) as typeof fetch;
  await fetchDeviceType({ force: true });
  assert.equal(usePlatformStore.getState().detectionDeferred, true);
  assert.equal(usePlatformStore.getState().capabilitiesUnknown(), true);
});

test("a pending refresh from a replaced backend cannot block or overwrite its successor", async () => {
  reset();
  let oldRelease: ((response: Response) => void) | undefined;
  let healthCalls = 0;
  globalThis.fetch = (async () => {
    healthCalls++;
    if (healthCalls === 1) return new Promise<Response>((resolve) => { oldRelease = resolve; });
    return Response.json(MLX);
  }) as typeof fetch;
  const system = systemApi(async () => Response.json({ device_backend: "mlx" }));
  await system.fetchSystemInfo();
  forgetMeasuredVerdict();
  await system.fetchSystemInfo({ force: true });
  await settle();
  assert.equal(healthCalls, 2);
  assert.equal(usePlatformStore.getState().isChatOnly(), false);
  oldRelease?.(Response.json(CPU));
  await settle();
  assert.equal(usePlatformStore.getState().isChatOnly(), false);
  assert.equal(usePlatformStore.getState().deviceType, "mac");
});

test("cached picker demand settles a backend restarted with warmup disabled", async () => {
  reset();
  let detected = true;
  const requests: string[] = [];
  globalThis.fetch = (async (url: string) => {
    requests.push(url);
    if (url.includes("detect_hardware=true")) detected = true;
    return Response.json(detected ? MLX : DEFERRED);
  }) as typeof fetch;
  const system = systemApi(async (url) => {
    requests.push(url);
    detected = true;
    return Response.json({ device_backend: "mlx", gpu: { available: true, devices: [] } });
  });
  await system.fetchSystemInfo();
  await settle();
  assert.equal(usePlatformStore.getState().capabilitiesUnknown(), false);

  forgetMeasuredVerdict();
  detected = false;
  await fetchDeviceType({ force: true });
  assert.equal(usePlatformStore.getState().detectionDeferred, true);
  const idleRequests = [...requests];
  await settle();
  assert.deepEqual(requests, idleRequests, "idle restart must not start discovery");

  await Promise.all([system.fetchSystemInfo(), system.fetchSystemInfo()]);
  await settle();
  assert.equal(usePlatformStore.getState().capabilitiesUnknown(), false,
    "reopening a cached picker must measure the new backend: " + requests.join(", "));
  assert.equal(usePlatformStore.getState().detectionDeferred, false);
  assert.equal(usePlatformStore.getState().isChatOnly(), false);
  assert.equal(requests.filter((url) => url.includes("detect_hardware=true")).length, 1,
    "concurrent cached demands share one hardware refresh");
  const settledRequests = [...requests];
  await system.fetchSystemInfo();
  await settle();
  assert.deepEqual(requests, settledRequests, "a settled cache hit needs no new health read");
});
