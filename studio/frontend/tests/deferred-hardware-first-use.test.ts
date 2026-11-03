// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import { installLocalStorageFake, readSrc, registerBundlerResolver } from "./helpers/kit.ts";

register("./helpers/vite-env-loader.mjs", import.meta.url);
registerBundlerResolver();
const { store } = installLocalStorageFake();
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: { platform: "MacIntel", userAgent: "Macintosh" },
});

const DEFERRED = {
  chat_only: true,
  hardware_detecting: true,
  hardware_detection_deferred: true,
  version: "2026.1.1",
};
const DETECTING = { chat_only: true, hardware_detecting: true, version: "2026.1.1" };
const GPU = { device_type: "linux", chat_only: false, chat_only_reason: null, version: "2026.1.1" };
let replies: Array<Record<string, unknown>> = [];
let urls: string[] = [];
globalThis.fetch = (async (url: string) => {
  urls.push(url);
  const body = replies.shift() ?? DEFERRED;
  const response = { ok: true, json: async () => ({ ...body }), clone: () => response };
  return response;
}) as unknown as typeof fetch;

const { fetchDeviceType, forgetMeasuredVerdict, usePlatformStore } = await import("../src/config/env.ts");
const { routeNeedsHardware } = await import("../src/config/hardware-verdict.ts");

function reset() {
  store.set("unsloth_auth_token", "token");
  forgetMeasuredVerdict();
  usePlatformStore.setState({
    fetched: false,
    verdictStale: false,
    detectionDeferred: false,
    chatOnly: true,
    chatOnlyReason: null,
  });
  urls = [];
  replies = [];
}

test("idle health keeps capabilities unknown and preserves the first real detection wait", async () => {
  reset();
  await fetchDeviceType();
  let state = usePlatformStore.getState();
  assert.deepEqual(urls, ["/api/health"]);
  assert.equal(state.detectionDeferred, true);
  assert.equal(state.capabilitiesUnknown(), true);
  assert.equal(state.isTrainUnavailable() && !state.capabilitiesUnknown(), false);

  replies = [DETECTING, GPU];
  await fetchDeviceType({ detectHardware: true });
  state = usePlatformStore.getState();
  assert.deepEqual(urls, [
    "/api/health",
    "/api/health?detect_hardware=true",
    "/api/health?detect_hardware=true",
  ]);
  assert.equal(state.detectionDeferred, false);
  assert.equal(state.capabilitiesUnknown(), false);
  assert.equal(state.isTrainUnavailable(), false);
});

test("explicit detection bypasses a cached platform and honours an unsupported result", async () => {
  reset();
  usePlatformStore.setState({ fetched: true, detectionDeferred: true, chatOnly: false });
  replies = [{ ...GPU, chat_only: true, chat_only_reason: "no_torch" }];
  await fetchDeviceType({ detectHardware: true });
  const state = usePlatformStore.getState();
  assert.deepEqual(urls, ["/api/health?detect_hardware=true"]);
  assert.equal(state.capabilitiesUnknown(), false);
  assert.equal(state.isTrainUnavailable(), true);
  assert.equal(state.chatOnlyReason, "no_torch");
});

test("an unauthenticated navigation never requests a hardware probe", async () => {
  reset();
  store.delete("unsloth_auth_token");
  await fetchDeviceType({ detectHardware: true });
  assert.deepEqual(urls, ["/api/health"]);
});

test("only hardware-gated destinations request detection, including direct child links", () => {
  for (const path of ["/studio", "/studio/run", "/video", "/video/model", "/export"]) {
    assert.equal(routeNeedsHardware(path), true, path);
  }
  for (const path of ["/", "/chat", "/login", "/hub", "/projects", "/studio-other"]) {
    assert.equal(routeNeedsHardware(path), false, path);
  }
  assert.match(
    readSrc("app/routes/__root.tsx"),
    /detectHardware: !preload && routeNeedsHardware\(location\.pathname\)/,
    "hover preloads must not trigger an otherwise idle ML import",
  );
});

test("an older idle response cannot defer detection after a first-use request was accepted", async () => {
  reset();
  await fetchDeviceType();
  const originalFetch = globalThis.fetch;
  const originalNow = Date.now;
  let releaseIdle: (() => void) | undefined;
  let clock = originalNow();
  // Finish the first-use bounded wait immediately with an actively detecting reply.
  Date.now = () => (clock += 6000);
  globalThis.fetch = (async (url: string) => {
    const demand = url.includes("detect_hardware=true");
    if (!demand) await new Promise<void>((resolve) => { releaseIdle = resolve; });
    const body = demand ? DETECTING : DEFERRED;
    const response = { ok: true, json: async () => ({ ...body }), clone: () => response };
    return response;
  }) as unknown as typeof fetch;
  try {
    const idle = fetchDeviceType({ force: true });
    await fetchDeviceType({ detectHardware: true });
    assert.equal(usePlatformStore.getState().detectionDeferred, false);
    releaseIdle?.();
    await idle;
    assert.equal(usePlatformStore.getState().detectionDeferred, false);
    assert.equal(usePlatformStore.getState().capabilitiesUnknown(), true);
  } finally {
    globalThis.fetch = originalFetch;
    Date.now = originalNow;
  }
});

test("the real route guard keeps hover preloads cold and starts detection on navigation", async () => {
  reset();
  const rootSource = readSrc("app/routes/__root.tsx");
  const match = /beforeLoad: async \(\{ location, preload \}\) => \{([\s\S]*?)\n  \},\n  component: RootLayout/.exec(rootSource);
  assert.ok(match, "root route guard moved");
  const beforeLoad = new Function(
    "fetchDeviceType", "routeNeedsHardware", "usePlatformStore",
    "isChatOnlyAllowed", "waitsOutUnknownVerdict", "redirect",
    "return async ({ location, preload }) => {" + match[1] + "};",
  )(
    fetchDeviceType, routeNeedsHardware, usePlatformStore,
    () => true, () => true, () => { throw new Error("unexpected redirect"); },
  ) as (options: { location: { pathname: string }; preload: boolean }) => Promise<void>;

  await beforeLoad({ location: { pathname: "/studio" }, preload: true });
  assert.deepEqual(urls, ["/api/health"]);
  replies = [GPU];
  await beforeLoad({ location: { pathname: "/studio" }, preload: false });
  assert.deepEqual(urls, ["/api/health", "/api/health?detect_hardware=true"]);
  assert.equal(usePlatformStore.getState().capabilitiesUnknown(), false);
});

test("auto-load keeps cached non-GGUF models while detection is deferred", () => {
  reset();
  const adapterSource = readSrc("features/chat/api/chat-adapter.ts");
  const lift = (name: string) => {
    const match = new RegExp(`function ${name}\\([^)]*\\): boolean \\{([\\s\\S]*?)\\n\\}`).exec(adapterSource);
    assert.ok(match, `${name} moved`);
    return new Function("usePlatformStore", "isGgufLocalRow", "isMlxId", "row", match[1])
      .bind(null, usePlatformStore, (row: { path: string }) => row.path.endsWith(".gguf"), () => false);
  };
  const cachedModelsRun = lift("cachedModelsRunOnThisPlatform");
  const runsHere = lift("runsOnThisPlatform");
  const safetensors = { id: "org/model", path: "/models/org/model" };

  usePlatformStore.setState({ fetched: true, detectionDeferred: true, chatOnly: true, deviceType: "linux" });
  assert.equal(cachedModelsRun(), true);
  assert.equal(runsHere(safetensors), true);

  usePlatformStore.setState({ detectionDeferred: false });
  assert.equal(cachedModelsRun(), false);
  assert.equal(runsHere(safetensors), false);
  assert.equal(runsHere({ id: "org/model-GGUF", path: "/models/m.gguf" }), true);
});

test("a deferred reply names the server platform and is re-read until measured", async () => {
  reset();
  replies = [{ ...DEFERRED, device_type: "linux" }, GPU];
  await fetchDeviceType();
  let state = usePlatformStore.getState();
  assert.equal(state.deviceType, "linux");
  assert.equal(state.capabilitiesUnknown(), true);
  await fetchDeviceType();
  state = usePlatformStore.getState();
  assert.deepEqual(urls, ["/api/health", "/api/health"]);
  assert.equal(state.capabilitiesUnknown(), false);
});

test("a deferred verdict neither redirects nor blocks Train entry points on a guess", async () => {
  reset();
  const rootSource = readSrc("app/routes/__root.tsx");
  const match = /beforeLoad: async \(\{ location, preload \}\) => \{([\s\S]*?)\n  \},\n  component: RootLayout/.exec(rootSource);
  assert.ok(match, "root route guard moved");
  const beforeLoad = new Function(
    "fetchDeviceType", "routeNeedsHardware", "usePlatformStore",
    "isChatOnlyAllowed", "waitsOutUnknownVerdict", "redirect",
    "return async ({ location, preload }) => {" + match[1] + "};",
  )(
    fetchDeviceType, routeNeedsHardware, usePlatformStore,
    () => false, () => false, () => { throw new Error("redirected"); },
  ) as (options: { location: { pathname: string }; preload: boolean }) => Promise<void>;

  await beforeLoad({ location: { pathname: "/settings" }, preload: false });
  replies = [{ ...GPU, chat_only: true, chat_only_reason: "no_gpu" }];
  await fetchDeviceType({ force: true });
  await assert.rejects(beforeLoad({ location: { pathname: "/settings" }, preload: false }), /redirected/);

  assert.match(
    readSrc("features/settings/tabs/data-tab.tsx"),
    /s\.isTrainUnavailable\(\) && !s\.capabilitiesUnknown\(\)/,
  );
});

test("a desktop backend reconnect asks for detection on an open hardware page", () => {
  const reconnect = /if \(verdictStaleRef\.current\) \{[\s\S]*?\}, \[status, desktopAuthReady\]\);/.exec(
    readSrc("app/provider.tsx"),
  );
  assert.ok(reconnect, "the reconnect refresh moved");
  assert.match(reconnect[0], /force: true,[\s\S]*detectHardware: routeNeedsHardware\(pathnameRef\.current\)/);
});
