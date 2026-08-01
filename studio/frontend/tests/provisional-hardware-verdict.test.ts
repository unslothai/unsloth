// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// /api/health answers before the backend has measured the host: until the background
// torch import lands, health reports its conservative pre-detection default,
// chat_only: true, with hardware_detecting set and no device_type.
//
// __root.tsx's beforeLoad awaits fetchDeviceType then redirects on isChatOnly(), so
// storing that provisional reply sends the first load to /chat with Train hidden on a
// machine that has GPUs.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { isDetectionDeferred, isProvisionalVerdict, resolveVerdict } = await import(
  "../src/config/hardware-verdict.ts"
);

const GPU_HOST = { chatOnly: false, chatOnlyReason: null };
const MAC_DEFAULT = { chatOnly: true, chatOnlyReason: null };

test("a detecting reply is provisional, a settled one is not", () => {
  assert.equal(isProvisionalVerdict({ hardware_detecting: true }), true);
  assert.equal(isProvisionalVerdict({ hardware_detecting: false }), false);
  assert.equal(
    isProvisionalVerdict({ chat_only: false }),
    false,
    "a reply with no hardware_detecting at all is a measured one",
  );
});

test("a provisional reply does not send a GPU host to chat-only", () => {
  const resolved = resolveVerdict(
    { chat_only: true, hardware_detecting: true },
    GPU_HOST,
  );
  assert.equal(
    resolved.chatOnly,
    false,
    "the provisional chat_only was stored; beforeLoad would redirect a GPU host to /chat",
  );
});

test("a provisional reply does not clear a reason the UI is explaining", () => {
  const resolved = resolveVerdict(
    { chat_only: true, hardware_detecting: true },
    { chatOnly: true, chatOnlyReason: "mlx_unavailable" },
  );
  assert.equal(
    resolved.chatOnlyReason,
    "mlx_unavailable",
    "the sidebar recovery poll only runs while it reads mlx_unavailable",
  );
});

test("a measured chat-only verdict is still honoured", () => {
  const resolved = resolveVerdict(
    { chat_only: true, chat_only_reason: "mlx_unavailable" },
    GPU_HOST,
  );
  assert.equal(resolved.chatOnly, true, "a real chat-only host was let into Train");
  assert.equal(resolved.chatOnlyReason, "mlx_unavailable");
});

test("a measured GPU verdict clears a chat-only default", () => {
  const resolved = resolveVerdict(
    { chat_only: false, chat_only_reason: null },
    MAC_DEFAULT,
  );
  assert.equal(
    resolved.chatOnly,
    false,
    "keeping the previous value has to stop once a measurement arrives",
  );
});

test("a measured reply with no chat_only field is not chat-only", () => {
  assert.equal(resolveVerdict({}, MAC_DEFAULT).chatOnly, false);
});

test("a deferred verdict is provisional but must not be waited on", () => {
  const deferred = { chat_only: true, hardware_detecting: true, hardware_detection_deferred: true };
  assert.equal(
    isProvisionalVerdict(deferred),
    true,
    "a deferred reply is still not a measurement, so it must not be stored",
  );
  assert.equal(
    isDetectionDeferred(deferred),
    true,
    "the kill switch stops anything settling, so the re-read loop must give up",
  );
});

test("a deferred verdict falls back to the backend's conservative default", () => {
  // Nothing settles while the kill switch is on, so keeping the previous value would
  // leave the browser-platform default (chatOnly false off macOS) in place all session
  // and offer Train on a CPU-only Linux host.
  const deferred = {
    chat_only: true,
    hardware_detecting: true,
    hardware_detection_deferred: true,
  };
  assert.equal(
    resolveVerdict(deferred, { chatOnly: false, chatOnlyReason: null }).chatOnly,
    true,
    "a never-settling reply left the optimistic platform default in place",
  );
});

test("a deferred verdict does not clear a reason the UI is explaining", () => {
  const deferred = {
    chat_only: true,
    hardware_detecting: true,
    hardware_detection_deferred: true,
  };
  assert.equal(
    resolveVerdict(deferred, { chatOnly: true, chatOnlyReason: "mlx_unavailable" })
      .chatOnlyReason,
    "mlx_unavailable",
    "the sidebar recovery poll only runs while it reads mlx_unavailable",
  );
});

test("an ordinary provisional reply is still not treated as deferred", () => {
  // The conservative fallback stays scoped to the kill switch: a warm-window reply
  // settles in ~1-2s, and taking its chat_only would send a GPU host to /chat.
  assert.equal(
    resolveVerdict({ chat_only: true, hardware_detecting: true }, GPU_HOST).chatOnly,
    false,
  );
});

test("an actively detecting reply is not deferred", () => {
  assert.equal(
    isDetectionDeferred({ chat_only: true, hardware_detecting: true }),
    false,
    "an ordinary warm-window reply must still be waited on",
  );
});

// The bounded re-read in config/env.ts is spent at most once per page load: a slow host
// leaves the reply provisional and `fetched` false, so an unguarded loop would repeat the
// full wait on every navigation. Asserted on source, since env.ts reaches import.meta.env
// through api-base.ts and cannot be imported outside vite.
test("the bounded hardware wait is spent at most once per page load", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/config/env.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /let hardwareWaitSpent = false/,
    "no once-per-load latch: every navigation can repeat the full wait",
  );
  assert.match(
    src,
    /hardwareWaitSpent\s*\n?\s*\?\s*0/,
    "the latch does not zero the deadline, so the wait is not actually skipped",
  );
  assert.match(
    src,
    /hardwareWaitSpent = true/,
    "the latch is never set, so it can never take effect",
  );
});

// A provisional reply omits device_type. A forced refresh in that window must not fall
// back to the browser platform: on WSL, SSH or any remote session that relabels the host
// as local, changing model filtering, paths and install commands.
test("a provisional forced refresh keeps the server-reported platform", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(
    new URL("../src/config/env.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /const keepPlatform =\s*\n?\s*data\.device_type === undefined && previous\.fetched/,
    "no guard: a provisional forced refresh overwrites the authoritative platform",
  );
  assert.match(
    src,
    /keepPlatform \? previous\.deviceType : detectLocalPlatform\(\)/,
    "the guard does not actually keep the stored platform",
  );
  assert.match(
    src,
    /fetched: data\.device_type !== undefined \|\| keepPlatform/,
    "fetched drops to false, so the authoritative platform is not treated as held",
  );
});
