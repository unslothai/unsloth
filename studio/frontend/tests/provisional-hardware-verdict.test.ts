// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// /api/health answers before the backend has measured the host: startup hands
// the torch import to a background thread, and until it lands health reports its
// conservative pre-detection default, chat_only: true, with hardware_detecting
// set and no device_type. Measured on a 4-GPU host with a cold launch: the first
// reply is chat_only: true, settling to false about a second later.
//
// __root.tsx's beforeLoad awaits fetchDeviceType and then redirects on
// isChatOnly(), so storing that provisional reply sends the first load to /chat
// with Train hidden on a machine that has GPUs.

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
  assert.equal(
    resolveVerdict(deferred, GPU_HOST).chatOnly,
    false,
    "a deferred reply still must not send a GPU host to chat-only",
  );
});

test("an actively detecting reply is not deferred", () => {
  assert.equal(
    isDetectionDeferred({ chat_only: true, hardware_detecting: true }),
    false,
    "an ordinary warm-window reply must still be waited on",
  );
});
