// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// remote-access-section.tsx pulls in the router, motion and hugeicons, so it
// cannot be imported here. These drive the pure state helpers it consumes.
import {
  type ApiRemoteAccessStatus,
  normalizeRemoteAccessStatus,
  remoteAccessAutoStartReadOnly,
  remoteAccessBlockMessage,
  remoteAccessPollDelay,
  remoteAccessSelfStopPoll,
  remoteAccessStopDisconnectsOrigin,
  remoteApiOrigin,
} from "../src/features/settings/api/remote-access-state.ts";

const TUNNEL = "https://calm-otter-review.trycloudflare.com";

function apiStatus(
  over: Partial<ApiRemoteAccessStatus> = {},
): ApiRemoteAccessStatus {
  return {
    state: "off",
    // biome-ignore lint/style/useNamingConvention: API schema
    auto_start: false,
    // biome-ignore lint/style/useNamingConvention: API schema
    default_auto_start: false,
    available: true,
    // biome-ignore lint/style/useNamingConvention: API schema
    can_start: true,
    // biome-ignore lint/style/useNamingConvention: API schema
    can_stop: false,
    // biome-ignore lint/style/useNamingConvention: API schema
    streaming_supported: true,
    ...over,
  };
}

// ── normalizeRemoteAccessStatus ──

test("normalize maps every snake_case field onto its camelCase name", () => {
  const s = normalizeRemoteAccessStatus(
    apiStatus({
      state: "online",
      url: TUNNEL,
      error: null,
      // biome-ignore lint/style/useNamingConvention: API schema
      auto_start: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      default_auto_start: true,
      available: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      managed_by: "settings",
      // biome-ignore lint/style/useNamingConvention: API schema
      can_start: false,
      // biome-ignore lint/style/useNamingConvention: API schema
      can_stop: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      block_reason: null,
      // biome-ignore lint/style/useNamingConvention: API schema
      password_pending: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      streaming_supported: true,
    }),
  );
  assert.deepEqual(s, {
    state: "online",
    url: TUNNEL,
    error: null,
    autoStart: true,
    defaultAutoStart: true,
    available: true,
    managedBy: "settings",
    canStart: false,
    canStop: true,
    blockReason: null,
    passwordPending: true,
    streamingSupported: true,
  });
  // No field may normalize to undefined -- a wrong key would silently do so.
  for (const [k, v] of Object.entries(s)) {
    assert.notEqual(v, undefined, `${k} normalized to undefined`);
  }
});

test("normalize defaults the optional fields an older backend may omit", () => {
  const s = normalizeRemoteAccessStatus(apiStatus());
  assert.equal(s.url, null);
  assert.equal(s.error, null);
  assert.equal(s.managedBy, null);
  assert.equal(s.blockReason, null);
  assert.equal(s.passwordPending, false);
});

test("passwordPending is strict: only a real true counts", () => {
  for (const raw of [undefined, null, 0, "", "true", 1]) {
    const s = normalizeRemoteAccessStatus(
      // biome-ignore lint/style/useNamingConvention: API schema
      apiStatus({ password_pending: raw as never }),
    );
    assert.equal(
      s.passwordPending,
      false,
      `password_pending=${JSON.stringify(raw)} must not be truthy`,
    );
  }
  assert.equal(
    // biome-ignore lint/style/useNamingConvention: API schema
    normalizeRemoteAccessStatus(apiStatus({ password_pending: true }))
      .passwordPending,
    true,
  );
});

// ── remoteAccessPollDelay ──

test("poll delay is fast only while a transition is in flight", () => {
  assert.equal(remoteAccessPollDelay(null), 5000);
  for (const state of ["starting", "stopping"] as const) {
    assert.equal(
      remoteAccessPollDelay(normalizeRemoteAccessStatus(apiStatus({ state }))),
      1000,
    );
  }
  for (const state of ["off", "online", "error"] as const) {
    assert.equal(
      remoteAccessPollDelay(normalizeRemoteAccessStatus(apiStatus({ state }))),
      5000,
    );
  }
});

// ── remoteApiOrigin ──

test("remote origin prefers the live tunnel and falls back to the local one", () => {
  assert.equal(remoteApiOrigin(TUNNEL, "http://127.0.0.1:8888"), TUNNEL);
  assert.equal(
    remoteApiOrigin(null, "http://127.0.0.1:8888"),
    "http://127.0.0.1:8888",
  );
  // An empty string is a real origin ("" from a non-browser context), not a miss.
  assert.equal(remoteApiOrigin(null, ""), "");
});

// ── remoteAccessAutoStartReadOnly ──

test("auto-start is read-only with no status, or under Colab ownership", () => {
  assert.equal(remoteAccessAutoStartReadOnly(null), true);
  assert.equal(
    remoteAccessAutoStartReadOnly(
      // biome-ignore lint/style/useNamingConvention: API schema
      normalizeRemoteAccessStatus(apiStatus({ managed_by: "colab" })),
    ),
    true,
  );
  assert.equal(
    remoteAccessAutoStartReadOnly(
      // biome-ignore lint/style/useNamingConvention: API schema
      normalizeRemoteAccessStatus(apiStatus({ block_reason: "colab" })),
    ),
    true,
  );
  assert.equal(
    remoteAccessAutoStartReadOnly(
      // biome-ignore lint/style/useNamingConvention: API schema
      normalizeRemoteAccessStatus(apiStatus({ managed_by: "launch" })),
    ),
    false,
  );
  assert.equal(
    remoteAccessAutoStartReadOnly(normalizeRemoteAccessStatus(apiStatus())),
    false,
  );
});

// ── remoteAccessStopDisconnectsOrigin ──
// This decides whether the user is warned that Stop will cut their own
// connection, so a false negative silently drops the warning.

test("stop-disconnects detects the browser sitting on the tunnel origin", () => {
  assert.equal(remoteAccessStopDisconnectsOrigin(TUNNEL, TUNNEL), true);
});

test("stop-disconnects normalizes trailing slashes on both sides", () => {
  assert.equal(remoteAccessStopDisconnectsOrigin(`${TUNNEL}/`, TUNNEL), true);
  assert.equal(remoteAccessStopDisconnectsOrigin(TUNNEL, `${TUNNEL}/`), true);
  assert.equal(
    remoteAccessStopDisconnectsOrigin(`${TUNNEL}///`, `${TUNNEL}/`),
    true,
  );
});

test("stop-disconnects is false for a local browser or a different tunnel", () => {
  assert.equal(
    remoteAccessStopDisconnectsOrigin(TUNNEL, "http://127.0.0.1:8888"),
    false,
  );
  assert.equal(
    remoteAccessStopDisconnectsOrigin(
      TUNNEL,
      "https://other-name.trycloudflare.com",
    ),
    false,
  );
  assert.equal(remoteAccessStopDisconnectsOrigin(null, TUNNEL), false);
  assert.equal(remoteAccessStopDisconnectsOrigin(null, ""), false);
});

test("stop-disconnects does not treat a path prefix as the same origin", () => {
  // A tunnel URL is an origin; nothing should match a longer path under it.
  assert.equal(
    remoteAccessStopDisconnectsOrigin(`${TUNNEL}/settings`, TUNNEL),
    false,
  );
});

// ── remoteAccessBlockMessage ──

test("every block reason the backend can emit has a message", () => {
  // Mirrors utils/remote_access_settings.py's block_reason chain.
  const reasons = [
    "server_starting",
    "admin_password_change_required",
    "explicitly_disabled",
    "launch_managed",
    "colab_managed",
    "colab",
  ];
  for (const reason of reasons) {
    for (const isDesktop of [true, false]) {
      const msg = remoteAccessBlockMessage(reason, isDesktop);
      assert.ok(
        msg && msg.length > 0,
        `no message for ${reason} (desktop=${isDesktop})`,
      );
    }
  }
});

test("the pending-password message is desktop-aware", () => {
  const desktop = remoteAccessBlockMessage(
    "admin_password_change_required",
    true,
  );
  const web = remoteAccessBlockMessage("admin_password_change_required", false);
  assert.notEqual(desktop, web);
  // Only the web copy should send the user to the CLI.
  assert.ok(!desktop?.includes("reset-password"));
  assert.ok(web?.includes("reset-password"));
});

test("an unknown or absent reason yields no message", () => {
  assert.equal(remoteAccessBlockMessage(null, false), null);
  assert.equal(remoteAccessBlockMessage("something_new", false), null);
  assert.equal(remoteAccessBlockMessage("", true), null);
});

// ── remoteAccessSelfStopPoll ──

// The frames a Settings page loaded from the tunnel actually sees after Stop:
// the route fabricates a terminal off, the stop worker then answers "stopping"
// for the ~50ms teardown drain, and finally the origin goes away.
const TERMINAL_OFF = normalizeRemoteAccessStatus(
  apiStatus({
    state: "off",
    // biome-ignore lint/style/useNamingConvention: API schema
    can_start: false,
  }),
);
const DRAINING = normalizeRemoteAccessStatus(
  apiStatus({
    state: "stopping",
    // biome-ignore lint/style/useNamingConvention: API schema
    managed_by: "settings",
    // biome-ignore lint/style/useNamingConvention: API schema
    can_start: false,
  }),
);

test("a teardown frame never overwrites the terminal off of a self-origin stop", () => {
  const settled = remoteAccessSelfStopPoll(DRAINING, true);
  assert.equal(settled.apply, false);
  assert.equal(settled.expectingDisconnect, true);
});

test("a stop that left the connector running takes the card back over", () => {
  const alive = normalizeRemoteAccessStatus(
    apiStatus({
      state: "online",
      url: TUNNEL,
      // biome-ignore lint/style/useNamingConvention: API schema
      managed_by: "settings",
      // biome-ignore lint/style/useNamingConvention: API schema
      can_start: false,
      // biome-ignore lint/style/useNamingConvention: API schema
      can_stop: true,
    }),
  );
  const settled = remoteAccessSelfStopPoll(alive, true);
  assert.equal(settled.apply, true);
  assert.equal(settled.expectingDisconnect, false);
});

test("a stop that failed still surfaces its error", () => {
  const failed = normalizeRemoteAccessStatus(
    apiStatus({ state: "error", error: "cloudflared could not be stopped" }),
  );
  const settled = remoteAccessSelfStopPoll(failed, true);
  assert.equal(settled.apply, true);
});

test("polls lead as usual when no self-origin stop is pending", () => {
  for (const frame of [TERMINAL_OFF, DRAINING]) {
    const settled = remoteAccessSelfStopPoll(frame, false);
    assert.equal(settled.apply, true);
    assert.equal(settled.expectingDisconnect, false);
  }
});

test("a self-origin stop settles on off, not a permanent stopping", () => {
  let shown = TERMINAL_OFF;
  let expecting = true;
  // Polling restarts in perform()'s finally, so the first poll can still land
  // inside the drain window before cloudflared exits.
  for (const frame of [DRAINING, DRAINING]) {
    const settled = remoteAccessSelfStopPoll(frame, expecting);
    expecting = settled.expectingDisconnect;
    if (settled.apply) {
      shown = frame;
    }
  }
  assert.equal(shown.state, "off");
  // The origin then dies; the section latches polling off on this flag.
  assert.equal(expecting, true);
});
