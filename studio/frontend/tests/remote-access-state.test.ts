// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/settings-api-resolver.mjs", import.meta.url);

// remote-access-section.tsx pulls in the router, motion and hugeicons, so it
// cannot be imported here. These drive the pure state helpers it consumes.
import {
  type ApiRemoteAccessStatus,
  normalizeRemoteAccessStatus,
  remoteAccessAutoStartKind,
  remoteAccessAutoStartReadOnly,
  remoteAccessBlockMessageId,
  remoteAccessHeaderActionDisabled,
  remoteAccessIsReady,
  remoteAccessPollDelay,
  remoteAccessPreferredKind,
  remoteAccessProgressSteps,
  remoteAccessSelfStopPoll,
  remoteAccessShowsCustomPanel,
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
      method: "custom",
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
      kind: "custom",
      // biome-ignore lint/style/useNamingConvention: API schema
      connector_registered: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      tunnel_serving: true,
      dns: "pending",
      // biome-ignore lint/style/useNamingConvention: API schema
      auto_start_kind: "custom",
      // biome-ignore lint/style/useNamingConvention: API schema
      auto_start_block_reason: "launch_managed",
      // biome-ignore lint/style/useNamingConvention: API schema
      custom_state: "configured",
      // biome-ignore lint/style/useNamingConvention: API schema
      custom_hostname: "studio.example.com",
      // biome-ignore lint/style/useNamingConvention: API schema
      custom_tunnel_name: "unsloth-AB12CD",
      // biome-ignore lint/style/useNamingConvention: API schema
      custom_runnable: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      login_url: "https://dash.cloudflare.com/login",
      // biome-ignore lint/style/useNamingConvention: API schema
      custom_error: "dns_conflict",
      // biome-ignore lint/style/useNamingConvention: API schema
      custom_error_detail: "record exists",
      // biome-ignore lint/style/useNamingConvention: API schema
      custom_error_phase: "provision",
      // biome-ignore lint/style/useNamingConvention: API schema
      custom_error_settled: true,
    }),
  );
  assert.deepEqual(s, {
    state: "online",
    url: TUNNEL,
    error: null,
    autoStart: true,
    method: "custom",
    defaultAutoStart: true,
    available: true,
    managedBy: "settings",
    canStart: false,
    canStop: true,
    blockReason: null,
    passwordPending: true,
    streamingSupported: true,
    kind: "custom",
    connectorRegistered: true,
    tunnelServing: true,
    dns: "pending",
    autoStartKind: "custom",
    autoStartBlockReason: "launch_managed",
    customState: "configured",
    customHostname: "studio.example.com",
    customTunnelName: "unsloth-AB12CD",
    customRunnable: true,
    loginUrl: "https://dash.cloudflare.com/login",
    customError: "dns_conflict",
    customErrorDetail: "record exists",
    customErrorPhase: "provision",
    customErrorSettled: true,
  });
  // No field may normalize to undefined -- a wrong key would silently do so.
  for (const [k, v] of Object.entries(s)) {
    assert.notEqual(v, undefined, `${k} normalized to undefined`);
  }
});

test("normalize defaults the optional fields an older backend may omit", () => {
  const s = normalizeRemoteAccessStatus(apiStatus());
  assert.equal(s.url, null);
  assert.equal(s.method, "temporary");
  assert.equal(s.error, null);
  assert.equal(s.managedBy, null);
  assert.equal(s.blockReason, null);
  assert.equal(s.passwordPending, false);
  assert.equal(s.kind, null);
  assert.equal(s.connectorRegistered, false);
  assert.equal(s.tunnelServing, false);
  assert.equal(s.dns, "unknown");
  assert.equal(s.autoStartKind, null);
  assert.equal(s.autoStartBlockReason, null);
  assert.equal(s.customState, "unconfigured");
  assert.equal(s.customHostname, null);
  assert.equal(s.customTunnelName, null);
  assert.equal(s.customRunnable, false);
  assert.equal(s.loginUrl, null);
  assert.equal(s.customError, null);
  assert.equal(s.customErrorDetail, null);
  assert.equal(s.customErrorPhase, null);
  assert.equal(s.customErrorSettled, false);
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

test("poll delay is fast while a lifecycle or custom transition is in flight", () => {
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
  for (const customState of ["provisioning", "tearing_down"] as const) {
    assert.equal(
      remoteAccessPollDelay(
        // biome-ignore lint/style/useNamingConvention: API schema
        normalizeRemoteAccessStatus(apiStatus({ custom_state: customState })),
      ),
      1000,
    );
  }
});

test("connection progress follows each method until it is ready", () => {
  const temporary = normalizeRemoteAccessStatus(
    apiStatus({
      state: "starting",
      kind: "temporary",
      // biome-ignore lint/style/useNamingConvention: API schema
      tunnel_serving: true,
    }),
  );
  assert.deepEqual(remoteAccessProgressSteps(temporary), [
    { id: "connecting", complete: false },
    { id: "openingLink", complete: true },
  ]);
  assert.equal(remoteAccessIsReady(temporary), false);

  const custom = normalizeRemoteAccessStatus(
    apiStatus({
      state: "online",
      kind: "custom",
      // biome-ignore lint/style/useNamingConvention: API schema
      connector_registered: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      tunnel_serving: false,
      dns: "pending",
    }),
  );
  assert.deepEqual(remoteAccessProgressSteps(custom), [
    { id: "connecting", complete: true },
    { id: "openingLink", complete: false },
    { id: "checkingHostname", complete: false },
  ]);
  assert.equal(remoteAccessIsReady(custom), false);

  const ready = { ...custom, tunnelServing: true, dns: "resolved" as const };
  assert.deepEqual(remoteAccessProgressSteps(ready), []);
  assert.equal(remoteAccessIsReady(ready), true);
  assert.deepEqual(remoteAccessProgressSteps({ ...ready, state: "stopping" }), [
    { id: "disconnecting", complete: false },
  ]);
});

test("the saved method is authoritative even when Custom is configured", () => {
  const temporary = normalizeRemoteAccessStatus(
    // biome-ignore lint/style/useNamingConvention: API schema
    apiStatus({ method: "temporary", custom_runnable: true }),
  );
  const custom = normalizeRemoteAccessStatus(
    // biome-ignore lint/style/useNamingConvention: API schema
    apiStatus({ method: "custom", custom_runnable: false, can_start: false }),
  );
  assert.equal(remoteAccessPreferredKind(temporary), "temporary");
  assert.equal(remoteAccessAutoStartKind(temporary), "temporary");
  assert.equal(remoteAccessPreferredKind(custom), "custom");
  assert.equal(remoteAccessAutoStartKind(custom), "custom");
  assert.equal(remoteAccessHeaderActionDisabled(custom, false), true);
});

test("an active Custom operation stays visible after a concurrent method change", () => {
  const provisioning = normalizeRemoteAccessStatus(
    apiStatus({
      method: "temporary",
      // biome-ignore lint/style/useNamingConvention: API schema
      custom_state: "provisioning",
    }),
  );
  assert.equal(remoteAccessShowsCustomPanel(provisioning), true);
  assert.equal(
    remoteAccessShowsCustomPanel({
      ...provisioning,
      customState: "tearing_down",
    }),
    true,
  );
  assert.equal(
    remoteAccessShowsCustomPanel({
      ...provisioning,
      customState: "configured",
    }),
    false,
  );
});

test("start and auto-start requests leave method selection to the saved setting", async () => {
  const {
    remoteAccessAutoStartRequest,
    remoteAccessMethodRequest,
    remoteAccessStartRequest,
  } = await import("../src/features/settings/api/remote-access.ts");
  assert.deepEqual(remoteAccessStartRequest(), {
    path: "/start",
    init: { method: "POST" },
  });
  assert.deepEqual(remoteAccessAutoStartRequest(true), {
    path: "/auto-start",
    init: {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: '{"enabled":true}',
    },
  });
  assert.deepEqual(remoteAccessMethodRequest("custom"), {
    path: "/method",
    init: {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: '{"method":"custom"}',
    },
  });
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
      const msg = remoteAccessBlockMessageId(reason, isDesktop);
      assert.ok(
        msg && msg.length > 0,
        `no message for ${reason} (desktop=${isDesktop})`,
      );
    }
  }
});

test("the pending-password message is desktop-aware", () => {
  const desktop = remoteAccessBlockMessageId(
    "admin_password_change_required",
    true,
  );
  const web = remoteAccessBlockMessageId(
    "admin_password_change_required",
    false,
  );
  assert.notEqual(desktop, web);
  assert.equal(desktop, "passwordDesktop");
  assert.equal(web, "passwordWeb");
});

test("an unknown or absent reason yields no message", () => {
  assert.equal(remoteAccessBlockMessageId(null, false), null);
  assert.equal(remoteAccessBlockMessageId("something_new", false), null);
  assert.equal(remoteAccessBlockMessageId("", true), null);
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
