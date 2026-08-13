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
  remoteAccessAuthorizationShouldOpen,
  remoteAccessAuthorizationAction,
  remoteAccessAuthorizationView,
  remoteAccessAutoStartReadOnly,
  remoteAccessBlockMessageId,
  remoteAccessHeaderActionDisabled,
  remoteAccessHeaderStatus,
  remoteAccessOperationRevision,
  remoteAccessPollDelay,
  remoteAccessPreferredKind,
  remoteAccessProgressStep,
  remoteAccessSelfStopPoll,
  remoteAccessSetupDialogShouldOpen,
  remoteAccessShowsCustomPanel,
  remoteAccessStopDisconnectsOrigin,
  remoteApiOrigin,
} from "../src/features/settings/api/remote-access-state.ts";
import { en } from "../src/i18n/locales/en.ts";

const TUNNEL = "https://calm-otter-review.trycloudflare.com";

function apiStatus(
  over: Partial<ApiRemoteAccessStatus> = {},
): ApiRemoteAccessStatus {
  return {
    state: "off",
    auto_start: false,
    default_auto_start: false,
    available: true,
    can_start: true,
    can_stop: false,
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
      auto_start: true,
      method: "custom",
      default_auto_start: true,
      available: true,
      managed_by: "settings",
      can_start: false,
      can_stop: true,
      block_reason: null,
      password_pending: true,
      streaming_supported: true,
      kind: "custom",
      connector_registered: true,
      tunnel_serving: true,
      dns: "pending",
      auto_start_kind: "custom",
      auto_start_block_reason: "launch_managed",
      custom_state: "configured",
      custom_operation_revision: 7,
      custom_hostname: "studio.example.com",
      custom_tunnel_name: "unsloth-AB12CD",
      custom_runnable: true,
      login_url: "https://dash.cloudflare.com/login",
      custom_error: "dns_conflict",
      custom_error_detail: "record exists",
      custom_error_phase: "provision",
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
    customOperationRevision: 7,
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
  assert.equal(s.customOperationRevision, 0);
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
      apiStatus({ password_pending: raw as never }),
    );
    assert.equal(
      s.passwordPending,
      false,
      `password_pending=${JSON.stringify(raw)} must not be truthy`,
    );
  }
  assert.equal(
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
        normalizeRemoteAccessStatus(apiStatus({ custom_state: customState })),
      ),
      1000,
    );
  }
});

test("connection progress reports only the current step", () => {
  const temporary = normalizeRemoteAccessStatus(
    apiStatus({
      state: "starting",
      kind: "temporary",
      connector_registered: true,
      tunnel_serving: false,
    }),
  );
  assert.equal(remoteAccessProgressStep(temporary), "openingLink");
  assert.equal(
    remoteAccessProgressStep({ ...temporary, connectorRegistered: false }),
    "connecting",
  );

  // Still starting: the connector serves, but the hostname has not resolved, so
  // the backend withholds online until there is a link to offer.
  const custom = normalizeRemoteAccessStatus(
    apiStatus({
      state: "starting",
      kind: "custom",
      connector_registered: true,
      tunnel_serving: false,
      // The hostname is not watched until the tunnel serves, so nothing is
      // known about it yet while the link is still opening.
      dns: "unknown",
    }),
  );
  assert.equal(remoteAccessProgressStep(custom), "openingLink");

  const ready = {
    ...custom,
    state: "online" as const,
    tunnelServing: true,
    dns: "resolved" as const,
    url: "https://studio.example.com",
  };
  assert.equal(remoteAccessProgressStep(ready), null);
  assert.equal(
    remoteAccessProgressStep({ ...ready, state: "stopping" }),
    "disconnecting",
  );

  // Serving, so the URL is published and the hostname is being watched; the
  // reported state stays starting until it resolves.
  assert.equal(
    remoteAccessProgressStep({
      ...custom,
      tunnelServing: true,
      dns: "pending",
      url: "https://studio.example.com",
    }),
    "checkingHostname",
  );
});

test("the header preserves lifecycle status and details only Starting", () => {
  const starting = normalizeRemoteAccessStatus(
    apiStatus({
      state: "starting",
      managed_by: "settings",
    }),
  );
  assert.deepEqual(remoteAccessHeaderStatus(starting), {
    state: "starting",
    owner: null,
    step: "connecting",
  });
  // Online is reported only once there is a link, so it has nothing left to
  // narrate. A stop names itself.
  assert.deepEqual(
    remoteAccessHeaderStatus({
      ...starting,
      state: "online",
      kind: "custom",
      connectorRegistered: true,
      tunnelServing: true,
      dns: "resolved",
      url: "https://studio.example.com",
    }),
    { state: "online", owner: null, step: null },
  );
  assert.equal(
    remoteAccessHeaderStatus({ ...starting, state: "stopping" }).step,
    null,
  );
  assert.equal(
    remoteAccessHeaderStatus({ ...starting, managedBy: "launch" }).owner,
    "launch",
  );
  assert.deepEqual(remoteAccessHeaderStatus(null), {
    state: null,
    owner: null,
    step: null,
  });
  assert.equal(en.settings.general.remoteAccess.stopAction, "Stop");
});

test("the authorization action waits for the URL before offering a link", () => {
  // Before confirming there is only the confirm button; after it, Cloudflare
  // still has to return a URL, and only then is there a link to offer.
  assert.equal(remoteAccessAuthorizationAction(false, null, false), "confirm");
  assert.equal(
    remoteAccessAuthorizationAction(false, "https://cloudflare.test/a", true),
    "confirm",
  );
  assert.equal(remoteAccessAuthorizationAction(true, null, true), "preparing");
  // A URL left over from a superseded attempt is not this one to open.
  assert.equal(
    remoteAccessAuthorizationAction(true, "https://cloudflare.test/a", false),
    "preparing",
  );
  assert.equal(
    remoteAccessAuthorizationAction(true, "https://cloudflare.test/a", true),
    "open",
  );
});

test("the saved method is authoritative even when Custom is configured", () => {
  const temporary = normalizeRemoteAccessStatus(
    apiStatus({ method: "temporary", custom_runnable: true }),
  );
  const custom = normalizeRemoteAccessStatus(
    apiStatus({ method: "custom", custom_runnable: false, can_start: false }),
  );
  assert.equal(remoteAccessPreferredKind(temporary), "temporary");
  assert.equal(remoteAccessPreferredKind(temporary), "temporary");
  assert.equal(remoteAccessPreferredKind(custom), "custom");
  assert.equal(remoteAccessPreferredKind(custom), "custom");
  assert.equal(remoteAccessHeaderActionDisabled(custom, false), true);
});

test("an active Custom operation stays visible after a concurrent method change", () => {
  const provisioning = normalizeRemoteAccessStatus(
    apiStatus({
      method: "temporary",
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

test("setup dialog and authorization follow the current provisioning revision", () => {
  const provisioning = normalizeRemoteAccessStatus(
    apiStatus({
      custom_state: "provisioning",
      custom_hostname: "studio.example.com",
      custom_operation_revision: 4,
    }),
  );
  assert.equal(remoteAccessSetupDialogShouldOpen(provisioning, null), true);
  assert.equal(remoteAccessSetupDialogShouldOpen(provisioning, 4), false);
  assert.equal(remoteAccessSetupDialogShouldOpen(provisioning, 3), true);
  const beforeProvision = {
    ...provisioning,
    customState: "unconfigured" as const,
  };
  assert.deepEqual(
    remoteAccessAuthorizationView(beforeProvision, false, null, null),
    { confirmationDisabled: true, current: false, phase: null },
  );
  assert.deepEqual(
    remoteAccessAuthorizationView(beforeProvision, true, null, null),
    { confirmationDisabled: false, current: false, phase: null },
  );
  assert.deepEqual(
    remoteAccessAuthorizationView(provisioning, false, null, null),
    { confirmationDisabled: false, current: false, phase: null },
  );
  assert.deepEqual(
    remoteAccessAuthorizationView(beforeProvision, true, "pending", null),
    { confirmationDisabled: false, current: false, phase: "connecting" },
  );
  assert.deepEqual(
    remoteAccessAuthorizationView(provisioning, false, 4, null),
    { confirmationDisabled: false, current: true, phase: "connecting" },
  );
  assert.deepEqual(
    remoteAccessAuthorizationView(
      { ...provisioning, customOperationRevision: 5 },
      false,
      4,
      null,
    ),
    { confirmationDisabled: false, current: false, phase: null },
  );
  assert.equal(
    remoteAccessAuthorizationShouldOpen(
      provisioning,
      true,
      4,
      "https://dash.cloudflare.com/old-login",
      null,
    ),
    false,
  );
  const readyToAuthorize = {
    ...provisioning,
    loginUrl: "https://dash.cloudflare.com/login",
  };
  assert.equal(
    remoteAccessAuthorizationView(readyToAuthorize, false, 4, null).phase,
    "approval",
  );
  assert.deepEqual(
    remoteAccessAuthorizationView(readyToAuthorize, false, 4, 4),
    { confirmationDisabled: true, current: false, phase: null },
  );
  assert.equal(
    remoteAccessAuthorizationShouldOpen(
      readyToAuthorize,
      true,
      null,
      null,
      null,
    ),
    false,
  );
  assert.equal(
    remoteAccessAuthorizationShouldOpen(readyToAuthorize, true, 4, null, null),
    true,
  );
  assert.equal(
    remoteAccessAuthorizationShouldOpen(
      { ...readyToAuthorize, customOperationRevision: 5 },
      true,
      4,
      null,
      null,
    ),
    false,
  );
  assert.equal(
    remoteAccessSetupDialogShouldOpen(
      { ...provisioning, customState: "unconfigured" },
      null,
    ),
    false,
  );
});

test("custom request errors belong to one backend operation revision", () => {
  const provisioning = normalizeRemoteAccessStatus(
    apiStatus({
      custom_state: "provisioning",
      custom_operation_revision: 4,
    }),
  );
  assert.notEqual(
    remoteAccessOperationRevision(provisioning, "cancel"),
    remoteAccessOperationRevision(
      { ...provisioning, customOperationRevision: 5 },
      "cancel",
    ),
  );
});

test("remote access requests carry their operation inputs", async () => {
  const {
    remoteAccessAutoStartRequest,
    remoteAccessCancelRequest,
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
  assert.deepEqual(remoteAccessCancelRequest(7), {
    path: "/custom/cancel",
    init: {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: '{"expected_revision":7}',
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
      normalizeRemoteAccessStatus(apiStatus({ managed_by: "colab" })),
    ),
    true,
  );
  assert.equal(
    remoteAccessAutoStartReadOnly(
      normalizeRemoteAccessStatus(apiStatus({ block_reason: "colab" })),
    ),
    true,
  );
  assert.equal(
    remoteAccessAutoStartReadOnly(
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
    "custom_tunnel_not_configured",
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
    can_start: false,
  }),
);
const DRAINING = normalizeRemoteAccessStatus(
  apiStatus({
    state: "stopping",
    managed_by: "settings",
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
      managed_by: "settings",
      can_start: false,
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
