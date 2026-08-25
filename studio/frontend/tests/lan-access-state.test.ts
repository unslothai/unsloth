// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// lan-access-section.tsx pulls in hugeicons, so only its pure helpers are tested here
import {
  type ApiLanAccessStatus,
  keylessLanAccessDescription,
  lanAccessAutoStartReadOnly,
  lanAccessBlockMessage,
  lanAccessErrorMessage,
  lanAccessStopDisconnectsOrigin,
  normalizeLanAccessStatus,
} from "../src/features/settings/api/lan-access-state.ts";

const LAN = "http://192.168.1.24:8888";
const SECOND = "http://10.0.0.7:8888";
const PUBLIC = "http://64.227.100.5:8888";

function apiStatus(over: Partial<ApiLanAccessStatus> = {}): ApiLanAccessStatus {
  return {
    state: "off",
    // biome-ignore lint/style/useNamingConvention: API schema
    auto_start: false,
    // biome-ignore lint/style/useNamingConvention: API schema
    can_start: true,
    // biome-ignore lint/style/useNamingConvention: API schema
    can_stop: false,
    ...over,
  };
}

// ── normalizeLanAccessStatus ──

test("normalize maps every snake_case field onto its camelCase name", () => {
  const s = normalizeLanAccessStatus(
    apiStatus({
      state: "online",
      urls: [LAN, SECOND],
      error: null,
      // biome-ignore lint/style/useNamingConvention: API schema
      auto_start: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      managed_by: "settings",
      // biome-ignore lint/style/useNamingConvention: API schema
      can_start: false,
      // biome-ignore lint/style/useNamingConvention: API schema
      can_stop: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      block_reason: null,
      // biome-ignore lint/style/useNamingConvention: API schema
      serves_web_ui: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      keyless_lan_eligible: true,
    }),
  );
  assert.deepEqual(s, {
    state: "online",
    urls: [LAN, SECOND],
    publicUrls: [],
    error: null,
    autoStart: true,
    managedBy: "settings",
    canStart: false,
    canStop: true,
    blockReason: null,
    servesWebUi: true,
    keylessLanEligible: true,
    keylessScope: "off",
    keylessTools: false,
  });
});

test("normalize defaults the optional fields an older backend may omit", () => {
  const s = normalizeLanAccessStatus(apiStatus());
  assert.deepEqual(s.urls, []);
  assert.deepEqual(s.publicUrls, []);
  assert.equal(s.error, null);
  assert.equal(s.managedBy, null);
  assert.equal(s.blockReason, null);
  assert.equal(s.servesWebUi, true);
  assert.equal(s.keylessLanEligible, false);
  assert.equal(s.keylessScope, "off");
  assert.equal(s.keylessTools, false);
});

test("keyless state and messaging preserve every security boundary", () => {
  const unknown = normalizeLanAccessStatus(
    apiStatus({ keyless_scope: "unknown", keyless_tools: true }),
  );
  assert.deepEqual(
    [unknown.keylessScope, unknown.keylessTools],
    ["off", false],
  );
  assert.ok(
    keylessLanAccessDescription(null).includes("Authentication is required"),
  );
  const cases: [Partial<ApiLanAccessStatus>, string][] = [
    [
      {
        state: "online",
        keyless_lan_eligible: false,
        keyless_scope: "inference",
      },
      "active private listener",
    ],
    [
      {
        state: "online",
        keyless_lan_eligible: true,
        keyless_scope: "inference",
      },
      "this active private LAN",
    ],
    [
      {
        state: "online",
        public_urls: [PUBLIC],
        keyless_lan_eligible: true,
        keyless_scope: "inference",
      },
      "never through the listed public URL",
    ],
    [{ keyless_scope: "full" }, "never granted over LAN or public URLs"],
    [
      { block_reason: "colab", keyless_scope: "inference" },
      "Colab never receives keyless access",
    ],
  ];
  for (const [overrides, fragment] of cases) {
    const status = normalizeLanAccessStatus(apiStatus(overrides));
    assert.ok(keylessLanAccessDescription(status).includes(fragment));
  }
});

test("urls survives a null or non-array payload without throwing", () => {
  for (const urls of [null, undefined, "nope" as never]) {
    const s = normalizeLanAccessStatus(apiStatus({ urls }));
    assert.deepEqual(s.urls, []);
    assert.deepEqual(s.publicUrls, []);
  }
});

test("a public address is carried through so the section can warn about it", () => {
  const s = normalizeLanAccessStatus(
    // biome-ignore lint/style/useNamingConvention: API schema
    apiStatus({ urls: [PUBLIC, LAN], public_urls: [PUBLIC] }),
  );
  assert.deepEqual(s.publicUrls, [PUBLIC]);
  assert.deepEqual(s.urls, [PUBLIC, LAN]);
});

test("servesWebUi is only false for an explicit false", () => {
  for (const raw of [undefined, null, true]) {
    assert.equal(
      // biome-ignore lint/style/useNamingConvention: API schema
      normalizeLanAccessStatus(apiStatus({ serves_web_ui: raw as never }))
        .servesWebUi,
      true,
    );
  }
  assert.equal(
    // biome-ignore lint/style/useNamingConvention: API schema
    normalizeLanAccessStatus(apiStatus({ serves_web_ui: false })).servesWebUi,
    false,
  );
});

// ── lanAccessAutoStartReadOnly ──

test("auto-start is read-only with no status, or under Colab", () => {
  assert.equal(lanAccessAutoStartReadOnly(null), true);
  assert.equal(
    lanAccessAutoStartReadOnly(
      // biome-ignore lint/style/useNamingConvention: API schema
      normalizeLanAccessStatus(apiStatus({ block_reason: "colab" })),
    ),
    true,
  );
  // a launch-managed bind still lets the preference be set for next time
  assert.equal(
    lanAccessAutoStartReadOnly(
      // biome-ignore lint/style/useNamingConvention: API schema
      normalizeLanAccessStatus(apiStatus({ block_reason: "launch_managed" })),
    ),
    false,
  );
  assert.equal(
    lanAccessAutoStartReadOnly(normalizeLanAccessStatus(apiStatus())),
    false,
  );
});

// ── lanAccessStopDisconnectsOrigin ──
// a false negative here leaves the page polling an origin its own stop just killed

test("stop-disconnects matches any of the bound addresses", () => {
  assert.equal(lanAccessStopDisconnectsOrigin([LAN, SECOND], SECOND), true);
  assert.equal(lanAccessStopDisconnectsOrigin([LAN], LAN), true);
});

test("stop-disconnects normalizes trailing slashes on both sides", () => {
  assert.equal(lanAccessStopDisconnectsOrigin([`${LAN}/`], LAN), true);
  assert.equal(lanAccessStopDisconnectsOrigin([LAN], `${LAN}/`), true);
  assert.equal(lanAccessStopDisconnectsOrigin([`${LAN}///`], `${LAN}/`), true);
});

test("stop-disconnects normalizes default HTTP and HTTPS ports", () => {
  assert.equal(
    lanAccessStopDisconnectsOrigin(
      ["http://192.168.1.24:80"],
      "http://192.168.1.24",
    ),
    true,
  );
  assert.equal(
    lanAccessStopDisconnectsOrigin(
      ["https://192.168.1.24:443"],
      "https://192.168.1.24",
    ),
    true,
  );
});

test("stop-disconnects is false for a loopback browser or another address", () => {
  assert.equal(
    lanAccessStopDisconnectsOrigin([LAN], "http://127.0.0.1:8888"),
    false,
  );
  assert.equal(lanAccessStopDisconnectsOrigin([LAN], SECOND), false);
  assert.equal(lanAccessStopDisconnectsOrigin([], LAN), false);
  assert.equal(lanAccessStopDisconnectsOrigin([], ""), false);
  assert.equal(lanAccessStopDisconnectsOrigin(["not a URL"], LAN), false);
});

test("stop-disconnects does not treat a different port as the same origin", () => {
  assert.equal(
    lanAccessStopDisconnectsOrigin([LAN], "http://192.168.1.24:9999"),
    false,
  );
});

// ── lanAccessBlockMessage ──

test("every block reason the backend can emit has a message", () => {
  // mirrors the block_reason chain in utils/lan_access_settings.py
  const reasons = [
    "server_starting",
    "colab",
    "launch_managed",
    "secure_launch",
    "admin_password_change_required",
  ];
  for (const reason of reasons) {
    for (const isDesktop of [true, false]) {
      const msg = lanAccessBlockMessage(reason, isDesktop);
      assert.ok(
        msg && msg.length > 0,
        `no message for ${reason} (desktop=${isDesktop})`,
      );
    }
  }
});

test("the pending-password message is desktop-aware", () => {
  const desktop = lanAccessBlockMessage("admin_password_change_required", true);
  const web = lanAccessBlockMessage("admin_password_change_required", false);
  assert.notEqual(desktop, web);
  assert.ok(!desktop?.includes("reset-password"));
  assert.ok(web?.includes("reset-password"));
});

test("an unknown or absent reason yields no message", () => {
  assert.equal(lanAccessBlockMessage(null, false), null);
  assert.equal(lanAccessBlockMessage("something_new", false), null);
  assert.equal(lanAccessBlockMessage("", true), null);
});

// ── lanAccessErrorMessage ──

test("every listener failure the backend can raise has a message", () => {
  // mirrors the RuntimeError reasons in lan_access.start_lan_listener
  for (const error of [
    "no_lan_address",
    "bind_failed",
    "listener_start_failed",
    "stop_timed_out",
  ]) {
    const msg = lanAccessErrorMessage(error);
    assert.ok(msg && msg.length > 0, `no message for ${error}`);
  }
});

test("no error means no message, and an unknown one still says something", () => {
  assert.equal(lanAccessErrorMessage(null), null);
  assert.ok(lanAccessErrorMessage("something_new")?.length);
});
