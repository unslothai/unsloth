// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The keyless usage example must only be offered for a base admission can accept.
 *
 * `keyless_api_access.keyless_authority_address_allowed` decides that on the backend:
 * loopback always, private LAN under `inference`, nothing else -- no hostname, no
 * IPv4-mapped literal, no unspecified or public address. The panel must give the same
 * answer or it prints a copy-paste `Bearer not-needed` command that answers 401.
 *
 * It has been wrong three ways, each time by testing something adjacent to the real question:
 *   1. `exposure === "private_lan"` short-circuited the host test, so
 *      `unsloth studio -H box.local` advertised keyless for a hostname.
 *   2. the replacement tested literal SYNTAX, so `[::ffff:192.168.1.24]`, `[::]` and
 *      `8.8.8.8` all sailed through.
 *   3. `isPrivateLanHost` unwrapped `::ffff:`, the one thing that form is refused for.
 *
 * So this pins the verdict table itself, not the presence of a guard, mirroring
 * `test_what_the_ui_advertises_matches_what_admission_accepts` in
 * studio/backend/tests/test_keyless_api_access_adversarial.py -- change both together.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  isKeylessAllowedAuthority,
  keylessBaseEligible,
} from "../src/features/settings/components/keyless-example-eligibility.ts";

// base -> whether the panel may advertise keyless for it, under scope=inference and the
// widest exposure the backend ever reports. Mirrors
// test_what_the_ui_advertises_matches_what_admission_accepts in
// studio/backend/tests/test_keyless_api_access_adversarial.py -- change both together.
const TABLE: [string, boolean, string][] = [
  ["http://127.0.0.1:8888", true, "loopback"],
  ["http://127.0.0.2:8888", true, "loopback, whole 127/8"],
  ["http://localhost:8888", true, "loopback name"],
  ["http://[::1]:8888", true, "loopback v6"],
  ["http://192.168.1.24:8888", true, "private LAN"],
  ["http://10.0.0.5:8888", true, "private LAN"],
  ["http://172.16.0.9:8888", true, "private LAN"],
  ["http://[fd00::1]:8888", true, "unique local"],
  [
    "http://[::ffff:192.168.1.24]:8888",
    false,
    "IPv4-mapped: admission refuses",
  ],
  [
    "http://[::ffff:c0a8:118]:8888",
    false,
    "IPv4-mapped hex: admission refuses",
  ],
  ["http://[::]:8888", false, "unspecified v6"],
  ["http://0.0.0.0:8888", false, "unspecified v4"],
  ["http://8.8.8.8:8888", false, "public"],
  ["http://[2001:db8::1]:8888", false, "public v6"],
  ["http://100.64.0.1:8888", false, "CGNAT, outside the private networks"],
  ["http://box.local:8888", false, "mDNS name"],
  ["http://studio.internal:8888", false, "internal name"],
  ["http://mybox:8888", false, "single-label name"],
];

test("the keyless example is offered exactly where admission accepts the authority", () => {
  for (const [base, expected, why] of TABLE) {
    assert.equal(
      keylessBaseEligible(base, "inference", "private_lan"),
      expected,
      `${base} (${why}): panel said ${!expected}, admission says ${expected}`,
    );
  }
});

test("exposure alone cannot make a rejected base eligible", () => {
  // The regression that started this: `private_lan` is computed from the RESOLVED
  // address, so it must never widen a base the authority rule rejected.
  for (const exposure of ["private_lan", "network", null] as const) {
    assert.equal(
      keylessBaseEligible("http://box.local:8888", "inference", exposure),
      false,
    );
    assert.equal(
      keylessBaseEligible(
        "http://[::ffff:192.168.1.24]:8888",
        "inference",
        exposure,
      ),
      false,
    );
    assert.equal(
      keylessBaseEligible("http://8.8.8.8:8888", "inference", exposure),
      false,
    );
  }
});

test("full scope never reaches a non-loopback authority", () => {
  // full is loopback-only in admission, so the panel must not offer a LAN base for it.
  assert.equal(
    keylessBaseEligible("http://192.168.1.24:8888", "full", "private_lan"),
    false,
  );
  assert.equal(
    keylessBaseEligible("http://127.0.0.1:8888", "full", null),
    true,
  );
});

test("the authority classifier refuses what admission refuses", () => {
  for (const host of [
    "::ffff:192.168.1.24",
    "::",
    "0.0.0.0",
    "8.8.8.8",
    "box.local",
  ]) {
    assert.equal(isKeylessAllowedAuthority(host), false, host);
  }
  for (const host of ["127.0.0.1", "::1", "192.168.1.24", "fd00::1"]) {
    assert.equal(isKeylessAllowedAuthority(host), true, host);
  }
});
