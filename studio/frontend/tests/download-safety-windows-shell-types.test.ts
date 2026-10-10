// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import policy from "../src/features/browser/dangerous-file-types.json" with { type: "json" };

test("Windows theme and search-connector files are treated as dangerous", () => {
  // Opening a crafted .theme or .searchConnector-ms reaches an attacker SMB share and leaks the NTLM hash.
  for (const ext of ["theme", "themepack", "deskthemepack", "searchconnector-ms"]) {
    assert.ok(policy.extensions.includes(ext), ext);
  }
});
