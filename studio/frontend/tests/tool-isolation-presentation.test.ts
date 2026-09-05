// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Every limitation identifier a backend can emit has user-facing text, and the two Windows
// profiles are labelled apart. The backend sources are read directly so a new limitation code
// fails here instead of rendering as a raw identifier in the dropdown.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  TOOL_ISOLATION_LIMITATION_TEXT,
  backendLabel,
} from "../src/features/chat/tool-isolation-labels.ts";

const BACKEND_SOURCES = [
  "../../backend/core/inference/os_sandbox.py",
  "../../backend/core/inference/windows_lpac.py",
];

function backendLimitationCodes(): Set<string> {
  const codes = new Set<string>();
  for (const relative of BACKEND_SOURCES) {
    const source = readFileSync(new URL(relative, import.meta.url), "utf8");
    for (const match of source.matchAll(/_LIMITATION_[A-Z0-9_]+\s*=\s*"([a-z0-9_]+)"/g)) {
      codes.add(match[1]);
    }
    for (const block of source.matchAll(/limitations\s*=\s*\(([^)]*)\)/g)) {
      for (const literal of block[1].matchAll(/"([a-z][a-z0-9_]+)"/g)) {
        codes.add(literal[1]);
      }
    }
  }
  return codes;
}

test("every backend limitation code has display text", () => {
  const codes = backendLimitationCodes();
  assert.ok(codes.has("nested_userns_blocked_by_seccomp"));
  assert.ok(codes.has("all_application_packages_ambient_read"));
  assert.ok(codes.has("ipv6_unavailable_on_host"));
  assert.ok(codes.has("detached_descendant_cleanup_unverified"));
  for (const code of codes) {
    const text = TOOL_ISOLATION_LIMITATION_TEXT[code];
    assert.ok(text && text.length > 20, `no display text for limitation ${code}`);
    assert.doesNotMatch(text, /\u2014/, `em dash in text for ${code}`);
  }
});

test("the Windows fallback profile is labelled AppContainer, not LPAC", () => {
  assert.equal(backendLabel("windows-lpac", "windows", "windows-lpac-preview-v1"), "LPAC (Windows)");
  assert.equal(backendLabel("windows-lpac", "windows", null), "LPAC (Windows)");
  assert.equal(
    backendLabel("windows-lpac", "windows", "windows-appcontainer-preview-v1"),
    "AppContainer (Windows)",
  );
  assert.equal(backendLabel("macos-seatbelt", "macos", "macos-seatbelt-preview-v1"), "Seatbelt (lifecycle unverified)");
  assert.equal(backendLabel("linux-bubblewrap", "native_linux", "linux-bubblewrap-v2"), "Bubblewrap");
  assert.equal(backendLabel("linux-bubblewrap", "wsl2", null), "Bubblewrap (WSL2)");
});
