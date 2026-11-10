// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

// npm overrides win silently even over a dependent pinning a newer version: none may be downgraded.
const HELPERS = ["@radix-ui/react-primitive", "@radix-ui/react-slot", "@radix-ui/react-context"];

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const lock = JSON.parse(readFileSync(path.join(root, "package-lock.json"), "utf8")) as {
  packages: Record<string, { version?: string; dependencies?: Record<string, string> }>;
};

const parse = (v: string) => {
  const m = /^[\^~]?(\d+)\.(\d+)\.(\d+)$/.exec(v);
  assert.ok(m, `unsupported version spec ${v}`);
  return m.slice(1).map(Number);
};

for (const name of HELPERS) {
  test(`${name} resolves to one copy that no dependent outgrows`, () => {
    const copies = Object.keys(lock.packages).filter((k) => k.endsWith(`node_modules/${name}`));
    assert.deepEqual(copies, [`node_modules/${name}`]);
    const [major, minor, patch] = parse(lock.packages[copies[0]].version!);

    for (const [dependent, meta] of Object.entries(lock.packages)) {
      const spec = meta.dependencies?.[name];
      if (!spec) continue;
      const [dMajor, dMinor, dPatch] = parse(spec);
      assert.equal(major, dMajor, `${dependent} wants ${name} ${spec}`);
      assert.ok(
        minor > dMinor || (minor === dMinor && patch >= dPatch),
        `${dependent} wants ${name} ${spec}, override resolves ${major}.${minor}.${patch}`,
      );
    }
  });
}
