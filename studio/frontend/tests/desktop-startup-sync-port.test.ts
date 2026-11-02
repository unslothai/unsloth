// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Desktop mounts the root before the backend port exists, and authFetch resolves its URL once,
// so the startup pin and override sync must wait for the Desktop shell.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path: string) => readFile(new URL(path, import.meta.url), "utf8");

test("the startup pin and override sync waits for the Desktop shell", async () => {
  const root = await read("../src/app/routes/__root.tsx");
  const start = root.indexOf("void backfillModelOverrides();");
  assert.ok(start > 0, "root no longer starts the override backfill");
  const effectStart = root.lastIndexOf("useEffect(() => {", start);
  const effect = root.slice(effectStart, root.indexOf("]);", start) + 3);
  assert.match(effect, /void hydratePins\(\);/);
  assert.match(effect, /if \(isAuthFlowRoute \|\| !desktopShellReady\) \{\s*return;\s*\}/);
  assert.match(effect, /\}, \[isAuthFlowRoute, desktopShellReady\]\);$/);
  assert.ok(root.indexOf("const desktopShellReady = useDesktopShellReady();") < effectStart);

  // Open from the start on the web; on Desktop only once the backend runs and sign-in finished.
  assert.match(await read("../src/app/desktop-shell-ready.ts"), /let ready = !isTauri;/);
  const provider = await read("../src/app/provider.tsx");
  assert.match(provider, /const canMountApp = status === "running" && desktopAuthReady;/);
  assert.match(provider, /setDesktopShellReady\(canMountApp && appShellReady\);/);
});
