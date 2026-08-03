// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("Python sandbox images target the desktop backend and pass the Tauri CSP", async () => {
  const component = await readFile(
    new URL(
      "../src/components/assistant-ui/tool-ui-python.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const config = JSON.parse(
    await readFile(
      new URL("../../src-tauri/tauri.conf.json", import.meta.url),
      "utf8",
    ),
  ) as { app?: { security?: { csp?: string } } };

  assert.ok(component.includes('import { apiUrl } from "@/lib/api-base";'));
  assert.ok(
    component.includes("src={apiUrl(") &&
      component.includes("`/api/inference/sandbox/"),
    "the image URL must use the dynamically selected desktop backend origin",
  );

  const csp = config.app?.security?.csp ?? "";
  const imgSrc = csp
    .split(";")
    .map((directive) => directive.trim())
    .find((directive) => directive.startsWith("img-src "));
  assert.ok(imgSrc, "desktop CSP must define img-src");
  assert.ok(
    imgSrc.split(" ").includes("http://127.0.0.1:*"),
    "desktop CSP must allow sandbox images from the loopback backend",
  );
});
