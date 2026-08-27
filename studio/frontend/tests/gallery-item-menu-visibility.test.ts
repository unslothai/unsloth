// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("gallery overlay actions follow interaction state instead of stale mouse focus", async () => {
  const [menu, imagesPage, videoPage] = await Promise.all([
    readFile(
      new URL("../src/components/gallery-item-menu.tsx", import.meta.url),
      "utf8",
    ),
    readFile(
      new URL("../src/features/images/images-page.tsx", import.meta.url),
      "utf8",
    ),
    readFile(
      new URL("../src/features/video/video-page.tsx", import.meta.url),
      "utf8",
    ),
  ]);

  const closedClasses =
    /:\s*"([^"]*opacity-0[^"]*)"/.exec(menu)?.[1].split(" ") ?? [];
  assert.match(menu, /active && open\s*\n\s*\? "opacity-100"/);
  assert.ok(
    closedClasses.includes("group-hover:opacity-100"),
    "mouse hover must reveal the action",
  );
  assert.ok(
    closedClasses.includes("has-[button:focus-visible]:opacity-100"),
    "keyboard focus must reveal the action",
  );
  assert.ok(
    closedClasses.includes("pointer-coarse:opacity-100"),
    "touch users need a persistent action",
  );
  assert.ok(
    !closedClasses.includes("focus-within:opacity-100"),
    "plain restored focus must not reveal the action",
  );

  assert.doesNotMatch(imagesPage, /focus-within:opacity-100/);
  assert.doesNotMatch(videoPage, /focus-within:opacity-100/);
});
