// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

test("a collapsed sidebar renders no resize edge, so its border has no hover or tooltip", async () => {
  const sidebar = await readSrcAsync("components/ui/sidebar.tsx");
  const at = sidebar.indexOf("<SidebarResizeHandle\n");
  assert.notEqual(at, -1, "could not find where the sidebar renders its resize edge");
  const guard = sidebar.slice(sidebar.lastIndexOf("{", at), at);
  assert.match(guard, /state === "expanded"/);
  assert.match(guard, /!collapseToZero \|\| pinned/);
});
