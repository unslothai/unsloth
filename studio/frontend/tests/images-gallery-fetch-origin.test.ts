// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const api = readSrc("features/images/api.ts");

test("an authenticated gallery fetch refuses a URL off this Studio before sending the bearer", () => {
  const body = api.slice(api.indexOf("export async function fetchGalleryResponse("));
  const guard = body.indexOf("if (!isStudioUrl(url))");
  assert.ok(guard > 0 && guard < body.indexOf("authFetch(url)"));
});

test("a gallery id from a linked instance cannot climb out of the gallery path", () => {
  assert.doesNotMatch(api, /gallery\/\$\{id\}/);
  assert.equal(new URL(`/p/gallery/${encodeURIComponent("../../../api/auth/x")}`, "http://h").pathname.includes("/api/auth/"), false);
});
