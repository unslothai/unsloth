// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

import { endpointKey } from "../src/features/hub/lib/endpoint-key.ts";

const DEFAULT = "https://huggingface.co";
const same = (a: string, b: string) => endpointKey(a) === endpointKey(b);

test("cosmetic differences do not turn the official Hub into a mirror", () => {
  // Proxying the default gives up direct access, and its own CSP entry, for
  // nothing. Every one of these is the same deployment.
  for (const variant of [
    "https://huggingface.co/",
    "https://huggingface.co///",
    "https://HuggingFace.co",
    "https://huggingface.co:443",
  ]) {
    assert.ok(same(variant, DEFAULT), `${variant} is the official Hub`);
  }
});

test("a mirror mounted on the Hub host is still a mirror", () => {
  // HF_ENDPOINT carries a path, and the backend builds its URLs from all of it.
  // Compared on origin alone this was taken for the default and fetched direct,
  // so the browser read the public repo while the backend read the mirror's.
  assert.ok(!same("https://huggingface.co/hf", DEFAULT));
  assert.ok(!same("https://huggingface.co/proxy/v1", DEFAULT));
  // The trailing slash is still cosmetic once a path is present.
  assert.ok(same("https://huggingface.co/hf/", "https://huggingface.co/hf"));
  // And the path is case-sensitive upstream, unlike the host.
  assert.ok(!same("https://huggingface.co/HF", "https://huggingface.co/hf"));
});

test("a different host is a mirror whatever its path", () => {
  assert.ok(!same("https://hf-mirror.example", DEFAULT));
  assert.ok(!same("http://huggingface.co", DEFAULT), "the scheme is part of it");
});

test("an unparseable endpoint compares as itself rather than throwing", () => {
  // hubProxyFirst has its own catch, but this runs before it: returning the raw
  // string keeps a malformed value distinct from the default, so the safe route
  // is chosen instead of a direct fetch at the wrong host.
  assert.equal(endpointKey("not a url"), "not a url");
  assert.ok(!same("not a url", DEFAULT));
});

test("hubProxyFirst decides on the whole endpoint", async () => {
  const src = await readFile(
    new URL("../src/features/hub/lib/hub-endpoint.ts", import.meta.url),
    "utf8",
  );
  const at = src.indexOf("export function hubProxyFirst");
  assert.notEqual(at, -1);
  const body = src.slice(at, src.indexOf("\n}", at));
  assert.ok(body.includes("endpointKey(endpoint)"));
  assert.ok(body.includes("endpointKey(DEFAULT_HUB_ENDPOINT)"));
  // .origin drops the path, which is the whole of what this fixes.
  assert.ok(!body.includes(".origin"));
});
