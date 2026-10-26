// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The endpoint check tells the owner the single most important finding about a mirror.
 */

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./bundler-resolver.mjs", import.meta.url);

const { hubEndpointCheckVerdict, isOfficialHubEndpoint } = await import(
  "../src/features/settings/lib/hub-endpoint-check.ts"
);

test("a mirror that serves everything itself passes", () => {
  assert.equal(
    hubEndpointCheckVerdict({ api: "ok", download: "mirror", xet: "none" }),
    "ok",
  );
  assert.equal(
    hubEndpointCheckVerdict({ api: "ok", download: "mirror", xet: "mirror" }),
    "ok",
  );
});

test("a whole-site redirect to huggingface.co outranks everything else", () => {
  assert.equal(
    hubEndpointCheckVerdict({
      api: "official",
      download: "official",
      xet: "none",
    }),
    "redirected",
  );
  assert.equal(
    hubEndpointCheckVerdict({
      api: "official",
      download: "failed",
      xet: "none",
    }),
    "redirected",
  );
});

test("a failed probe is reported before downloads leaving the mirror", () => {
  assert.equal(
    hubEndpointCheckVerdict({
      api: "failed",
      download: "official",
      xet: "none",
    }),
    "failed",
  );
  assert.equal(
    hubEndpointCheckVerdict({ api: "ok", download: "failed", xet: "none" }),
    "failed",
  );
  assert.equal(
    hubEndpointCheckVerdict({ api: "ok", download: "mirror", xet: "failed" }),
    "failed",
  );
});

test("downloads or Xet chunks fetched from Hugging Face are called out", () => {
  assert.equal(
    hubEndpointCheckVerdict({ api: "ok", download: "official", xet: "none" }),
    "downloadsOfficial",
  );
  assert.equal(
    hubEndpointCheckVerdict({ api: "ok", download: "mirror", xet: "official" }),
    "xetOfficial",
  );
});

test("only a mirror can be checked", () => {
  for (const raw of [
    "",
    "  ",
    "huggingface.co",
    "https://HuggingFace.co/",
    "hf.co",
  ]) {
    assert.equal(isOfficialHubEndpoint(raw), true, raw);
  }
  for (const raw of [
    "hf-mirror.com",
    "https://huggingface.co.evil.example",
    "http://127.0.0.1:8080",
    "https://[",
  ]) {
    assert.equal(isOfficialHubEndpoint(raw), false, raw);
  }
});
