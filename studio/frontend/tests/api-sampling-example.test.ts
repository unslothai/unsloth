// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import test from "node:test";

import {
  apiExampleModelName,
  buildApiSamplingExample,
} from "../src/features/api-monitor/api-sampling-example.ts";

test("client fields are keyword arguments, Unsloth fields ride extra_body", () => {
  const snippet = buildApiSamplingExample({
    baseUrl: "http://127.0.0.1:8888/v1",
    model: "unsloth/gemma-3-270m-it-GGUF:Q4_K_M",
    sampling: {
      temperature: 1,
      top_p: 0.95,
      presence_penalty: 0,
      top_k: 64,
      min_p: 0,
      repetition_penalty: 1,
    },
  });
  assert.match(snippet, /base_url="http:\/\/127\.0\.0\.1:8888\/v1"/);
  assert.match(snippet, /model="unsloth\/gemma-3-270m-it-GGUF:Q4_K_M"/);
  assert.match(snippet, /\n {4}temperature=1,\n {4}top_p=0\.95,\n {4}presence_penalty=0,/);
  assert.match(snippet, /extra_body=\{\n {8}"top_k": 64,\n {8}"min_p": 0,\n {8}"repetition_penalty": 1,\n {4}\},/);
});

test("no values from the server leaves a request that still runs", () => {
  const snippet = buildApiSamplingExample({
    baseUrl: "http://h/v1",
    model: "m",
    sampling: {},
  });
  assert.doesNotMatch(snippet, /extra_body|temperature/);
  assert.match(snippet, /messages=\[/);
});

test("the quant is pinned once", () => {
  assert.equal(apiExampleModelName("unsloth/x-GGUF", "Q4_K_M"), "unsloth/x-GGUF:Q4_K_M");
  assert.equal(apiExampleModelName("unsloth/x-GGUF:Q8_0", "Q4_K_M"), "unsloth/x-GGUF:Q8_0");
  assert.equal(apiExampleModelName("unsloth/x", null), "unsloth/x");
});
