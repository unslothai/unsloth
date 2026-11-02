// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import test from "node:test";

import {
  apiExampleLanguage,
  apiExampleModelName,
  buildApiExample,
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

const SAMPLING = { temperature: 1, top_p: 0.95, top_k: -1, min_p: 0.01 };

test("curl and JavaScript put every sampling field at the top level of the body", () => {
  const curl = buildApiExample({ baseUrl: "http://h/v1", model: "m", sampling: SAMPLING, language: "curl", os: "unix" });
  assert.match(curl, /^curl http:\/\/h\/v1\/chat\/completions \\/);
  const body = JSON.parse(curl.slice(curl.indexOf("-d '") + 4, curl.lastIndexOf("'")));
  assert.deepEqual(body, { model: "m", messages: [{ role: "user", content: "Hello!" }], ...SAMPLING });

  const js = buildApiExample({ baseUrl: "http://h/v1", model: "m", sampling: SAMPLING, language: "javascript", os: "unix" });
  assert.match(js, /baseURL: "http:\/\/h\/v1"/);
  assert.match(js, /\n {2}top_k: -1,\n {2}min_p: 0\.01,/);
  assert.doesNotMatch(js, /extra_body/);
});

test("Windows curl sends the body through a file PowerShell cannot mangle", () => {
  const win = buildApiExample({ baseUrl: "http://h/v1", model: "it's", sampling: {}, language: "curl", os: "windows" });
  assert.match(win, /^\$body = '/);
  assert.match(win, /"model": "it''s"/);
  assert.match(win, /curl\.exe http:\/\/h\/v1\/chat\/completions `/);
  assert.match(win, /-d "@body\.json"$/);
  // Python and JavaScript read the same on every platform.
  const args = { baseUrl: "http://h/v1", model: "m", sampling: SAMPLING } as const;
  for (const language of ["python", "javascript"] as const) {
    assert.equal(
      buildApiExample({ ...args, language, os: "windows" }),
      buildApiExample({ ...args, language, os: "unix" }),
    );
  }
});

test("the stored settings pick names the language", () => {
  assert.equal(apiExampleLanguage("pythonTools", "curl"), "python");
  assert.equal(apiExampleLanguage("javascriptAdvanced", "curl"), "javascript");
  assert.equal(apiExampleLanguage(null, "python"), "python");
  assert.equal(apiExampleLanguage("ruby", "curl"), "curl");
});

test("every example box follows the stored language and platform, not a copy taken at mount", async () => {
  const { readFileSync } = await import("node:fs");
  const box = readFileSync(
    new URL("../src/features/api-monitor/components/api-example-box.tsx", import.meta.url),
    "utf8",
  );
  // The quick start and the closed dialog are mounted together; local state would let them drift apart.
  assert.doesNotMatch(box, /useState/);
  assert.match(box, /useSettingsPanelPrefsStore\(\(s\) => s\.apiExampleLang\)/);
  assert.match(box, /useSettingsPanelPrefsStore\(\(s\) => s\.apiExampleOs\)/);
});
