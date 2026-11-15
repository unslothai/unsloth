// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  buildDecisionApiSnippet,
  buildDecisionCurl,
  decisionApiKey,
  decisionMcpUrl,
} from "../src/features/settings/components/decision-api-snippets.ts";
import { en } from "../src/i18n/locales/en.ts";
import { readSrc } from "./helpers/kit.ts";

const BASE = "https://studio.example.com";
const KEY = "sk-unsloth-abc";
const LOCALES = [
  "ar", "cs", "de", "es-419", "es", "fa", "fr", "he", "hi", "id", "it", "ja", "ko",
  "nl", "pl", "pt-br", "ru", "sv", "th", "tr", "uk", "vi", "zh-CN", "zh-TW",
];

test("the Decision tab and the API Monitor pass the key and tunnel to the section", () => {
  assert.match(
    readSrc("features/settings/components/api-example-tabs.tsx"),
    /<DecisionApiSection\s+apiKey=\{apiKey\}\s+useTunnel=\{useTunnel\}\s+showHeading=\{false\}\s+\{\.\.\.keyless\}/,
  );
  assert.match(
    readSrc("features/api-monitor/api-monitor-page.tsx"),
    /<DecisionApiSection\s+apiKey=\{freshKey\}\s+useTunnel=\{useTunnel\}/,
  );
});

test("the API Monitor's examples follow the keyless policy, as the Settings tab's do", () => {
  const monitor = readSrc("features/api-monitor/api-monitor-page.tsx");
  assert.match(monitor, /useKeylessExampleAccess\(isOwner\)/);
  for (const section of ["MediaApiExamples", "AudioApiExamples", "DecisionApiSection"]) {
    assert.match(monitor, new RegExp(`<${section}[^>]*\\{\\.\\.\\.keyless\\}`), section);
  }
  assert.match(readSrc("features/settings/tabs/api-keys-tab.tsx"), /useKeylessExampleAccess\(isOwner\)/);
  assert.match(
    readSrc("features/settings/hooks/use-keyless-example-access.ts"),
    /\}, \[isOwner, cloudflareUrl, settingsOpen\]\);/,
  );
});

test("the section shows examples and the MCP URL on the host in use", () => {
  const section = readSrc("features/settings/components/decision-api-section.tsx");
  assert.match(section, /<ApiExampleBox/);
  assert.match(section, /<UrlPill url=\{decisionMcpUrl\(base, settings\.mcpUrl\)\}/);
  assert.equal(
    decisionMcpUrl(BASE, "http://127.0.0.1:8888/mcp/decisions/"),
    `${BASE}/mcp/decisions/`,
  );
});

test("Try's JSON view copies the edited body, raw and as cURL", () => {
  const dialog = readSrc("features/settings/components/decision-try-dialog.tsx");
  assert.match(dialog, /label=\{t\("decisions\.copyBody"\)\}\s+text=\{jsonText\}/);
  assert.match(
    dialog,
    /label=\{t\("decisions\.copyCurl"\)\}\s+text=\{buildDecisionCurl\(\{\s+\.\.\.curlTarget,\s+body: jsonText,/,
  );
});

test("curl posts a valid body to the Decision route", () => {
  const snippet = buildDecisionApiSnippet({ base: BASE, apiKey: KEY, language: "curl", os: "unix" });
  assert.match(snippet, /^curl https:\/\/studio\.example\.com\/v1\/systemone \\/);
  assert.match(snippet, /Authorization: Bearer sk-unsloth-abc/);
  const body = JSON.parse(snippet.slice(snippet.indexOf("'") + 1, snippet.lastIndexOf("'")));
  assert.equal(body.model, "default");
  assert.equal(body.questions.category.type, "choice");
});

test("the SDK examples use the SDK's own names and a base without /v1, which it adds itself", () => {
  const python = buildDecisionApiSnippet({ base: BASE, apiKey: KEY, language: "python", os: "unix" });
  assert.match(python, /pip install typesafe-sdk/);
  assert.match(python, /from typesafe_sdk import Choice, TypeSafeClient/);
  assert.match(python, /TypeSafeClient\(base_url="https:\/\/studio\.example\.com", api_key="sk-unsloth-abc"\)/);
  assert.match(python, /client\.system_one\(/);

  const js = buildDecisionApiSnippet({ base: BASE, apiKey: KEY, language: "javascript", os: "unix" });
  assert.match(js, /npm install @typesafe-ai\/sdk/);
  assert.match(js, /baseURL: "https:\/\/studio\.example\.com",\n {2}apiKey: "sk-unsloth-abc"/);
  assert.match(js, /client\.systemOne\(/);
});

test("a body with quotes survives each shell", () => {
  const body = JSON.stringify({ state: "it's café" });
  assert.match(
    buildDecisionCurl({ base: BASE, apiKey: KEY, body, os: "unix" }),
    /-d '\{"state":"it'\\''s café"\}'$/,
  );
  const windows = buildDecisionCurl({ base: BASE, apiKey: KEY, body, os: "windows" });
  assert.ok(windows.startsWith(`$body = '{"state":"it''s caf\\u00e9"}'\n`));
  assert.match(windows, /curl\.exe https:\/\/studio\.example\.com\/v1\/systemone `/);
});

test("the key shown is the real one, keyless only where it works, else the placeholder", () => {
  const local = { base: "http://127.0.0.1:8888", tunnel: false, exposure: null };
  assert.equal(decisionApiKey(KEY, { ...local, scope: "off" }), KEY);
  assert.equal(decisionApiKey(null, { ...local, scope: "inference" }), "not-needed");
  assert.equal(decisionApiKey(null, { ...local, scope: "off" }), "sk-unsloth-YOUR_KEY");
  assert.equal(
    decisionApiKey(null, { ...local, tunnel: true, scope: "inference" }),
    "sk-unsloth-YOUR_KEY",
  );
});

test("the new strings exist in every locale", () => {
  assert.ok(en.settings.apiKeys.decisionApi.codeTitle);
  for (const locale of LOCALES) {
    const source = readSrc(`i18n/locales/${locale}.ts`);
    for (const key of ["codeTitle", "mcpUrl", "mcpUrlDescription", "copyBody", "copyCurl"]) {
      assert.match(source, new RegExp(`\\n\\s+${key}: "`), `${locale} ${key}`);
    }
  }
});
