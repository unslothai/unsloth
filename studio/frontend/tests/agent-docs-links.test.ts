// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  SUPPORTED_AGENTS,
  UNSLOTH_START_DOCS_URL,
} from "../src/features/settings/components/coding-agent-list.ts";
import { readSrc } from "./helpers/kit.ts";

test("Pi and Mistral Vibe docs open the unsloth start page", () => {
  for (const id of ["pi", "vibe"]) {
    const agent = SUPPORTED_AGENTS.find((a) => a.id === id);
    assert.equal(agent?.docsUrl, UNSLOTH_START_DOCS_URL, id);
  }
});

test("the API tab hands owners to the Agents tab and gives other accounts the agent's docs", () => {
  const src = readSrc("features/settings/components/usage-examples.tsx");
  assert.match(
    src,
    /\{isOwner \? \([\s\S]*?setAgentsTabAgent\(agent\);\s*useSettingsDialogStore\.getState\(\)\.setActiveTab\("agents"\);[\s\S]*?\) : \([\s\S]*?href=\{agentDetails\.docsUrl\}/,
  );
  const visibility = readSrc("features/settings/settings-tab-visibility.ts");
  assert.match(visibility, /OWNER_ONLY_SETTINGS_TABS[\s\S]*?"agents"/);
  assert.doesNotMatch(src, /const DOC_LINKS/);
  assert.doesNotMatch(src, /href: "https?:\/\//);
});

test("the API tab's platform follows the user's machine until they pick one", () => {
  const src = readSrc("features/settings/components/usage-examples.tsx");
  assert.match(
    src,
    /const os: Os =\s*osOverride \?\?\s*\(\(isLoopbackUrl\(base\) \? deviceType === "windows" : isWindowsClient\)/,
  );
  assert.match(src, /p\.startsWith\("win"\) \|\| p\.includes\("windows"\)/);
});

test("with no model loaded the example still renders, naming a placeholder", () => {
  const src = readSrc("features/settings/components/usage-examples.tsx");
  assert.match(src, /const MODEL_PLACEHOLDER = "YOUR_MODEL";/);
  assert.match(
    src,
    /buildSnippets\(base, key, toolsKey, model \?\? MODEL_PLACEHOLDER, os\)/,
  );
  assert.doesNotMatch(src, /\{snippets \? \(/);
});

test("the usage example collapses past eight lines and its copy button is icon-only", () => {
  const src = readSrc("features/settings/components/usage-examples.tsx");
  assert.match(src, /const COLLAPSED_LINES = 8;/);
  assert.match(
    src,
    /snippets\[lang\]\.split\("\\n"\)\.length > COLLAPSED_LINES/,
  );
  assert.match(src, /expanded && "\[&>section\]:max-h-none"/);
  assert.doesNotMatch(src, /t\("settings\.apiKeys\.copy"\)/);
});
