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

test("every API example section has a platform toggle that follows the user's machine", () => {
  const parts = readSrc("features/settings/components/api-example-parts.tsx");
  assert.match(
    parts,
    /const os: ExampleOs =\s*override \?\?\s*\(\(isLoopbackUrl\(base\) \? deviceType === "windows" : isWindowsClient\)/,
  );
  assert.match(parts, /p\.startsWith\("win"\) \|\| p\.includes\("windows"\)/);
  for (const name of [
    "usage-examples",
    "audio-api-examples",
    "media-api-examples",
  ]) {
    const src = readSrc(`features/settings/components/${name}.tsx`);
    assert.match(src, /const \[os, setOs\] = useExampleOs\(base\);/, name);
    // the code box draws the Platform row, under the Language pills
    assert.match(src, /os=\{os\}\s*onOsChange=\{setOs\}/, name);
    // no section hides the toggle behind its curl pick any more
    assert.doesNotMatch(src, /lang === "curl" \? \(\s*<div/, name);
    assert.match(
      src,
      /pills=\{[\s\S]*?<ExampleToggle\s*label=\{t\("settings\.apiKeys\.exampleLanguage"\)\}/,
      name,
    );
    assert.equal(
      src.match(/<ExampleToggle\b/g)?.length,
      name === "audio-api-examples" ? 1 : 2,
      name,
    );
  }
  // the header rows are the old usage-example pill rows: Language row with copy, then Platform
  assert.match(
    parts,
    /<div className=\{HEADER_ROW\}>\s*<PlatformToggle value=\{os\} onChange=\{onOsChange\} \/>/,
  );
  assert.match(
    parts,
    /"relative whitespace-nowrap rounded-full px-2\.5 py-1 text-ui-11 font-medium/,
  );
  assert.doesNotMatch(parts, /hub-tab-toggle/);
});

test("Request and Image or video share the Language row; Audio keeps its grouped Example dropdown", () => {
  const usage = readSrc("features/settings/components/usage-examples.tsx");
  const media = readSrc("features/settings/components/media-api-examples.tsx");
  const audio = readSrc("features/settings/components/audio-api-examples.tsx");
  for (const [src, key] of [
    [usage, "exampleRequest"],
    [media, "exampleKind"],
  ]) {
    assert.match(
      src,
      new RegExp(
        `pills=\\{\\s*<>\\s*<ExampleToggle\\s*label=\\{t\\("settings\\.apiKeys\\.${key}"\\)\\}[\\s\\S]*?<PillDivider />\\s*<ExampleToggle\\s*label=\\{t\\("settings\\.apiKeys\\.exampleLanguage"`,
      ),
      key,
    );
  }
  assert.doesNotMatch(media, /@\/components\/ui\/select/);
  assert.match(
    audio,
    /<Select value=\{example\} onValueChange=\{pickExample\}>/,
  );
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

test("the example code box starts short, drags taller, and its copy button is icon-only", () => {
  const src = readSrc("features/settings/components/api-example-parts.tsx");
  assert.match(src, /const START_LINES = 8;/);
  assert.match(src, /code\.split\("\\n"\)\.length > START_LINES/);
  // the cap is lifted so a drag can grow it; a height is set, from the drag once there is one
  assert.match(
    src,
    /"\[&>section\]:max-h-none \[&>section\]:min-h-24 \[&>section\]:pr-8 \[&_\[data-streamdown=code-block-header\]\]:hidden",\s*height === null\s*\? tall && "\[&>section\]:h-46"\s*: "\[&>section\]:h-\(--example-h\)"/,
  );
  // our own handle: CSS resize draws a native grabber the rounded corner clips into an arc
  assert.doesNotMatch(src, /resize-y|webkit-resizer/);
  assert.match(src, /onPointerDown=\{startResize\}/);
  assert.match(src, /setPointerCapture\(event\.pointerId\)/);
  for (const name of [
    "usage-examples",
    "audio-api-examples",
    "media-api-examples",
    "api-example-parts",
  ]) {
    const file = readSrc(`features/settings/components/${name}.tsx`);
    assert.doesNotMatch(file, /t\("settings\.apiKeys\.copy"\)/, name);
    assert.doesNotMatch(
      file,
      /SegmentedPill group=\{`\$\{pillId\}-(lang|tab|kind|run)`\}/,
      name,
    );
  }
});
