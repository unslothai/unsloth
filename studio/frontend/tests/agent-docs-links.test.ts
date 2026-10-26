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
    /\{isOwner \? \([\s\S]*?setAgentsTabAgent\(agent\);[\s\S]{0,300}else useSettingsDialogStore\.getState\(\)\.setActiveTab\("agents"\);[\s\S]*?\) : \([\s\S]*?href=\{agentDetails\.docsUrl\}/,
  );
  assert.match(src, /if \(agentsOnly\) useSettingsDialogStore\.getState\(\)\.openDialog\("agents"\);/);
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
  // one header row, no divider: the choices, then Platform pushed to its end
  assert.match(
    parts,
    /<div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 px-2 pt-1\.5">\s*\{pills\}\s*<div className="ml-auto">\s*<PlatformToggle value=\{os\} onChange=\{onOsChange\} \/>\s*<\/div>\s*<\/div>/,
  );
  // copy stays put over the code area, on its first line
  assert.match(
    parts,
    /<button\s*type="button"\s*onClick=\{handleCopy\}\s*className="absolute top-2\.5 right-3 [^"]*"[\s\S]*?<\/button>\s*<HighlightedCode/,
  );
  // code scrolling up fades out under the header row
  assert.match(
    parts,
    /\[mask-image:linear-gradient\(to_bottom,transparent,#000_0\.75rem\)\]/,
  );
  // Language and Platform keep the old usage-example pills; what the example does is a toggle
  assert.match(parts, /variant = "pills"/);
  assert.match(
    parts,
    /toggle\s*\? "inline-flex h-9 items-center px-3 text-ui-12"\s*: "px-2\.5 py-1 text-ui-11"/,
  );
});

test("Request and Image or video are toggles above the code box; Audio keeps its grouped Example dropdown", () => {
  const usage = readSrc("features/settings/components/usage-examples.tsx");
  const media = readSrc("features/settings/components/media-api-examples.tsx");
  const audio = readSrc("features/settings/components/audio-api-examples.tsx");
  for (const [src, key] of [
    [usage, "exampleRequest"],
    [media, "exampleKind"],
  ]) {
    const toggle = src.search(
      new RegExp(
        `<ExampleToggle\\s*label=\\{t\\("settings\\.apiKeys\\.${key}"\\)\\}\\s*variant="toggle"`,
      ),
    );
    assert.ok(toggle > 0, key);
    assert.ok(toggle < src.indexOf("<ExampleCodeBox"), key);
    // the box header holds only the Language pills (and its own Platform pills)
    assert.match(
      src,
      /pills=\{\s*<ExampleToggle\s*label=\{t\("settings\.apiKeys\.exampleLanguage"\)\}[^/]*\/>\s*\}/,
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
    /"relative \[&>section\]:max-h-none \[&>section\]:min-h-24 \[&>section\]:pr-8 \[&_\[data-streamdown=code-block-header\]\]:hidden",\s*height === null\s*\? fitContent\s*\? "\[&>section\]:max-h-\[60dvh\]"\s*: tall && "\[&>section\]:h-46"\s*: "\[&>section\]:h-\(--example-h\)"/,
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
