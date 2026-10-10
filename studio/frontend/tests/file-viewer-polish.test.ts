// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { AiSpeechIcon, TextToSpeechIcon } = await import("../src/lib/hugeicons-derived.ts");
const { ATTACHMENT_KIND_ICONS } = await import("../src/features/chat/lib/attachment-file-kind.ts");
const { audioWorkflowTab } = await import("../src/features/audio/workflows.ts");

const read = (file: string) => readFileSync(new URL(`../src/${file}`, import.meta.url), "utf8");

test("audio reads as ai-speech everywhere and Text to Speech as text-to-speech", () => {
  assert.equal(ATTACHMENT_KIND_ICONS.audio, AiSpeechIcon);
  assert.equal(audioWorkflowTab("speak").icon, TextToSpeechIcon);
  assert.notEqual(AiSpeechIcon, TextToSpeechIcon);
  for (const file of ["components/app-sidebar.tsx", "components/command-palette.tsx", "features/library/file-kind.ts"]) {
    assert.doesNotMatch(read(file), /AudioWave01Icon/, file);
  }
});

test("a picture sits where fit puts it before it is measured, so opening it never jumps", () => {
  const zoom = read("components/media-zoom.tsx");
  // `inset` would be cleared after `left` is set once measured, dropping the picture to the left edge.
  assert.doesNotMatch(zoom, /\{ inset: 0 \}/);
  assert.match(zoom, /left: inset\.left,\s*top: inset\.top,\s*width: `calc\(100% - \$\{inset\.left \+ inset\.right\}px\)`/);
});

test("the lightbox backdrop is a faint grey in light mode and near black in dark", () => {
  const css = read("index.css");
  assert.match(css, /\.media-lightbox-overlay \{\s*background: color-mix\(in oklab, color-mix\(in oklab, var\(--background\), black 7%\) 64%, transparent\);/);
  assert.match(css, /\.dark \.media-lightbox-overlay \{\s*background: color-mix\(in oklab, color-mix\(in oklab, var\(--background\), black 45%\) 70%, transparent\);/);
  assert.match(read("components/media-viewer.tsx"), /"media-lightbox-overlay bg-transparent /);
});

test("the video player uses Studio's icons and full screen turns into exit full screen", () => {
  const video = read("features/browser/video-file.tsx");
  assert.doesNotMatch(video, /from "lucide-react"/);
  assert.match(video, /addEventListener\("fullscreenchange", sync\)/);
  assert.match(video, /icon=\{fullscreened \? ArrowShrink01Icon : ArrowExpand01Icon\}/);
  assert.match(video, /fullscreened \? t\("browser\.video\.exitFullscreen"\) : t\("browser\.video\.fullscreen"\)/);
});

test("video tabs share the document bar: Copy and Open splits and a download icon, no Save as", () => {
  const panel = read("features/browser/browser-panel.tsx");
  assert.doesNotMatch(panel, /VideoFileToolbar|isVideoEntry|browser\.video\.saveAs|browser\.video\.copyName/);
  assert.match(panel, /return kind === "html" \|\| kind === "code";\n\}/);
  assert.match(panel, /<CopySplit tab=\{tab\} entry=\{entry\} blob=\{blob\}/);
  assert.match(panel, /<OpenSplit key=\{entry\.fileId\} entry=\{entry\} blob=\{blob\} \/>/);
  // A clip or a track has no zoom and nothing to mark.
  assert.match(panel, /const timed = media === "video" \|\| media === "audio";/);
  assert.match(panel, /\{timed \? null : \(\s*<ScaleMenu/);
});

test("the desktop Open lists apps, opens a local copy, and never opens programs", () => {
  const openWith = read("features/browser/open-with.ts");
  assert.match(openWith, /const copies = new WeakMap<Blob, Map<string, Promise<LocalCopy>>>\(\);/);
  assert.match(openWith, /"browser_file_local_copy"/);
  const panel = read("features/browser/browser-panel.tsx");
  assert.match(panel, /const runsCode = isDangerousDownload\(entry\.name\);/);
  const rust = readFileSync(new URL("../../src-tauri/src/browser_open_with.rs", import.meta.url), "utf8");
  assert.match(rust, /if crate::browser_downloads::runs_code\(&path\) \{\s*return Err/);
  assert.match(rust, /if !app_paths\(path\)\.0\.iter\(\)\.any\(\|app\| app == with\)/);
  const main = readFileSync(new URL("../../src-tauri/src/main.rs", import.meta.url), "utf8");
  for (const command of ["browser_file_local_copy", "browser_file_apps", "browser_file_open", "browser_file_reveal"]) {
    assert.match(main, new RegExp(`browser_open_with::${command},`), command);
  }
});
