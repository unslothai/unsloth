// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const API = readSrc("features/chat/api/mcp-servers-api.ts");
const CATALOG = readSrc("features/chat/editor-mcp-catalog.tsx");
const SETUP = readSrc("features/chat/editor-mcp-setup.tsx");
const COMPOSER = readSrc("features/chat/mcp-composer-button.tsx");
const BLENDER = readSrc("features/chat/blender-mcp-setup.tsx");
const GUIDE = readSrc("features/chat/mcp-install-guide.ts");
const PERMISSION = readSrc("features/chat/permission-mode-select.tsx");

test("the MCP menu keeps Blender and adds the three managed editors after it", () => {
  for (const id of ["unity", "unreal", "godot", "figma", "roblox"]) {
    assert.match(CATALOG, new RegExp(`\\n  ${id}: \\{`));
  }
  assert.match(API, /export type EditorMcpId = "unity" \| "unreal" \| "godot" \| "figma" \| "roblox";/);
  const blender = COMPOSER.indexOf('<span className="truncate">Blender</span>');
  const editors = COMPOSER.indexOf(".filter((id) => !EDITOR_MCP[id].more)");
  const more = COMPOSER.indexOf("<DropdownMenuSubTrigger>More</DropdownMenuSubTrigger>");
  assert.ok(blender > 0 && editors > blender && more > editors);
  for (const id of ["godot", "figma", "roblox"]) {
    assert.match(CATALOG, new RegExp(`${id}: \\{[\\s\\S]*?more: true,`));
  }
  assert.match(COMPOSER, /<BlenderMcpSetup/);
  assert.match(COMPOSER, /<EditorMcpSetup\s+key=\{editorOpen\}\s+editorId=\{editorOpen\}/);
});

test("editor setup posts to its own managed routes; only Unreal has a project field, only Roblox has no port", () => {
  assert.match(API, /mcpRequest\(`\/builtins\/\$\{id\}`, \{ method: "PUT", body: payload \}\)/);
  assert.match(API, /mcpRequest\(`\/builtins\/\$\{id\}\/test`, \{ method: "POST", body: payload \}\)/);
  assert.equal(CATALOG.match(/projectField: true/g)?.length, 1);
  assert.match(CATALOG, /unreal: \{[\s\S]*?projectField: true/);
  assert.equal(CATALOG.match(/noPort: true/g)?.length, 1);
  assert.match(CATALOG, /roblox: \{[\s\S]*?noPort: true/);
  assert.match(SETUP, /port: copy\.noPort \? null : Number\(port\), project_path: projectPath\.trim\(\)/);
  assert.match(SETUP, /disabled=\{locked \|\| !config\.available \|\| !ready \|\| !consent\}/);
});

test("install state is looked up only for the app being set up, which then links its install guide", () => {
  assert.doesNotMatch(COMPOSER, /listMcpBuiltins|Not installed/);
  assert.match(SETUP, /listMcpBuiltins\(wait, editorId\)/);
  assert.match(BLENDER, /listMcpBuiltins\(wait, "blender"\)/);
  assert.match(API, /`\/builtins\?detect=\$\{encodeURIComponent\(detect\)\}`/);
  assert.match(API, /installed\?: boolean \| null;/);
  for (const setup of [SETUP, BLENDER]) {
    assert.match(setup, /config\?\.installed === false && !config\.is_enabled/);
    assert.match(setup, />Guide to Install</);
    // Closing mid-request would unmount the setup before it finishes enabling.
    assert.match(setup, /disabled=\{locked\} onClick=\{\(\) => openInstallGuide\(/);
  }
  assert.match(SETUP, /openInstallGuide\(copy\.guide, onClose\)/);
  assert.match(BLENDER, /openInstallGuide\("blender", onClose\)/);
  for (const [id, section] of [["unity", "unity"], ["unreal", "unreal-engine"], ["godot", "godot"], ["figma", "figma"], ["roblox", "roblox-studio"]]) {
    assert.match(CATALOG, new RegExp(`\\n  ${id}: \\{[\\s\\S]*?guide: "${section}",`));
  }
  assert.equal(COMPOSER.match(/onClose=\{/g)?.length, 2);
  assert.match(GUIDE, /"https:\/\/unsloth\.ai\/docs\/basics\/mcp\/mcp-engines"/);
  assert.match(GUIDE, /close\?\.\(\);\n  if \(!openUrlInBrowserPanel\(url\)\) openLink\(url\);/);
});

test("Unreal's step says its MCP server is off until Auto Start Server is on", () => {
  assert.match(CATALOG, /Auto Start Server/);
  assert.doesNotMatch(CATALOG, /starts with the editor on port 8000/);
});

test("editors that run code outside the sandbox say so before consent", () => {
  for (const id of ["unity", "unreal", "godot"]) {
    assert.match(CATALOG, new RegExp(`\\n  ${id}: \\{((?!\\n  \\},)[\\s\\S])*?sandboxWarning: true,`));
  }
  for (const id of ["figma", "roblox"]) {
    assert.doesNotMatch(CATALOG, new RegExp(`\\n  ${id}: \\{((?!\\n  \\},)[\\s\\S])*?sandboxWarning`));
  }
  assert.match(SETUP, /t\("settings\.sandbox\.editorMcpWarning", \{ name: copy\.name \}\)/);
  // Run automatically must not promise the sandbox covers MCP tools (#19769).
  assert.doesNotMatch(PERMISSION, /keep code in the sandbox/);
  assert.match(PERMISSION, /Never ask\. MCP runs outside the sandbox/);
  assert.match(SETUP, /aria-describedby=\{copy\.sandboxWarning \? `\$\{editorId\}-mcp-sandbox-warning` : undefined\}/);
});
