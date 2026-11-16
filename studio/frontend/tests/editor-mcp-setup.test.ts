// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const API = readSrc("features/chat/api/mcp-servers-api.ts");
const CATALOG = readSrc("features/chat/editor-mcp-catalog.tsx");
const SETUP = readSrc("features/chat/editor-mcp-setup.tsx");
const COMPOSER = readSrc("features/chat/mcp-composer-button.tsx");

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

test("rows stay visible and only hint when the backend cannot find the editor", () => {
  assert.match(COMPOSER, /item\.installed === false/);
  assert.equal(COMPOSER.match(/>Not installed</g)?.length, 2);
  assert.match(SETUP, /config\?\.installed === false && !config\.is_enabled/);
  assert.match(API, /installed\?: boolean \| null;/);
});
