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
  for (const id of ["unity", "unreal", "godot"]) {
    assert.match(CATALOG, new RegExp(`\\n  ${id}: \\{`));
  }
  assert.match(API, /export type EditorMcpId = "unity" \| "unreal" \| "godot";/);
  const blender = COMPOSER.indexOf('<span className="truncate">Blender</span>');
  const editors = COMPOSER.indexOf("(Object.keys(EDITOR_MCP) as EditorMcpId[]).map");
  assert.ok(blender > 0 && editors > blender);
  assert.match(COMPOSER, /<BlenderMcpSetup/);
  assert.match(COMPOSER, /<EditorMcpSetup\s+key=\{editorOpen\}\s+editorId=\{editorOpen\}/);
});

test("editor setup posts to its own managed routes and only Unreal asks for a project", () => {
  assert.match(API, /mcpRequest\(`\/builtins\/\$\{id\}`, \{ method: "PUT", body: payload \}\)/);
  assert.match(API, /mcpRequest\(`\/builtins\/\$\{id\}\/test`, \{ method: "POST", body: payload \}\)/);
  assert.equal(CATALOG.match(/needsProject: true/g)?.length, 1);
  assert.match(CATALOG, /unreal: \{[\s\S]*?needsProject: true/);
  assert.match(SETUP, /const settings = \{ port: Number\(port\), project_path: projectPath\.trim\(\) \};/);
  assert.match(SETUP, /disabled=\{locked \|\| !config\.available \|\| !ready \|\| !consent\}/);
});

test("rows stay visible and only hint when the backend cannot find the editor", () => {
  assert.match(COMPOSER, /item\.installed === false/);
  assert.equal(COMPOSER.match(/>Not installed</g)?.length, 2);
  assert.match(SETUP, /config\?\.installed === false && !config\.is_enabled/);
  assert.match(API, /installed\?: boolean \| null;/);
});
