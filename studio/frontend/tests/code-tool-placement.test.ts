// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A connection with its own sandbox keeps it; local python/terminal only for those without one,
// since the persisted toggle would otherwise move execution onto the user's machine.

import assert from "node:assert/strict";
import test from "node:test";

import {
  codeToolCanRun,
  selectCodeToolNames,
} from "../src/features/chat/api/code-tool-placement.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  providerHostsCodeExecution,
  providerSupportsBuiltinCodeExecution,
} = await import("../src/features/chat/provider-capabilities.ts");

const SOURCE = readSrc("features/chat/api/chat-adapter.ts");
const COMPOSER_SOURCE = readSrc("features/chat/shared-composer.tsx");
const CHAT_PAGE_SOURCE = readSrc("features/chat/chat-page.tsx");

test("a provider with its own sandbox keeps running the code there", () => {
  assert.deepEqual(
    selectCodeToolNames({
      codeToolsEnabled: true,
      hostedCodeExecutionForThisTurn: true,
      providerHostsCodeExecution: true,
    }),
    { local: [], hosted: ["code_execution"] },
  );
});

test("a provider with a sandbox its MODEL cannot use runs nothing, not local code", () => {
  assert.deepEqual(
    selectCodeToolNames({
      codeToolsEnabled: true,
      hostedCodeExecutionForThisTurn: false,
      providerHostsCodeExecution: true,
    }),
    { local: [], hosted: [] },
  );
});

test("unsupported models on managed custom Responses never fall back to local code", () => {
  const baseUrl = "https://api.openai.com/v1";
  const hosted = providerSupportsBuiltinCodeExecution("custom", "gpt-4.1", baseUrl, "responses");
  const providerHosted = providerHostsCodeExecution("custom", baseUrl, "responses");
  assert.equal(codeToolCanRun({ hostedCodeExecutionForThisTurn: hosted,
    providerHostsCodeExecution: providerHosted, supportsStudioTools: true }), false);
  assert.deepEqual(selectCodeToolNames({ codeToolsEnabled: true,
    hostedCodeExecutionForThisTurn: hosted, providerHostsCodeExecution: providerHosted }),
  { local: [], hosted: [] });
  assert.match(COMPOSER_SOURCE, /providerHostsCodeExecution\(\s*selectedExternalProvider\?\.providerType,\s*selectedExternalProvider\?\.baseUrl,\s*selectedExternalProvider\?\.apiType,/);
  assert.match(CHAT_PAGE_SOURCE, /providerHostsCodeExecution\(\s*provider\?\.providerType,\s*provider\?\.baseUrl,\s*provider\?\.apiType,/);
});

test("a provider with no sandbox uses Unsloth's own tools", () => {
  assert.deepEqual(
    selectCodeToolNames({
      codeToolsEnabled: true,
      hostedCodeExecutionForThisTurn: false,
      providerHostsCodeExecution: false,
    }),
    { local: ["python", "terminal", "edit_file", "view_image"], hosted: [] },
  );
});

test("edit_file is local-only, never a stand-in for a hosted sandbox", () => {
  for (const hosted of [true, false]) {
    const names = selectCodeToolNames({
      codeToolsEnabled: true,
      hostedCodeExecutionForThisTurn: hosted,
      providerHostsCodeExecution: true,
    });
    assert.ok(!names.local.includes("edit_file"));
    assert.ok(!names.hosted.includes("edit_file"));
  }
});

test("the pill being off asks for nothing on either side", () => {
  for (const providerHostsCodeExecution of [true, false]) {
    assert.deepEqual(
      selectCodeToolNames({
        codeToolsEnabled: false,
        hostedCodeExecutionForThisTurn: providerHostsCodeExecution,
        providerHostsCodeExecution,
      }),
      { local: [], hosted: [] },
    );
  }
});

function studioToolsBranch(): string {
  const start = SOURCE.indexOf("...(ragEnabled || projectRagEnabled\n");
  assert.ok(start > 0, "the Unsloth-tools enabled_tools list moved");
  const end = SOURCE.indexOf("mcp_enabled:", start);
  assert.ok(end > start, "the Unsloth-tools branch moved");
  return SOURCE.slice(start, end);
}

test("the Unsloth branch never hardcodes local code tools", () => {
  const branch = studioToolsBranch();

  assert.doesNotMatch(
    branch,
    /codeToolsEnabled \? \["python", "terminal"\]/,
    "the Code pill must not send local execution regardless of provider",
  );
  assert.match(branch, /\.\.\.studioLocalCodeTools/);
  assert.match(branch, /\.\.\.hostedCodeToolsForThisTurn/);
});

test("the branch is only taken when a tool Unsloth itself can run is on", () => {
  // Hosted requests must not send permission_mode; routes/inference.py 400s on it.
  const gate = SOURCE.slice(
    SOURCE.indexOf("...(supportsStudioToolsForThisTurn &&"),
    SOURCE.indexOf("enable_tools: true", SOURCE.indexOf("...(supportsStudioToolsForThisTurn &&")),
  );

  assert.ok(gate.length > 0, "the Unsloth-tools gate moved");
  assert.doesNotMatch(
    gate,
    /^\s*codeToolsEnabled \|\|$/m,
    "a bare codeToolsEnabled sends the Unsloth body for a hosted-only turn",
  );
  assert.match(gate, /studioLocalCodeTools\.length > 0/);
});

test("response details record Code from the placement, local or hosted", () => {
  const start = SOURCE.indexOf("const buildResponseDetails = (");
  const tools = SOURCE.slice(start, SOURCE.indexOf("images:", start));
  assert.match(tools, /code:\s*hostedCodeToolsForThisTurn\.length > 0 \|\|/);
  assert.match(
    tools,
    /\(supportsStudioToolsForThisTurn &&\s*studioLocalCodeTools\.length > 0\)/,
  );
});

test("a model with its provider's sandbox can run code", () => {
  assert.equal(
    codeToolCanRun({
      hostedCodeExecutionForThisTurn: true,
      providerHostsCodeExecution: true,
      supportsStudioTools: true,
    }),
    true,
  );
});

test("a model that cannot use its provider's sandbox offers nothing", () => {
  assert.equal(
    codeToolCanRun({
      hostedCodeExecutionForThisTurn: false,
      providerHostsCodeExecution: true,
      supportsStudioTools: true,
    }),
    false,
  );
});

test("a connection with no sandbox of its own runs Unsloth's tools", () => {
  assert.equal(
    codeToolCanRun({
      hostedCodeExecutionForThisTurn: false,
      providerHostsCodeExecution: false,
      supportsStudioTools: true,
    }),
    true,
  );
});

test("and not when the loop cannot run them either", () => {
  assert.equal(
    codeToolCanRun({
      hostedCodeExecutionForThisTurn: false,
      providerHostsCodeExecution: false,
      supportsStudioTools: false,
    }),
    false,
  );
});

test("the pill is offered exactly when the placement sends something", () => {
  for (const hostedCodeExecutionForThisTurn of [true, false]) {
    for (const providerHostsCodeExecution of [true, false]) {
      const names = selectCodeToolNames({
        codeToolsEnabled: true,
        hostedCodeExecutionForThisTurn,
        providerHostsCodeExecution,
      });
      const sendsSomething = names.hosted.length > 0 || names.local.length > 0;
      assert.equal(
        codeToolCanRun({
          hostedCodeExecutionForThisTurn,
          providerHostsCodeExecution,
          supportsStudioTools: true,
        }),
        sendsSomething,
        `hosted=${hostedCodeExecutionForThisTurn} sandbox=${providerHostsCodeExecution}`,
      );
    }
  }
});
