// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Where the Code pill runs code.
//
// Before Studio's tool loop reached the general external providers, only
// openai_codex carried studio_tools, so `codeToolsEnabled` on an OpenAI,
// Anthropic or Gemini connection fell through to the hosted branch and sent
// `code_execution` -- the model's code ran in the PROVIDER's sandbox. Now that
// those providers take the Studio branch, the same stored pill would send
// ["python", "terminal"] and run the model's code on the USER's machine. The
// toggle is persisted (unsloth_chat_code_tools_enabled), so nobody re-consents:
// the trust boundary moves during an update, with nothing in the composer or
// the stream saying so.
//
// The rule this file pins: a connection that has its own sandbox keeps it.
// Studio's local python/terminal are for connections that have none.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  codeToolCanRun,
  selectCodeToolNames,
} from "../src/features/chat/api/code-tool-placement.ts";

const SOURCE = readFileSync(
  fileURLToPath(new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url)),
  "utf8",
);

// ── the rule itself ────────────────────────────────────────────────

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
  // e.g. an OpenAI connection on a model outside the code-execution family.
  // Pre-loop this sent no code tool at all; falling back to python/terminal
  // would relocate execution rather than preserve it.
  assert.deepEqual(
    selectCodeToolNames({
      codeToolsEnabled: true,
      hostedCodeExecutionForThisTurn: false,
      providerHostsCodeExecution: true,
    }),
    { local: [], hosted: [] },
  );
});

test("a provider with no sandbox uses Studio's own tools", () => {
  // llama.cpp / vLLM / Ollama / custom, and the cloud providers that ship no
  // code sandbox. Local execution is the only meaning the pill can have there,
  // it is what openai_codex has always done, and it paints tool cards under the
  // permission gate rather than happening invisibly.
  assert.deepEqual(
    selectCodeToolNames({
      codeToolsEnabled: true,
      hostedCodeExecutionForThisTurn: false,
      providerHostsCodeExecution: false,
    }),
    { local: ["python", "terminal"], hosted: [] },
  );
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

// ── the adapter has to actually use it ─────────────────────────────

// Same technique as hosted-image-tool-with-studio-tools.test.ts: the body is
// built inside a run closure that needs a live runtime, provider store and
// encryption key, so the structural property is read out of the source.
function studioToolsBranch(): string {
  const start = SOURCE.indexOf("...(ragEnabled || projectRagEnabled\n");
  assert.ok(start > 0, "the Studio-tools enabled_tools list moved");
  const end = SOURCE.indexOf("mcp_enabled:", start);
  assert.ok(end > start, "the Studio-tools branch moved");
  return SOURCE.slice(start, end);
}

test("the Studio branch never hardcodes local code tools", () => {
  const branch = studioToolsBranch();

  assert.doesNotMatch(
    branch,
    /codeToolsEnabled \? \["python", "terminal"\]/,
    "the Code pill must not send local execution regardless of provider",
  );
  // Both sides come from the one helper above, so the local and hosted names
  // cannot drift apart or both be sent for a single pill.
  assert.match(branch, /\.\.\.studioLocalCodeTools/);
  assert.match(branch, /\.\.\.hostedCodeToolsForThisTurn/);
});

test("the branch is only taken when a tool Studio itself can run is on", () => {
  // Code alone on a hosted-sandbox provider is a hosted request: it must reach
  // the hosted branch, which sends no permission_mode. Sending the Studio body
  // for it would ask the backend to confirm tool calls on a passthrough request,
  // which routes/inference.py answers with a 400.
  const gate = SOURCE.slice(
    SOURCE.indexOf("...(supportsStudioToolsForThisTurn &&"),
    SOURCE.indexOf("enable_tools: true", SOURCE.indexOf("...(supportsStudioToolsForThisTurn &&")),
  );

  assert.ok(gate.length > 0, "the Studio-tools gate moved");
  assert.doesNotMatch(
    gate,
    /^\s*codeToolsEnabled \|\|$/m,
    "a bare codeToolsEnabled sends the Studio body for a hosted-only turn",
  );
  assert.match(gate, /studioLocalCodeTools\.length > 0/);
});

// ── Whether the pill is offered at all ─────────────────────────────

// Until Studio's loop reached the general external providers, the composer
// keyed the Code pill on the hosted flag alone, so a model without the hosted
// sandbox simply did not offer it. Keying it on the Studio-tools flag instead
// offered it everywhere, including where the rule above deliberately runs
// nothing, and the user got a lit toggle that sent enable_tools: false.

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

test("a connection with no sandbox of its own runs Studio's tools", () => {
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
