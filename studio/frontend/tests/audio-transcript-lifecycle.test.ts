// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc, readText } from "./helpers/kit.ts";
import { AUTH_SESSION_ENDING_EVENT } from "../src/features/auth/session-events.ts";
import { readTranscriptDraft, writeTranscriptDraft } from "../src/features/audio/transcript-draft.ts";

const source = readSrc("features/audio/audio-page.tsx");

test("macOS termination checks unsaved transcripts before allowing exit", () => {
  const native = readText("../../src-tauri/src/main.rs");
  const predicate = native.slice(
    native.indexOf("fn quit_requires_confirmation"),
    native.indexOf("fn cleanup_child_processes"),
  );
  assert.match(predicate, /\|\| renderer.unsaved_transcript/);
  assert.match(
    native,
    /begin_or_attach_termination\(quit_requires_confirmation\(app\)\)/,
  );
  assert.match(native, /&& confirm_quit_with_unsaved_transcript\(&app\)/);
});

function section(start: string, end: string): string {
  return source.slice(
    source.indexOf(start),
    source.indexOf(end, source.indexOf(start)),
  );
}

function callback(name: string, scope: Record<string, unknown>) {
  const tree = ts.createSourceFile(
    "audio-page.tsx",
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let expression: ts.Expression | undefined;
  function visit(node: ts.Node) {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(tree) === name &&
      node.initializer &&
      ts.isCallExpression(node.initializer)
    ) {
      expression = node.initializer.arguments[0];
    }
    ts.forEachChild(node, visit);
  }
  visit(tree);
  assert.ok(expression, `missing callback ${name}`);
  const { outputText } = ts.transpileModule(
    `return (${expression.getText(tree)});`,
    {
      compilerOptions: { target: ts.ScriptTarget.ES2022 },
    },
  );
  return new Function(...Object.keys(scope), outputText)(
    ...Object.values(scope),
  );
}

test("declining replacement prevents recording setup", async () => {
  const events: string[] = [];
  const start = callback("handleRecordToggle", {
    isRecording: false,
    micPendingGeneration: { current: null },
    busyRef: { current: null },
    micRequestGeneration: { current: 0 },
    transcriptVersion: { current: 7 },
    setMicRequestPending: () => {},
    confirmTranscriptReplacement: () => {
      events.push("confirm");
      return false;
    },
    prepareTranscriptionModel: async () => {
      events.push("prepare");
      return null;
    },
  });
  await start();
  assert.deepEqual(events, ["confirm"]);
});

test("recording approval applies once and only to the transcript it covered", async () => {
  for (const approvedVersion of [7, 6, undefined]) {
    const events: string[] = [];
    const run = callback("runTranscription", {
      transcriptionAbort: { current: null },
      busyRef: { current: null },
      transcriptVersion: { current: 7 },
      activeRef: { current: false },
      confirmTranscriptReplacement: () => {
        events.push("confirm");
        return false;
      },
      prepareTranscriptionModel: async () => {
        events.push("prepare");
        return null;
      },
      setBusy: () => {},
      AbortController,
    });
    await run(new Blob(["recording"]), "Recording", approvedVersion);
    assert.deepEqual(events, approvedVersion === 7 ? ["prepare"] : ["confirm"]);
  }
  const recording = section(
    "const handleRecordToggle",
    "// Release the microphone",
  );
  assert.match(recording, /const confirmedVersion = transcriptVersion.current/);
  assert.match(
    recording,
    /runTranscription\(blob, "Recording", confirmedVersion\)/,
  );
});

test("changing model residency preserves the transcript and its recorded origin", () => {
  const refresh = section("const refreshSttStatus", "const sttSelected");
  const release = section(
    "const releaseTranscribeSelection",
    "const ensureClipSrc",
  );
  assert.doesNotMatch(refresh, /clearTranscript\(/);
  assert.doesNotMatch(release, /clearTranscript\(/);
  assert.match(source, /setTranscriptModel\(result.model\)/);
  assert.match(source, /setTranscriptModel\(record.model\)/);
});

test("the previous result is replaced only after its replacement model is ready", () => {
  const run = section("const runTranscription", "const handleRecordToggle");
  assert.match(run, /confirmTranscriptReplacement\(\)/);
  assert.ok(
    run.indexOf("await prepareTranscriptionModel()") <
      run.indexOf("clearTranscript()"),
  );
  assert.ok(
    run.indexOf("clearTranscript()") <
      run.indexOf("await transcribeWithProgress"),
  );
});

test("leaving the page stops microphone capture while transcription can finish into history", () => {
  const lifecycle = section(
    "// Release the microphone",
    "const handleTranscribeFile",
  );
  assert.match(
    lifecycle,
    /if \(!active\) \{\s*stopAndDiscardRecording\(\);\s*\}/,
  );
  assert.match(
    lifecycle,
    /useEffect\(\(\) => \(\) => transcriptionAbort.current\?\.abort\(\), \[\]\)/,
  );
  const run = section("const runTranscription", "const handleRecordToggle");
  assert.doesNotMatch(
    run.slice(run.indexOf("await transcribeWithProgress")),
    /!activeRef.current/,
  );
});

test("logout checks unsaved transcripts before revoking the session or navigating", async () => {
  const eventName = AUTH_SESSION_ENDING_EVENT;
  const sidebar = readSrc("components/app-sidebar.tsx");
  const parse = (text: string) =>
    ts.createSourceFile(
      "component.tsx", text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX,
    );
  const audioTree = parse(source);
  let protection: string | undefined;
  function findProtection(node: ts.Node) {
    if (
      ts.isCallExpression(node) &&
      node.expression.getText(audioTree) === "useEffect" &&
      node.arguments[0]?.getText(audioTree).includes('"set_renderer_activity"')
    ) {
      protection = node.arguments[0].getText(audioTree);
    }
    ts.forEachChild(node, findProtection);
  }
  findProtection(audioTree);
  assert.ok(protection);
  const sidebarTree = parse(sidebar);
  const handlers: string[] = [];
  function findLogout(node: ts.Node) {
    if (
      ts.isArrowFunction(node) &&
      node.modifiers?.some((item) => item.kind === ts.SyntaxKind.AsyncKeyword) &&
      node.body.getText(sidebarTree).includes("await logout()")
    ) {
      handlers.push(node.getText(sidebarTree));
    }
    ts.forEachChild(node, findLogout);
  }
  findLogout(sidebarTree);
  assert.equal(handlers.length, 2);
  const execute = (expression: string, scope: Record<string, unknown>) => {
    const { outputText } = ts.transpileModule(`return (${expression});`, {
      compilerOptions: { target: ts.ScriptTarget.ES2022 },
    });
    return new Function(...Object.keys(scope), outputText)(
      ...Object.values(scope),
    );
  };
  for (const handler of handlers) {
    for (const scenario of ["decline", "accept", "saved", "exported", "unmounted"]) {
      const events: string[] = [];
      const drafts = new Map<string, string>();
      Object.defineProperty(globalThis, "sessionStorage", {
        configurable: true,
        value: {
          getItem: (key: string) => drafts.get(key) ?? null,
          setItem: (key: string, value: string) => drafts.set(key, value),
          removeItem: (key: string) => drafts.delete(key),
        },
      });
      const target = new EventTarget();
      const window = Object.assign(target, {
        confirm: () => {
          events.push("confirm");
          return scenario === "accept";
        },
      });
      const cleanup = execute(protection, {
        transcript: "unsaved text",
        transcriptRecord: scenario === "saved" ? { id: "saved" } : null,
        transcriptExported: scenario === "exported",
        draftKey: "test-draft",
        transcribedName: "speech.wav",
        transcriptModel: "tiny",
        writeTranscriptDraft,
        isTauri: false,
        window,
        AUTH_SESSION_ENDING_EVENT: eventName,
      })();
      if (scenario === "unmounted") cleanup?.();
      await execute(handler, {
        window,
        Event,
        AUTH_SESSION_ENDING_EVENT: eventName,
        logout: async () => {
          events.push("logout");
        },
        clearAuthTokens: () => {
          events.push("clear");
        },
        navigate: () => {
          events.push("navigate");
        },
      })();
      assert.deepEqual(
        events,
        scenario === "decline"
          ? ["confirm"]
          : scenario === "accept"
            ? ["confirm", "logout", "navigate"]
            : ["logout", "navigate"],
      );
      cleanup?.();
      assert.deepEqual(
        readTranscriptDraft("test-draft"),
        scenario === "decline" || scenario === "unmounted"
          ? { text: "unsaved text", title: "speech.wav", model: "tiny" }
          : null,
      );
    }
  }
});

test("remounting audio restores the unsaved transcript and its origin", () => {
  const tree = ts.createSourceFile("audio-page.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const names = new Set(["draftKey", "recoveredTranscript", "transcript", "transcribedName", "transcriptModel"]);
  const declarations: string[] = [];
  function visit(node: ts.Node) {
    if (ts.isVariableDeclaration(node) && ts.isArrayBindingPattern(node.name)) {
      const name = node.name.elements[0]?.getText(tree);
      if (name && names.has(name)) declarations.push(`const ${node.getText(tree)};`);
    }
    ts.forEachChild(node, visit);
  }
  visit(tree);
  const draft = { text: "recovered speech", title: "speech.wav", model: "tiny" };
  const { outputText } = ts.transpileModule(
    declarations.join("\n") + "\nreturn {text: transcript, title: transcribedName, model: transcriptModel};",
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  );
  const scope = {
    useState: (initial: unknown) => [typeof initial === "function" ? initial() : initial, () => {}],
    transcriptDraftKey: () => "test-draft",
    readTranscriptDraft: () => draft,
  };
  const restored = new Function(...Object.keys(scope), outputText)(...Object.values(scope));
  assert.deepEqual(restored, draft);
});
