// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";

const source = readSrc("features/audio/audio-page.tsx");

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
