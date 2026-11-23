// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { readSrc } from "./helpers/kit.ts";
import {
  EMPTY_TRANSCRIPT_DETAILS,
  detailsFrom,
} from "../src/features/audio/transcript-model.ts";

const segments = [{ start: 0, end: 1.1, text: "hello world" }];
const words = [
  { start: 0, end: 0.5, word: "hello" },
  { start: 0.5, end: 1.1, word: "world" },
];
const record = (timing: Record<string, unknown>) => ({
  id: "t1",
  title: "timed.wav",
  text: "hello world",
  model: "tiny",
  ...timing,
});

function compile(file: string, find: (node: ts.Node, tree: ts.SourceFile) => ts.Node | undefined) {
  const tree = ts.createSourceFile(file, readSrc(file), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let found: ts.Node | undefined;
  const visit = (node: ts.Node) => {
    found ??= find(node, tree);
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(found, `missing export in ${file}`);
  return ts.transpileModule(`return (${found.getText(tree)});`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText;
}

function run(code: string, scope: Record<string, unknown>) {
  return new Function(...Object.keys(scope), code)(...Object.values(scope));
}

const previewDownload = compile("features/audio/hooks/use-transcription.ts", (node, tree) =>
  ts.isVariableDeclaration(node) &&
  node.name.getText(tree) === "handleDownloadTranscript" &&
  node.initializer &&
  ts.isCallExpression(node.initializer)
    ? node.initializer.arguments[0]
    : undefined,
);

const galleryDownload = compile("features/audio/transcript-gallery.tsx", (node) =>
  ts.isFunctionDeclaration(node) && node.name?.text === "downloadRecord" ? node : undefined,
);

function previewScope(
  listed: ReturnType<typeof record>,
  getTranscript: () => Promise<unknown>,
  exported: unknown[],
  failures: string[],
) {
  return {
    transcriptVersion: { current: 1 },
    transcriptRecord: listed,
    transcriptDetails: EMPTY_TRANSCRIPT_DETAILS,
    transcribedName: listed.title,
    transcript: listed.text,
    transcriptModel: listed.model,
    speakerNames: {},
    setTranscriptExported: () => {},
    getTranscript,
    detailsFrom,
    toastFailure: (message: string) => failures.push(message),
    downloadTranscript: async (_format: string, input: { details: unknown }) => {
      exported.push(input.details);
      return true;
    },
  };
}

test("a preview JSON export carries saved timings while their fetch is pending or after it failed", async () => {
  for (const timing of [
    { segment_count: 1, has_words: true },
    { segment_count: 0, has_words: true },
  ]) {
    const exported: unknown[] = [];
    const failures: string[] = [];
    const listed = record(timing);
    const download = run(
      previewDownload,
      previewScope(
        listed,
        async () => ({ ...listed, segments: timing.segment_count ? segments : [], words }),
        exported,
        failures,
      ),
    );
    await download("json");
    assert.deepEqual(failures, []);
    assert.equal(exported.length, 1);
    const details = exported[0] as typeof EMPTY_TRANSCRIPT_DETAILS;
    assert.deepEqual(details.segments, timing.segment_count ? segments : []);
    assert.deepEqual(details.words, words);
  }

  const exported: unknown[] = [];
  const failures: string[] = [];
  const listed = record({ segment_count: 1 });
  const download = run(
    previewDownload,
    previewScope(
      listed,
      async () => {
        throw new Error("offline");
      },
      exported,
      failures,
    ),
  );
  await download("json");
  assert.deepEqual(exported, []);
  assert.equal(failures.length, 1);
});

test("a gallery JSON export of a transcript with only word timings keeps the words", async () => {
  const exported: { details: typeof EMPTY_TRANSCRIPT_DETAILS }[] = [];
  const fetched: string[] = [];
  const download = run(galleryDownload, {
    getTranscript: async (id: string) => {
      fetched.push(id);
      return { ...record({ segment_count: 0, has_words: true }), words };
    },
    detailsFrom,
    toast: { error: () => {} },
    downloadTranscript: async (_format: string, input: { details: typeof EMPTY_TRANSCRIPT_DETAILS }) => {
      exported.push(input);
      return true;
    },
  });
  await download(record({ segment_count: 0, has_words: true }), "json");
  assert.deepEqual(fetched, ["t1"]);
  assert.deepEqual(exported[0]?.details.words, words);
});
