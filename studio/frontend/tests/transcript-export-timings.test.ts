// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { readSrc } from "./helpers/kit.ts";
import {
  type TranscriptDetails,
  detailsFrom,
} from "../src/features/audio/transcript-model.ts";
import { needsSavedTimings } from "../src/features/audio/transcript-export.ts";

const segments = [{ start: 0, end: 1.1, text: "hello world", speaker: "S1" }];
const words = [
  { start: 0, end: 0.5, word: "hello" },
  { start: 0.5, end: 1.1, word: "world" },
];
const speakers = [{ id: "S1", label: "Speaker 1" }];
const record = (timing: Record<string, unknown>) => ({
  id: "t1",
  title: "timed.wav",
  text: "hello world",
  model: "tiny",
  ...timing,
});
const offline = async () => {
  throw new Error("offline");
};

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

function harness(getTranscript: () => Promise<unknown>, version = { current: 1 }) {
  const exported: TranscriptDetails[] = [];
  const failures: string[] = [];
  const names: unknown[] = [];
  const fetched: unknown[] = [];
  const marked: boolean[] = [];
  const shared = {
    getTranscript: (id: string) => {
      fetched.push(id);
      return getTranscript();
    },
    detailsFrom,
    needsSavedTimings,
    downloadTranscript: async (
      _format: string,
      input: { details: TranscriptDetails; names: unknown },
    ) => {
      exported.push(input.details);
      names.push(input.names);
      return true;
    },
  };
  const preview = (listed: ReturnType<typeof record> & { speaker_names?: unknown }) =>
    run(previewDownload, {
      ...shared,
      transcriptVersion: version,
      transcriptRecord: listed,
      transcriptDetails: detailsFrom(listed as Record<string, unknown>),
      transcribedName: listed.title,
      transcript: listed.text,
      transcriptModel: listed.model,
      speakerNames: listed.speaker_names ?? {},
      setTranscriptExported: (value: boolean) => marked.push(value),
      toastFailure: (message: string) => failures.push(message),
    });
  const gallery = run(galleryDownload, {
    ...shared,
    toast: { error: (message: string) => failures.push(message) },
  });
  return { exported, failures, names, fetched, marked, preview, gallery };
}

test("a preview export loads saved timings the selection has not loaded yet", async () => {
  for (const [format, timing] of [
    ["json", { segment_count: 1, has_words: true }],
    ["json", { segment_count: 0, has_words: true }],
    ["txt", { segment_count: 1, speakers, speaker_names: { S1: "Old" } }],
    ["srt", { segment_count: 1 }],
  ] as const) {
    const listed = record({ ...timing, id: `t-${format}` });
    const full = {
      ...listed,
      segments: timing.segment_count ? segments : [],
      words,
      speaker_names: { S1: "Renamed" },
    };
    const h = harness(async () => full);
    await h.preview(listed)(format);
    assert.deepEqual(h.failures, []);
    assert.deepEqual(h.exported, [detailsFrom(full)]);
    assert.deepEqual(h.names, [full.speaker_names]);
    assert.deepEqual(h.fetched, [listed.id]);
  }
});

test("a preview export uses timings the selection already loaded", async () => {
  for (const timing of [
    { segment_count: 1, segments },
    { has_words: true, words },
  ]) {
    const h = harness(offline);
    await h.preview(record(timing))("json");
    assert.deepEqual(h.failures, []);
    assert.deepEqual(h.exported, [detailsFrom(timing)]);
  }
});

test("an export marks the transcript exported only if it is still the one open", async () => {
  for (const switched of [false, true]) {
    const version = { current: 1 };
    const listed = record({ segment_count: 1 });
    const h = harness(async () => {
      if (switched) version.current += 1;
      return { ...listed, segments };
    }, version);
    await h.preview(listed)("json");
    assert.equal(h.exported.length, 1);
    assert.deepEqual(h.marked, switched ? [] : [true]);
  }
});

test("a timed export whose timings fail to load downloads nothing", async () => {
  for (const [format, timing] of [
    ["json", { segment_count: 1 }],
    ["json", { has_words: true }],
    ["txt", { segment_count: 1, speakers }],
    ["vtt", { segment_count: 1 }],
  ] as const) {
    const h = harness(offline);
    await h.preview(record(timing))(format);
    await h.gallery(record(timing), format);
    assert.deepEqual(h.exported, []);
    assert.equal(h.failures.length, 2);
  }
});

test("exports that need no saved timings never wait on them", async () => {
  for (const [format, timing] of [
    ["json", {}],
    ["txt", { segment_count: 1 }],
    ["txt", { has_words: true, speakers }],
  ] as const) {
    const h = harness(offline);
    await h.preview(record(timing))(format);
    await h.gallery(record(timing), format);
    assert.deepEqual(h.failures, []);
    assert.equal(h.exported.length, 2);
  }
});

test("a gallery JSON export of a transcript with only word timings keeps the words", async () => {
  const listed = record({ id: "t2", segment_count: 0, has_words: true });
  const h = harness(async () => ({ ...listed, words }));
  await h.gallery(listed, "json");
  assert.deepEqual(h.fetched, ["t2"]);
  assert.deepEqual(h.exported[0]?.words, words);
});
