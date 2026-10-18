// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const {
  audioRowDescription,
  catalogById,
  catalogModelOption,
  catalogMoreRows,
  createAudioCatalogStore,
  umbrellaFolderFindings,
  withCatalogPipelineTag,
} = await import("../src/features/audio/audio-catalog-rows.ts");
type Row = Parameters<typeof audioRowDescription>[0];

const REPO = "audio-cpp/audio.cpp-gguf";

function row(over: Partial<Row>): Row {
  const folder = over.folder ?? "New-TTS-GGUF";
  return {
    id: `${REPO}/${folder}`,
    repo_id: REPO,
    folder,
    source: "umbrella",
    display_name: folder,
    family: "new_tts",
    task: "tts",
    workflows: ["speak"],
    description: null,
    languages: null,
    status: "supported",
    size_bytes: 1024,
    downloaded: false,
    unsupported_reason: null,
    ...over,
  };
}

test("More rows skip the seed, UNOFFERED and unsupported rows, and keep Community rows in status order", () => {
  const rows = [
    row({ folder: "Kokoro-82M-GGUF" }),
    row({ folder: "Sidon-GGUF" }),
    row({
      folder: "Broken-GGUF",
      workflows: [],
      unsupported_reason:
        "Unsloth Studio could not tell which model family this is.",
    }),
    row({ folder: "Zeta-GGUF", display_name: "Zeta", status: "wip" }),
    row({ folder: "Beta-GGUF", display_name: "Beta", status: "experimental" }),
    row({ folder: "Alpha-GGUF", display_name: "Alpha" }),
    row({
      id: "Jmica/Drama-3-GGUF",
      repo_id: "Jmica/Drama-3-GGUF",
      folder: "",
      source: "community",
      display_name: "Drama 3",
      status: "community",
      workflows: ["speak", "clone"],
    }),
  ];
  assert.deepEqual(
    catalogMoreRows(rows).map((r) => r.display_name),
    ["Alpha", "Drama 3", "Beta", "Zeta"],
  );
  // Judged against the installed runtime's specs, a supported UNOFFERED folder is listed.
  assert.deepEqual(
    catalogMoreRows(rows, true).map((r) => r.display_name),
    ["Alpha", "Sidon-GGUF", "Drama 3", "Beta", "Zeta"],
  );
});

test("catalogById keys by lower-case id; an option carries name, description and size", () => {
  const byId = catalogById([
    row({ folder: "A-GGUF" }),
    row({ folder: "B-GGUF", workflows: [], unsupported_reason: "no" }),
  ]);
  assert.equal(byId.get(`${REPO}/a-gguf`)?.supported, true);
  assert.equal(byId.get(`${REPO}/b-gguf`)?.supported, false);
  const option = catalogModelOption(
    row({ display_name: "New TTS", size_bytes: 2048 }),
  );
  assert.deepEqual(
    [option.name, option.isGguf, option.sizeBytes, option.description],
    ["New-TTS-GGUF", true, 2048, "New TTS · Text to Speech"],
  );
  const community = catalogModelOption(
    row({
      folder: "",
      id: "Jmica/Drama-3-GGUF",
      display_name: "Jmica/Drama-3-GGUF",
    }),
  );
  assert.deepEqual(
    [community.name, community.description],
    ["Jmica/Drama-3-GGUF", "Text to Speech"],
  );
});

test("audioRowDescription names pages, up to three languages and the status", () => {
  assert.equal(
    audioRowDescription(
      row({
        workflows: ["speak", "clone"],
        languages: ["en", "zh", "ja", "ko", "de"],
        status: "experimental",
      }),
    ),
    "Text to Speech, Clone · English, Chinese, Japanese +2 · Experimental",
  );
  for (const languages of [["auto"], null]) {
    assert.equal(
      audioRowDescription(row({ languages, status: null })),
      "Text to Speech",
    );
  }
  assert.equal(
    audioRowDescription(row({ languages: ["zz-not-a-tag-%"] })),
    "Text to Speech · ZZ-NOT-A-TAG-%",
  );
});

test("an unclassified umbrella folder warns; a stale UNOFFERED entry fails", () => {
  const { warnings, failures } = umbrellaFolderFindings(
    new Set(["Kokoro-82M-GGUF", "Brand-New-GGUF"]),
    new Set(["Kokoro-82M-GGUF"]),
    { "Gone-GGUF": "needs a newer audio runtime" },
  );
  assert.equal(warnings.length, 1);
  assert.match(
    warnings[0],
    /'Brand-New-GGUF\/' is not in the bundled seed; Unsloth Studio lists it/,
  );
  assert.deepEqual(failures, [
    `${REPO}: AUDIO_CPP_UNOFFERED_FOLDERS lists 'Gone-GGUF', which the repo no longer has`,
  ]);
});

test("a failed catalog fetch leaves the static seed, with no rows", async () => {
  let calls = 0;
  const store = createAudioCatalogStore(async () => {
    calls += 1;
    throw new Error("offline");
  });
  await store.refresh();
  assert.deepEqual(store.getSnapshot().rows, []);
  assert.equal(store.getSnapshot().source, "static");
  assert.equal(store.getSnapshot().runtimeSpecs, false);
  await store.refresh();
  assert.equal(calls, 1, "a failure is not refetched on every render");
});

test("a fetched catalog notifies and is cached for the TTL", async () => {
  let clock = 0;
  let calls = 0;
  const store = createAudioCatalogStore(
    async () => {
      calls += 1;
      return {
        source: "hub",
        fetched_at: 1,
        runtime_specs: true,
        models: [row({})],
      };
    },
    { now: () => clock },
  );
  let notified = 0;
  store.subscribe(() => {
    notified += 1;
  });
  await store.refresh();
  assert.equal(store.getSnapshot().source, "hub");
  assert.equal(store.getSnapshot().runtimeSpecs, true);
  assert.equal(store.getSnapshot().byId.size, 1);
  await store.refresh();
  clock = 300_001;
  await store.refresh();
  assert.deepEqual([calls, notified], [2, 2]);
});

test("the picker renders the More group with ListLabel through the row filter", () => {
  const src = readSrc(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  assert.match(
    src,
    /<ListLabel key="more-models" divider>\s*\{moreModels\.label\}/,
  );
  assert.match(src, /\[\.\.\.recommendedRows, \.\.\.moreRows\]\.map/);
  // More rows join the roving option list in drawn order, so each gets its own option id.
  const keys = src.slice(
    src.indexOf("const hubOptionKeys"),
    src.indexOf("const selectedHubOptionKey"),
  );
  assert.match(
    keys,
    /\.\.\.\[\.\.\.recommendedRows, \.\.\.moreRows\]\s*\.filter\(\(r\) => !loadedRowIds\.has/,
  );
  assert.match(
    src,
    /\.filter\(\(model\) => !rowFilter \|\| rowFilter\(\{ id: model\.id \}\)\)/,
  );
  // The format and fit filters the Recommended rows obey apply to More rows too.
  const more = src.slice(
    src.indexOf("const moreRows"),
    src.indexOf("const recommendedMeta"),
  );
  assert.match(
    more,
    /matchesFormatFilter\(model\.id, model\.isGguf \?\? true, formatFilter\)/,
  );
  assert.match(
    more,
    /!fitOnDeviceOnly \|\|\s*cachedIdFor\(row\.id\) !== null \|\|\s*hfModelFitsDevice\(/,
  );
  // An empty, loading or failed Recommended list still shows the More group.
  assert.match(
    src,
    /recommendedRows\.length \+ moreRows\.length === 0 &&\s*recommendedEmpty === "loading"/,
  );
  assert.match(
    src,
    /recommendedRows\.length \+ moreRows\.length === 0 &&\s*recommendedEmpty === "failed"/,
  );
  assert.match(src, /recommendedRows\.length \+ moreRows\.length === 0 \?/);
  const selector = readSrc(
    "features/model-picker/components/model-selector.tsx",
  );
  assert.equal(selector.match(/moreModels=\{moreModels\}/g)?.length, 2);
  assert.match(
    readSrc("features/audio/hooks/use-audio-catalog.ts"),
    /createAudioCatalogStore\(fetchAudioCatalog\)/,
  );
});

test("a catalog ASR pick carries the speech-recognition tag so it loads on Transcribe", () => {
  const asr = row({
    folder: "New-ASR-GGUF",
    task: "asr",
    workflows: ["transcribe"],
  });
  const tts = row({ folder: "New-TTS-GGUF" });
  const byId = catalogById([asr, tts]);
  const meta = { pipelineTag: null as string | null, isGguf: true };
  assert.equal(
    withCatalogPipelineTag(byId, asr.id.toUpperCase(), meta).pipelineTag,
    "automatic-speech-recognition",
  );
  assert.equal(withCatalogPipelineTag(byId, tts.id, meta), meta);
  const tagged = { ...meta, pipelineTag: "text-to-speech" };
  assert.equal(withCatalogPipelineTag(byId, asr.id, tagged), tagged);
  assert.match(
    readSrc("features/audio/audio-page.tsx"),
    /handleModelSelect\(id, withCatalogPipelineTag\(audioCatalog\.byId, id, meta\)\)/,
  );
});
