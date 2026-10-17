// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { allowedPages, pageModels, recommendedPageModels, rowOnPage } =
  await import("../src/features/audio/page-models.ts");
const { audioModelsForTask } = await import("../src/features/audio/catalog.ts");
const { AUDIO_CPP_MODELS, audioCppWorkflowsFor } = await import(
  "../src/features/audio/audio-cpp-catalog.ts"
);

type Prefs = Parameters<typeof pageModels>[2];
const SEED = audioModelsForTask("tts");
const NEW_ID = "audio-cpp/audio.cpp-gguf/New-TTS-GGUF";
const catalog = (
  entries: Record<string, { workflows: string[]; supported?: boolean }>,
) =>
  new Map(
    Object.entries(entries).map(([id, e]) => [
      id.toLowerCase(),
      {
        workflows: e.workflows,
        supported: e.supported ?? e.workflows.length > 0,
      },
    ]),
  );
const ids = (
  page: Parameters<typeof pageModels>[0],
  prefs: Prefs,
  rows = SEED,
) => pageModels(page, rows, prefs).map((row) => row.id);
const folder = (id: string) => id.split("/").pop();

test("each page lists the seed models whose catalog entry runs there, in the page's order", () => {
  for (const page of ["clone", "edit", "convert", "separate"] as const) {
    const expected = SEED.filter((row) => {
      const entry = AUDIO_CPP_MODELS.find(
        (m) => m.id.toLowerCase() === row.id.toLowerCase(),
      );
      return entry && audioCppWorkflowsFor(entry).includes(page);
    }).map((row) => row.id);
    assert.deepEqual(
      [...ids(page, { isMac: false })].sort(),
      [...expected].sort(),
      page,
    );
  }
  assert.deepEqual(ids("clone", { isMac: false }).slice(0, 3).map(folder), [
    "Qwen3-TTS-12Hz-0.6B-Base-GGUF",
    "VoxCPM2-GGUF",
    "Chatterbox-GGUF",
  ]);
  assert.equal(folder(ids("separate", { isMac: false })[0]), "HTDemucs-GGUF");
  assert.equal(folder(ids("edit", { isMac: false })[0]), "DotTTS-Edit-GGUF");
  // Music keeps its seed order; only the empty state names its two picks.
  const music = ids("music", { isMac: false });
  assert.deepEqual(
    music,
    SEED.filter((row) => music.includes(row.id)).map((row) => row.id),
  );
  assert.deepEqual(
    recommendedPageModels("music", SEED, { isMac: false }).map((r) =>
      folder(r.id),
    ),
    ["ACE-Step1.5-GGUF", "Stable-Audio-3-Small-Music-GGUF"],
  );
  assert.equal(
    recommendedPageModels("clone", SEED, { isMac: false }).length,
    2,
  );
});

test("a catalog model outside the seed lands on the pages the backend names", () => {
  const prefs = {
    isMac: false,
    catalog: catalog({ [NEW_ID]: { workflows: ["speak", "clone"] } }),
  };
  const rows = [...SEED, { id: NEW_ID, name: "New TTS" }];
  assert.ok(ids("speak", prefs, rows).includes(NEW_ID));
  assert.ok(ids("clone", prefs, rows).includes(NEW_ID));
  assert.ok(!ids("convert", prefs, rows).includes(NEW_ID));
  assert.ok(rowOnPage("clone", { id: NEW_ID, task: "text-to-speech" }, prefs));
  assert.ok(
    !rowOnPage("convert", { id: NEW_ID, task: "text-to-speech" }, prefs),
  );
  // Ids match whatever their case.
  assert.ok(rowOnPage("clone", { id: NEW_ID.toUpperCase() }, prefs));
});

test("Speak keeps the curated non-GGUF models and Clone stays audio.cpp-only", () => {
  const speak = ids("speak", { isMac: false });
  const clone = ids("clone", { isMac: false });
  const curated = speak.filter(
    (id) => !id.toLowerCase().startsWith("audio-cpp/"),
  );
  assert.ok(curated.length > 0);
  for (const id of curated) assert.ok(!clone.includes(id), id);
  assert.ok(
    speak.some((id) => /orpheus|csm/i.test(id)),
    "a curated speech model is on Speak",
  );
});

test("on a Mac a catalog model is runnable, an unknown non-audio.cpp model is not", () => {
  const prefs = {
    isMac: true,
    catalog: catalog({
      "jmica/drama-3-gguf": { workflows: ["speak", "clone"] },
    }),
  };
  const rows = [
    { id: "Jmica/Drama-3-GGUF", name: "Drama 3" },
    { id: "someone/Unknown-TTS", name: "Unknown" },
  ];
  assert.deepEqual(ids("speak", prefs, rows), ["Jmica/Drama-3-GGUF"]);
  assert.deepEqual(ids("speak", { isMac: false }, rows), [
    "Jmica/Drama-3-GGUF",
    "someone/Unknown-TTS",
  ]);
});

test("backend workflows beat the catalog, which beats the seed, which beats the name hints", () => {
  const kokoro =
    AUDIO_CPP_MODELS.find((m) => m.id.endsWith("/Kokoro-82M-GGUF"))?.id ?? "";
  const prefs = {
    isMac: false,
    catalog: catalog({ [kokoro]: { workflows: ["clone"] } }),
  };
  assert.deepEqual(
    allowedPages({ id: kokoro, audioWorkflows: ["music"] }, prefs),
    ["music"],
  );
  assert.deepEqual(allowedPages({ id: kokoro }, prefs), ["clone"]);
  assert.deepEqual(
    allowedPages({ id: kokoro, task: "text-to-speech" }, { isMac: false }),
    ["speak"],
  );
  const mio = "x/MioTTS-GGUF";
  assert.deepEqual(
    allowedPages({ id: mio, task: "text-to-speech" }, { isMac: false }),
    ["clone"],
  );
  assert.deepEqual(
    allowedPages(
      { id: mio, task: "text-to-speech" },
      {
        isMac: false,
        catalog: catalog({ [mio]: { workflows: ["speak"] } }),
      },
    ),
    ["speak"],
  );
});

test("an unsupported catalog model lands on no page", () => {
  const prefs = {
    isMac: false,
    catalog: catalog({ [NEW_ID]: { workflows: [] } }),
  };
  assert.deepEqual(
    allowedPages({ id: NEW_ID, task: "text-to-speech" }, prefs),
    [],
  );
  for (const page of ["speak", "clone", "music", "transcribe"] as const) {
    assert.deepEqual(ids(page, prefs, [{ id: NEW_ID, name: "New" }]), [], page);
  }
});

test("the Audio page builds its picker lists from pageModels and the catalog hook", () => {
  const page = readSrc("features/audio/audio-page.tsx");
  assert.match(page, /const audioCatalog = useAudioCatalog\(true\);/);
  assert.match(
    page,
    /pageModels\(ttsWorkflow, MODELS_BY_MODE\.speak, pagePrefs\)/,
  );
  assert.match(
    page,
    /rowOnPage\(pageWorkflow, row, pagePrefs\),\s*\[pageWorkflow, pagePrefs\]/,
  );
  assert.match(
    page,
    /pageWorkflow === "transcribe" \|\| pageWorkflow === "edit"\s*\?\s*undefined/,
  );
  assert.match(page, /moreModels=\{selectorMoreModels\}/);
  assert.doesNotMatch(
    page,
    /\b(speak|clone|edit|convert|separate|music)PageModels\b/,
  );
});
