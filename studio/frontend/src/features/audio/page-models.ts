// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which Audio page lists a model. Free of React so the node test runner can load it directly.

import {
  AUDIO_CPP_SEP_AUDIO_TYPE,
  audioCppModelFor,
  isAudioRuntimeGguf,
  isCloneAndConvertFamilyId,
  isCloneOnlyFamilyId,
  isConvertOnlyFamilyId,
  isSpeakAndCloneFamilyId,
} from "./audio-cpp-catalog";
import {
  isMusicGenerationModel,
  macTtsCatalogChoiceIsRunnable,
} from "./catalog";
import { CONVERT_MODEL_ORDER } from "./convert-policy";
import {
  type AudioWorkflowId,
  audioWorkflowForAudioType,
  audioWorkflowForTask,
} from "./workflows";

export interface PageModelRow {
  id: string;
  task?: string | null;
  audioType?: string | null;
  audioWorkflows?: readonly string[] | null;
}

export interface PageCatalogEntry {
  workflows: readonly string[];
  supported: boolean;
}

export interface PageModelPrefs {
  isMac: boolean;
  /** /api/inference/audio/catalog rows keyed by lower-case id. */
  catalog?: ReadonlyMap<string, PageCatalogEntry>;
}

const PAGES: readonly AudioWorkflowId[] = [
  "speak",
  "clone",
  "edit",
  "convert",
  "music",
  "separate",
  "transcribe",
];

// Only the audio runtime runs these, so their seed lists hold audio.cpp rows alone.
const AUDIO_CPP_ONLY_PAGES: ReadonlySet<AudioWorkflowId> = new Set([
  "clone",
  "edit",
  "convert",
  "separate",
]);

/** Folder names in the order a page lists them; the rest keep their seed order. */
export const PAGE_ORDER: Partial<Record<AudioWorkflowId, readonly string[]>> = {
  clone: ["Qwen3-TTS-12Hz-0.6B-Base-GGUF", "VoxCPM2-GGUF", "Chatterbox-GGUF"],
  convert: CONVERT_MODEL_ORDER,
  edit: ["DotTTS-Edit-GGUF", "Vevo2-GGUF", "FireRedAudio-GGUF"],
  separate: [
    "HTDemucs-GGUF",
    "BS-RoFormer-ep368-GGUF",
    "HTDemucs-6stems-GGUF",
    "Mel-Band-RoFormer-GGUF",
  ],
};

// Music keeps its seed order in the picker; only the empty-state picks are named.
const RECOMMENDED_MUSIC_MODELS = [
  "ACE-Step1.5-GGUF",
  "Stable-Audio-3-Small-Music-GGUF",
];

const known = (workflows: readonly string[]) =>
  PAGES.filter((page) => workflows.includes(page));

function catalogEntry(
  id: string,
  prefs: PageModelPrefs,
): PageCatalogEntry | undefined {
  return prefs.catalog?.get(id.trim().toLowerCase().replace(/\/+$/, ""));
}

// Hub rows carry no workflows before download, so the name hints and a permissive last step
// apply to them; seed rows fall back to Speak, as every non-catalog seed is a speech model.
function pagesFor(
  row: PageModelRow,
  prefs: PageModelPrefs,
  hubRow: boolean,
): readonly AudioWorkflowId[] {
  if (row.audioWorkflows && row.audioWorkflows.length > 0) {
    return known(row.audioWorkflows);
  }
  const entry = catalogEntry(row.id, prefs);
  if (entry) return entry.supported ? known(entry.workflows) : [];
  if (row.audioType === AUDIO_CPP_SEP_AUDIO_TYPE) return ["separate"];
  // MiniMax Music 3 is tagged text-to-speech on the Hub.
  if (isMusicGenerationModel(row.id, row.audioType)) return ["music"];
  const byTask = audioWorkflowForTask(row.task);
  if (byTask === "transcribe" || byTask === "music") return [byTask];
  const catalogModel = audioCppModelFor(row.id);
  if (catalogModel?.workflows) return known(catalogModel.workflows);
  if (hubRow) {
    // Convert first: chatterbox and vevo2 are in the clone-only hint too.
    if (isCloneAndConvertFamilyId(row.id)) return ["clone", "convert"];
    if (isConvertOnlyFamilyId(row.id)) return ["convert"];
    if (isCloneOnlyFamilyId(row.id)) return ["clone"];
    if (isSpeakAndCloneFamilyId(row.id)) return ["speak", "clone"];
  }
  const catalogTask = catalogModel?.task;
  if (catalogTask === "music") return ["music"];
  if (catalogTask === "asr") return ["transcribe"];
  if (byTask === "speak" || catalogTask === "tts" || row.audioType) {
    // Speech models clone only when the backend says so; an older row stays on Speak.
    return [audioWorkflowForAudioType(row.audioType)];
  }
  return hubRow ? PAGES.filter((page) => page !== "separate") : ["speak"];
}

/** The pages a row may list on: backend workflows, then the fetched catalog, then the seed. */
export function allowedPages(
  row: PageModelRow,
  prefs: PageModelPrefs,
): AudioWorkflowId[] {
  return [...pagesFor(row, prefs, true)];
}

/** The picker's row filter for `page`. */
export function rowOnPage(
  page: AudioWorkflowId,
  row: PageModelRow,
  prefs: PageModelPrefs,
): boolean {
  return pagesFor(row, prefs, true).includes(page);
}

function runsOnMac(row: PageModelRow, prefs: PageModelPrefs): boolean {
  return (
    macTtsCatalogChoiceIsRunnable(row.id) ||
    isAudioRuntimeGguf(row.id, row.audioType) ||
    catalogEntry(row.id, prefs) !== undefined
  );
}

function rankIn(order: readonly string[], id: string): number {
  const index = order.findIndex((name) => id.endsWith(`/${name}`));
  return index === -1 ? order.length : index;
}

/** The seed or catalog rows `page` lists, in its order. */
export function pageModels<T extends PageModelRow>(
  page: AudioWorkflowId,
  rows: readonly T[],
  prefs: PageModelPrefs,
): T[] {
  return rows
    .filter(
      (row) =>
        pagesFor(row, prefs, false).includes(page) &&
        (!AUDIO_CPP_ONLY_PAGES.has(page) ||
          isAudioRuntimeGguf(row.id, row.audioType) ||
          catalogEntry(row.id, prefs) !== undefined) &&
        (!prefs.isMac || runsOnMac(row, prefs)),
    )
    .map((row, index) => ({ row, index }))
    .sort(
      (a, b) =>
        rankIn(PAGE_ORDER[page] ?? [], a.row.id) -
          rankIn(PAGE_ORDER[page] ?? [], b.row.id) || a.index - b.index,
    )
    .map(({ row }) => row);
}

/** The two models a page's empty state offers. */
export function recommendedPageModels<T extends PageModelRow>(
  page: AudioWorkflowId,
  rows: readonly T[],
  prefs: PageModelPrefs,
): T[] {
  const listed = pageModels(page, rows, prefs);
  if (page !== "music") return listed.slice(0, 2);
  return listed
    .filter(
      (row) =>
        rankIn(RECOMMENDED_MUSIC_MODELS, row.id) <
        RECOMMENDED_MUSIC_MODELS.length,
    )
    .sort(
      (a, b) =>
        rankIn(RECOMMENDED_MUSIC_MODELS, a.id) -
        rankIn(RECOMMENDED_MUSIC_MODELS, b.id),
    );
}
