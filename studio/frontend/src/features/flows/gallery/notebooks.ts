// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NotebookEntry } from "../api";

/** Notebooks that ship as a ready flow (backend `build_examples.py` notebook templates). */
export const NOTEBOOK_TEMPLATES: Record<string, string> = {
  "Llama3.1_(8B)-Alpaca.ipynb": "notebook-text-sft",
  "Qwen2.5_VL_(7B)-Vision.ipynb": "notebook-vision-sft",
  "Orpheus_(3B)-TTS.ipynb": "notebook-tts-sft",
  "EmbeddingGemma_(300M).ipynb": "notebook-embedding",
};

export interface NotebookActions {
  /** Example id to copy, when the notebook ships as a flow. */
  template: string | null;
  /** Studio can train this kind of notebook: download it and import the .ipynb. */
  importable: boolean;
  colab: string;
  download: string;
}

export function notebookActions(entry: NotebookEntry): NotebookActions {
  const runnable = Boolean(entry.studio_task);
  return {
    template: runnable
      ? (NOTEBOOK_TEMPLATES[entry.notebook_file] ?? null)
      : null,
    importable: runnable,
    colab: entry.colab_url,
    download: `https://raw.githubusercontent.com/unslothai/notebooks/main/nb/${encodeURIComponent(entry.notebook_file)}`,
  };
}

/** Templates first, then what Studio can train, then Colab-only; search matches title, file and category. */
export function visibleNotebooks(
  entries: NotebookEntry[],
  query: string,
): NotebookEntry[] {
  const needle = query.toLowerCase().replace(/[^a-z0-9]+/g, "");
  const rank = (e: NotebookEntry) => {
    const a = notebookActions(e);
    return a.template ? 0 : a.importable ? 1 : 2;
  };
  return entries
    .filter(
      (e) =>
        !needle ||
        `${e.title}${e.notebook_file}${e.category}`
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "")
          .includes(needle),
    )
    .sort((a, b) => rank(a) - rank(b));
}

/** The filename to import as an Unsloth notebook (a picked or dropped `.ipynb` file), else null. */
export function notebookName(
  data: Blob | string,
  name?: string,
): string | null {
  const picked =
    name ??
    (typeof data === "object" && typeof (data as File).name === "string"
      ? (data as File).name
      : undefined);
  return picked?.toLowerCase().endsWith(".ipynb") ? picked : null;
}
