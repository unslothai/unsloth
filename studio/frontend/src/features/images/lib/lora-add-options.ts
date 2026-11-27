// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionLoraInfo } from "../api.ts";

export interface LoraAddOption {
  lora: DiffusionLoraInfo;
  added: boolean;
}

export interface LoraAddGroup {
  heading: string;
  options: LoraAddOption[];
}

const GROUPS: { heading: string; match: (l: DiffusionLoraInfo) => boolean }[] =
  [
    { heading: "Fine-tuned in Unsloth", match: (l) => Boolean(l.fine_tuned) },
    {
      heading: "On this device",
      match: (l) => !l.fine_tuned && l.source === "local",
    },
    {
      heading: "From Hugging Face",
      match: (l) => !l.fine_tuned && l.source !== "local",
    },
  ];

/** Keeps catalog order within each group. */
export function loraAddGroups(
  catalog: readonly DiffusionLoraInfo[],
  takenIds: ReadonlySet<string>,
  query: string,
): LoraAddGroup[] {
  const needle = query.trim().toLowerCase();
  const shown = needle
    ? catalog.filter(
        (l) =>
          l.display_name.toLowerCase().includes(needle) ||
          l.id.toLowerCase().includes(needle),
      )
    : catalog;
  return GROUPS.map(({ heading, match }) => ({
    heading,
    options: shown
      .filter(match)
      .map((lora) => ({ lora, added: takenIds.has(lora.id) })),
  })).filter((g) => g.options.length > 0);
}

/** A typed `owner/name[:file]` not in the catalog, offered as a Hugging Face id. */
export function customLoraId(
  query: string,
  catalog: readonly DiffusionLoraInfo[],
  takenIds: ReadonlySet<string>,
): { id: string; added: boolean } | null {
  const id = query.trim();
  if (!id.includes("/") || /\s/.test(id)) return null;
  return catalog.some((l) => l.id === id)
    ? null
    : { id, added: takenIds.has(id) };
}
