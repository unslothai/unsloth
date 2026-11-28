// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { chatLocalModelOptions } from "../chat/local-model-options";
import { isDiffusionTask } from "../model-picker/components/model-selector/model-capabilities";
import type { CachedGgufRepo, LocalModelInfo } from "../hub/inventory/api";

function isTaskTextModel(row: {
  audio_type?: string | null;
  task?: string | null;
  pipeline_tag?: string | null;
  capabilities?: { can_chat?: boolean | null } | null;
}): boolean {
  const task = row.task ?? row.pipeline_tag;
  if (isDiffusionTask(task) || row.capabilities?.can_chat === false) return false;
  if (row.audio_type) return row.audio_type === "audio_vlm";
  return !task || ![
    "automatic-speech-recognition", "speech-recognition-unsupported", "text-to-speech",
    "text-to-audio", "audio-to-audio",
  ].includes(task);
}

export function taskLocalModelChoices(
  rows: readonly LocalModelInfo[],
  cachedGguf: readonly CachedGgufRepo[],
) {
  const local = chatLocalModelOptions(rows.filter(isTaskTextModel))
    .map(({ id, name }) => ({ id, name }));
  const available = new Set(local.map(({ id }) => id));
  const incompatible = new Set(
    rows.filter((row) => !available.has(row.id) && !isTaskTextModel(row)).map(({ id }) => id),
  );
  // Keep repository-level GGUF choices for tasks that can switch the resident model.
  for (const repo of cachedGguf) {
    const id = repo.repo_id;
    if (available.has(id) || repo.partial) continue;
    if (!isTaskTextModel(repo)) incompatible.add(id);
    if (!incompatible.has(id)) {
      available.add(id);
      local.push({ id, name: id });
    }
  }
  return { local, incompatible };
}
