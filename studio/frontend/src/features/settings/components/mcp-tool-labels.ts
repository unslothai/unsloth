// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";

// Short labels for the tools served at /mcp/. The raw name shows under each one.
export const MCP_TOOL_LABEL_KEYS: Readonly<Record<string, TranslationKey>> = {
  studio_status: "settings.mcp.toolLabels.studioStatus",
  list_models: "settings.mcp.toolLabels.listModels",
  load_model: "settings.mcp.toolLabels.loadModel",
  unload_model: "settings.mcp.toolLabels.unloadModel",
  plan_load: "settings.mcp.toolLabels.planLoad",
  download_model: "settings.mcp.toolLabels.downloadModel",
  chat: "settings.mcp.toolLabels.chat",
  embed: "settings.mcp.toolLabels.embed",
  system_one: "settings.mcp.toolLabels.systemOne",
  generate_image: "settings.mcp.toolLabels.generateImage",
  generate_audio: "settings.mcp.toolLabels.generateAudio",
  transcribe: "settings.mcp.toolLabels.transcribe",
  analyze_audio: "settings.mcp.toolLabels.analyzeAudio",
  generate_video: "settings.mcp.toolLabels.generateVideo",
  run_recipe: "settings.mcp.toolLabels.runRecipe",
  get_job: "settings.mcp.toolLabels.getJob",
  datasets: "settings.mcp.toolLabels.datasets",
  start_training: "settings.mcp.toolLabels.startTraining",
  list_training_runs: "settings.mcp.toolLabels.listTrainingRuns",
  export_model: "settings.mcp.toolLabels.exportModel",
  list_media: "settings.mcp.toolLabels.listMedia",
  voices: "settings.mcp.toolLabels.voices",
  cancel: "settings.mcp.toolLabels.cancel",
};

/** A tool this build doesn't know yet gets null, and shows its raw name. */
export function mcpToolLabelKey(name: string): TranslationKey | null {
  return Object.hasOwn(MCP_TOOL_LABEL_KEYS, name)
    ? MCP_TOOL_LABEL_KEYS[name]
    : null;
}
