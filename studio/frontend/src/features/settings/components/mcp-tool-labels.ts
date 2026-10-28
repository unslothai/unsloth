// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";

// Short labels for the tools served at /mcp/. The raw name shows on hover.
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

type McpToolGroupId =
  | "models"
  | "generate"
  | "training"
  | "library"
  | "jobs"
  | "other";

// Backend areas (each tool module's short name) folded into a few groups.
const MCP_TOOL_GROUPS: readonly {
  id: McpToolGroupId;
  labelKey: TranslationKey;
  areas: readonly string[];
}[] = [
  {
    id: "models",
    labelKey: "settings.mcp.tools.groups.models",
    areas: ["status", "models"],
  },
  {
    id: "generate",
    labelKey: "settings.mcp.tools.groups.generate",
    areas: ["text", "images", "audio", "video"],
  },
  {
    id: "training",
    labelKey: "settings.mcp.tools.groups.training",
    areas: ["training", "export"],
  },
  {
    id: "library",
    labelKey: "settings.mcp.tools.groups.library",
    areas: ["data", "library"],
  },
  {
    id: "jobs",
    labelKey: "settings.mcp.tools.groups.jobs",
    areas: ["jobs", "cancel"],
  },
];

export type McpToolGroup<T> = {
  id: McpToolGroupId;
  labelKey: TranslationKey;
  tools: T[];
};

/** Tools in group order, keeping server order inside a group. Areas this build doesn't know go last. */
export function groupMcpTools<T extends { area: string }>(
  tools: readonly T[],
): McpToolGroup<T>[] {
  const groups: McpToolGroup<T>[] = MCP_TOOL_GROUPS.map((group) => ({
    id: group.id,
    labelKey: group.labelKey,
    tools: tools.filter((tool) => group.areas.includes(tool.area)),
  }));
  const known = new Set(MCP_TOOL_GROUPS.flatMap((group) => group.areas));
  groups.push({
    id: "other",
    labelKey: "settings.mcp.tools.groups.other",
    tools: tools.filter((tool) => !known.has(tool.area)),
  });
  return groups.filter((group) => group.tools.length > 0);
}
