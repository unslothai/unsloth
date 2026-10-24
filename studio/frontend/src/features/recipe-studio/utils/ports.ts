// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeConfig } from "../types";
import { HANDLE_IDS } from "./handles";

export type RecipePortKind = "data" | "model" | "provider" | "tool";

export type RecipePortSide = "left" | "right";

/**
 * One labelled connection point on a card. Inputs are always on the left and outputs on the
 * right; the card stacks them in list order, one row each. `id` is the React Flow handle id.
 */
export type RecipePort = {
  id: string;
  side: RecipePortSide;
  kind: RecipePortKind;
  label: string;
};

export type RecipePortOptions = {
  /** The LLM's prompt boxes are open on the canvas. */
  promptInput?: boolean;
  /** The recipe has a tool profile, or this LLM already uses one. */
  toolsInput?: boolean;
  /** Something is wired into the card's data input; show it even where it is optional. */
  dataInput?: boolean;
};

export const RECIPE_PORT_KINDS: ReadonlyArray<{ kind: RecipePortKind; label: string }> = [
  { kind: "data", label: "Data" },
  { kind: "model", label: "Model" },
  { kind: "provider", label: "Provider" },
  { kind: "tool", label: "Tool" },
];

const input = (id: string, label: string, kind: RecipePortKind = "data"): RecipePort => ({
  id,
  side: "left",
  kind,
  label,
});

const output = (id: string, label: string, kind: RecipePortKind = "data"): RecipePort => ({
  id,
  side: "right",
  kind,
  label,
});

export const REROUTE_PORTS: RecipePort[] = [
  input(HANDLE_IDS.rerouteIn, ""),
  output(HANDLE_IDS.rerouteOut, ""),
];

export function getAuxNodePorts(label: string): RecipePort[] {
  return [output(HANDLE_IDS.llmInputOut, label)];
}

/** Ports a recipe card shows, top to bottom. */
export function getRecipeNodePorts(
  config: NodeConfig | undefined,
  options: RecipePortOptions = {},
): RecipePort[] {
  if (!config) {
    return [];
  }
  const dataOut = output(HANDLE_IDS.dataOut, config.name);
  switch (config.kind) {
    case "markdown_note":
      return [];
    case "seed":
      return [output(HANDLE_IDS.dataOut, "seed rows")];
    case "sampler": {
      if (config.sampler_type === "subcategory") {
        return [input(HANDLE_IDS.dataIn, "parent"), dataOut];
      }
      if (config.sampler_type === "timedelta") {
        return [input(HANDLE_IDS.dataIn, "from date"), dataOut];
      }
      return options.dataInput ? [input(HANDLE_IDS.dataIn, "refs"), dataOut] : [dataOut];
    }
    case "expression":
      return [input(HANDLE_IDS.dataIn, "refs"), dataOut];
    case "llm":
      return [
        input(HANDLE_IDS.dataIn, "prompt refs"),
        input(HANDLE_IDS.modelIn, "model", "model"),
        ...(options.toolsInput || config.tool_alias?.trim()
          ? [input(HANDLE_IDS.toolsIn, "tools", "tool")]
          : []),
        ...(options.promptInput ? [input(HANDLE_IDS.promptIn, "prompt")] : []),
        dataOut,
      ];
    case "validator":
      return [input(HANDLE_IDS.semanticIn, "checks"), dataOut];
    case "model_provider":
      return [output(HANDLE_IDS.semanticOut, "provider", "provider")];
    case "model_config":
      return [
        input(HANDLE_IDS.semanticIn, "provider", "provider"),
        output(HANDLE_IDS.semanticOut, "model", "model"),
      ];
    case "tool_config":
      return [output(HANDLE_IDS.semanticOut, "tools", "tool")];
    default:
      return [];
  }
}

const INFRA_KINDS = new Set(["model_provider", "model_config", "tool_config"]);

/** The output and input a wire from `source` to `target` attaches to. */
export function resolveRecipeEdgeHandles(
  source: NodeConfig | undefined,
  target: NodeConfig | undefined,
): { sourceHandle: string; targetHandle: string } {
  const sourceHandle =
    source && INFRA_KINDS.has(source.kind) ? HANDLE_IDS.semanticOut : HANDLE_IDS.dataOut;
  let targetHandle: string = HANDLE_IDS.dataIn;
  if (target?.kind === "llm") {
    if (source?.kind === "model_config") {
      targetHandle = HANDLE_IDS.modelIn;
    } else if (source?.kind === "tool_config") {
      targetHandle = HANDLE_IDS.toolsIn;
    }
  } else if (target?.kind === "validator" || target?.kind === "model_config") {
    targetHandle = HANDLE_IDS.semanticIn;
  }
  return { sourceHandle, targetHandle };
}
