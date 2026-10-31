// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import { useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import type { DocGraph } from "../stores/doc-store";
import type { FlowGroup, FlowNode } from "../types";
import { groupColumns, groupGenerate, groupPayload } from "./recipe-columns";

/** Why the recipe cannot run yet when its file seed came without the file; mirrors `recipe_share.missing_seed`. */
export function missingSeed(group: FlowGroup | undefined): string | null {
  const seed = group?.config?.seed_config as
    | { source?: Record<string, unknown> }
    | undefined;
  const source = seed?.source;
  const type = source?.seed_type;
  if (type !== "local" && type !== "unstructured") return null;
  const paths = Array.isArray(source?.paths) ? source.paths : [];
  if (source?.path || paths.some(Boolean)) return null;
  const what = type === "unstructured" ? "document" : "CSV or JSON file";
  return `Load the seed first: this recipe has no ${what} for its seed yet, so open the seed and pick one`;
}

function seedWired(groupId: string): boolean {
  const graph = useDocStore.getState().graph;
  const step = groupGenerate(graph, groupId);
  return (
    !!step &&
    graph.edgeOrder.some((id) => {
      const edge = graph.edges[id];
      return edge.target === step.id && edge.target_port === "seed";
    })
  );
}

/** The columns a run sends: disabled ones are left out, as the compiler does. */
export function runColumns(graph: DocGraph, groupId: string): FlowNode[] {
  return groupColumns(graph, groupId).filter((n) => !n.disabled);
}

/** The recipe's Hub seed source while its path is still a repo id (`owner/name`), not a reader path; mirrors `data.bare_hub_seed`. */
export function bareHubSeed(
  recipe: Record<string, unknown>,
): Record<string, unknown> | null {
  const seed = recipe.seed_config as { source?: Record<string, unknown> } | undefined;
  const source = seed?.source;
  if (!source || source.seed_type !== "hf") return null;
  const path = typeof source.path === "string" ? source.path.trim() : "";
  if (!path || path.startsWith("datasets/") || path.startsWith("hf://") || path.includes("*"))
    return null;
  return source;
}

export type RecipeCheck = {
  valid: boolean;
  errors: string[];
  rawDetail: string | null;
};

/** Recipe Studio's check of the group as it would run: its client checks, then the server's validate, over
 * the columns a run sends and with a bare Hub seed resolved. Every check and run gate in Flows uses this. */
export async function checkRecipeGroupDetail(groupId: string): Promise<RecipeCheck> {
  const graph = useDocStore.getState().graph;
  const group = graph.groups[groupId];
  if (!group) return { valid: false, errors: ["This recipe is gone"], rawDetail: null };
  try {
    const columns = runColumns(graph, groupId);
    const { clientErrors } = await import("./recipe-adapter");
    const local = clientErrors(group, columns);
    if (local.length) return { valid: false, errors: local, rawDetail: null };
    const { inspectSeedDataset, validateRecipe } = await import(
      "@/features/recipe-studio/api"
    );
    // A copy: the resolved path is for this check, not written into the group.
    const recipe = structuredClone(groupPayload(group, columns).recipe);
    const hub = bareHubSeed(recipe);
    if (hub) {
      // As Recipe Studio's seed Load does (and the run): the reader needs the resolved path.
      const inspected = await inspectSeedDataset(hubSeedInspect(hub));
      hub.path = inspected.resolved_path;
    }
    const result = await validateRecipe({ recipe });
    const errors = (result.errors ?? []).map((e) => e.message);
    return {
      valid: result.valid !== false && errors.length === 0,
      errors,
      rawDetail: result.raw_detail ?? null,
    };
  } catch (error) {
    return {
      valid: false,
      errors: [error instanceof Error ? error.message : String(error)],
      rawDetail: null,
    };
  }
}

/** The check's messages; empty when the recipe is ready to run. */
export async function checkRecipeGroup(groupId: string): Promise<string[]> {
  return (await checkRecipeGroupDetail(groupId)).errors;
}

/** The `/seed/inspect` request for a bare Hub seed, with what the run sends too (split, subset, token). */
export function hubSeedInspect(hub: Record<string, unknown>) {
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    dataset_name: String(hub.path).trim(),
    split: typeof hub.split === "string" ? hub.split : "train",
    subset: typeof hub.subset === "string" ? hub.subset : undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    hf_token: typeof hub.token === "string" && hub.token ? hub.token : undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    preview_size: 1,
  };
}

/** Runs the group's Recipe Dataset step with `run` (`runToNode`), unless its seed file still has to be loaded. */
export function runRecipeGroup(
  groupId: string,
  run: (nodeId: string) => unknown,
): void {
  const graph = useDocStore.getState().graph;
  const step = groupGenerate(graph, groupId);
  if (!step) {
    toast.error("This recipe has no Recipe Dataset step");
    return;
  }
  const blocked = seedWired(groupId)
    ? null
    : missingSeed(graph.groups[groupId]);
  if (blocked) {
    const [title, detail] = blocked.split(": ");
    toast.error(title, { description: detail });
    useUiStore
      .getState()
      .setRecipeSetup({ groupId, kind: "seed", name: "seed" });
    return;
  }
  void run(step.id);
}

/** Flows' quick add, listing the blocks a recipe can take, at `screen`. */
export function openAddBlock(
  groupId: string,
  screen: { x: number; y: number },
): void {
  const group = useDocStore.getState().graph.groups[groupId];
  if (!group) return;
  useUiStore.getState().setQuickAdd({
    screen,
    flow: { x: group.rect[0] + 40, y: group.rect[1] + 60 },
    recipe: groupId,
  });
}
