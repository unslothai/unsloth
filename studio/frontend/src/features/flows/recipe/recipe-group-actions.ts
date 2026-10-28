// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import { useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import type { FlowGroup } from "../types";
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

/** Recipe Studio's server-side check of the group as it would run. */
export async function checkRecipeGroup(groupId: string): Promise<string[]> {
  const graph = useDocStore.getState().graph;
  const group = graph.groups[groupId];
  if (!group) return ["This recipe is gone"];
  try {
    const { validateRecipe } = await import("@/features/recipe-studio/api");
    const result = await validateRecipe({
      recipe: groupPayload(group, groupColumns(graph, groupId)).recipe,
    });
    return (result.errors ?? []).map((e) => e.message);
  } catch (error) {
    return [error instanceof Error ? error.message : String(error)];
  }
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
  const blocked = seedWired(groupId) ? null : missingSeed(graph.groups[groupId]);
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
