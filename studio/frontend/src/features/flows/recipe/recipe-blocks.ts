// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A recipe group's blocks as Recipe Studio configs: add one, take one out, commit an edit.

import {
  type BlockKind,
  type BlockType,
  getBlockDefinition,
} from "@/features/recipe-studio/blocks/definitions";
import type { NodeConfig } from "@/features/recipe-studio/types";
import { toast } from "sonner";
import { newId, useDocStore } from "../stores/doc-store";
import { type RecipeSetupTarget, useUiStore } from "../stores/ui-store";
import { commitRecipe, nextColumnSlot } from "./recipe-actions";
import {
  type StudioView,
  columnNode,
  commitStudioEdit,
  isColumnConfig,
  savedView,
  studioView,
} from "./recipe-adapter";
import { groupColumns } from "./recipe-columns";

export function viewOf(groupId: string): StudioView | null {
  const graph = useDocStore.getState().graph;
  const group = graph.groups[groupId];
  return group ? studioView(group, groupColumns(graph, groupId)) : null;
}

/** Saves `configs` as an edit of `before`; returns the view the next edit compares against. */
export function commitConfigs(
  groupId: string,
  before: StudioView,
  configs: Record<string, NodeConfig>,
): StudioView {
  const graph = useDocStore.getState().graph;
  const group = graph.groups[groupId];
  if (!group) return before;
  const edit = commitStudioEdit(
    group,
    groupColumns(graph, groupId),
    before,
    configs,
  );
  commitRecipe(groupId, edit.nodes, edit.group, `recipe:${groupId}`);
  return savedView(before, configs);
}

export function setupConfig(view: StudioView, target: RecipeSetupTarget) {
  return Object.values(view.configs).find((config) =>
    target.kind === "seed"
      ? config.kind === "seed"
      : // Cards go by the trimmed name, so a name mid-typing ("My ") is still its card.
        config.kind === target.kind &&
        config.name.trim() === target.name.trim(),
  );
}

/** Takes a seed, provider, preset or tool profile out of the recipe. */
export function removeSetup(target: RecipeSetupTarget): void {
  const view = viewOf(target.groupId);
  const config = view && setupConfig(view, target);
  if (!view || !config) return;
  const { [config.id]: _removed, ...rest } = view.configs;
  commitConfigs(target.groupId, view, rest);
  useUiStore.getState().setRecipeSetup(null);
}

/** Adds a Recipe Studio block to a group and opens it in the inspector; columns land in the next free slot. */
export function addRecipeBlock(
  groupId: string,
  kind: string,
  type: string,
): void {
  const definition = getBlockDefinition(kind as BlockKind, type as BlockType);
  const view = viewOf(groupId);
  if (!definition || !view) return;
  const id = `flows-${newId("c")}`;
  const config = definition.createConfig(id, Object.values(view.configs));
  // A recipe's seed is saved without a name and opens again as `seed`, a name every block shares.
  if (
    config.kind === "seed" &&
    Object.values(view.configs).some(
      (c) => c.kind !== "seed" && c.name.trim() === "seed",
    )
  ) {
    toast.error("Rename the block called seed first", {
      description:
        "A recipe's seed takes the name seed, and block names are unique.",
    });
    return;
  }
  const configs = { ...view.configs, [id]: config };
  const ui = useUiStore.getState();
  if (isColumnConfig(config)) {
    const graph = useDocStore.getState().graph;
    const slot = nextColumnSlot(graph, groupId);
    const node = columnNode(
      config,
      configs,
      groupId,
      slot.pos,
      `${groupId}_${newId("col")}`,
    );
    if (!node) return;
    commitRecipe(groupId, [node], slot.group, `recipe:${groupId}:add`);
    ui.setSelection([node.id], []);
    ui.openInspector(node.id);
    return;
  }
  commitConfigs(groupId, view, configs);
  ui.setSelection([], []);
  ui.setRecipeSetup({
    groupId,
    kind: config.kind as RecipeSetupTarget["kind"],
    name: config.kind === "seed" ? "seed" : config.name,
  });
}
