// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Recipe Studio's own block dialogs over a Flows recipe group. Lazy: none of it loads until a column is opened.

import { getBlockDefinition } from "@/features/recipe-studio/blocks/definitions";
import type {
  BlockKind,
  BlockType,
} from "@/features/recipe-studio/blocks/definitions";
import { ConfigDialog } from "@/features/recipe-studio/dialogs/config-dialog";
import { ProcessorsDialog } from "@/features/recipe-studio/dialogs/processors-dialog";
import { RecipeConfigsContext } from "@/features/recipe-studio/stores/recipe-configs-context";
import type {
  NodeConfig,
  RecipeProcessorConfig,
} from "@/features/recipe-studio/types";
import { buildDialogOptions } from "@/features/recipe-studio/utils/recipe-studio-view";
import { useMemo, useRef, useState } from "react";
import { newId, useDocStore } from "../stores/doc-store";
import { type RecipeEditorTarget, useUiStore } from "../stores/ui-store";
import type { FlowNode } from "../types";
import { commitRecipe, nextColumnSlot } from "./recipe-actions";
import {
  applyConfigPatch,
  columnNode,
  commitStudioEdit,
  isColumnConfig,
  studioView,
} from "./recipe-adapter";
import { groupColumns } from "./recipe-columns";

function open(target: RecipeEditorTarget) {
  const graph = useDocStore.getState().graph;
  const group = graph.groups[target.groupId];
  const columns = groupColumns(graph, target.groupId);
  const view = studioView(group, columns);
  let configs = view.configs;
  let configId: string | undefined;
  let added: { nodeId: string; pos: [number, number] } | null = null;
  if (target.nodeId) configId = view.byNode[target.nodeId];
  else if (target.seed)
    configId = Object.values(configs).find((c) => c.kind === "seed")?.id;
  else if (target.configName)
    configId = Object.values(configs).find(
      (c) => c.name === target.configName,
    )?.id;
  else if (target.add) {
    const definition = getBlockDefinition(
      target.add.kind as BlockKind,
      target.add.type as BlockType,
    );
    if (definition) {
      configId = `flows-${newId("c")}`;
      const config = definition.createConfig(configId, Object.values(configs));
      configs = { ...configs, [configId]: config };
      if (isColumnConfig(config)) {
        added = {
          nodeId: `${target.groupId}_${newId("col")}`,
          pos: nextColumnSlot(graph, target.groupId).pos,
        };
      }
    }
  }
  return { view, initial: configs, configId, added };
}

function RecipeEditorOpen({ target }: { target: RecipeEditorTarget }) {
  const [opened] = useState(() => open(target));
  const [configs, setConfigs] = useState(opened.initial);
  const [processors, setProcessors] = useState<RecipeProcessorConfig[]>(
    opened.view.processors,
  );
  const latest = useRef({ configs, processors });
  const options = useMemo(
    () => buildDialogOptions(Object.values(configs)),
    [configs],
  );
  const close = () => useUiStore.getState().setRecipeEditor(null);

  const commit = (
    nextConfigs: Record<string, NodeConfig>,
    nextProcessors: RecipeProcessorConfig[],
  ) => {
    latest.current = { configs: nextConfigs, processors: nextProcessors };
    const graph = useDocStore.getState().graph;
    const group = graph.groups[target.groupId];
    if (!group) return;
    const columns = groupColumns(graph, target.groupId);
    const edit = commitStudioEdit(
      group,
      columns,
      opened.view,
      nextConfigs,
      nextProcessors,
    );
    const nodes: FlowNode[] = [...edit.nodes];
    let nextGroup = edit.group;
    const added = opened.added;
    const config = opened.configId ? nextConfigs[opened.configId] : undefined;
    if (added && config) {
      const existing = graph.nodes[added.nodeId];
      const node = columnNode(
        config,
        nextConfigs,
        target.groupId,
        existing?.pos ?? added.pos,
        added.nodeId,
      );
      if (node) nodes.push(node);
      if (!existing) nextGroup = nextColumnSlot(graph, target.groupId).group;
    }
    commitRecipe(target.groupId, nodes, nextGroup, `recipe:${target.groupId}`);
  };

  if (target.processors) {
    return (
      <ProcessorsDialog
        open={true}
        onOpenChange={(value) => !value && close()}
        processors={processors}
        onProcessorsChange={(next) => {
          setProcessors(next);
          commit(latest.current.configs, next);
        }}
      />
    );
  }
  const config = opened.configId ? configs[opened.configId] : null;
  return (
    <RecipeConfigsContext.Provider value={configs}>
      <ConfigDialog
        open={true}
        onOpenChange={(value) => !value && close()}
        config={config ?? null}
        categoryOptions={options.categoryOptions}
        modelConfigAliases={options.modelConfigAliases}
        modelProviderOptions={options.modelProviderOptions}
        localProviderNames={options.localProviderNames}
        toolProfileAliases={[]}
        datetimeOptions={options.datetimeOptions}
        onUpdate={(id, patch) => {
          const next = applyConfigPatch(latest.current.configs, id, patch);
          setConfigs(next);
          commit(next, latest.current.processors);
        }}
      />
    </RecipeConfigsContext.Provider>
  );
}

export default function RecipeEditor() {
  const target = useUiStore((s) => s.recipeEditor);
  if (!target) return null;
  return <RecipeEditorOpen key={JSON.stringify(target)} target={target} />;
}
