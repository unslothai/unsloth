// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Recipe Studio's own block dialogs over a Flows recipe group. Lazy: none of it loads until a column is opened.

import { getBlockDefinition } from "@/features/recipe-studio/blocks/definitions";
import type {
  BlockKind,
  BlockType,
} from "@/features/recipe-studio/blocks/definitions";
import { validateRecipe } from "@/features/recipe-studio/api";
import { ConfigDialog } from "@/features/recipe-studio/dialogs/config-dialog";
import { RunDialog } from "@/features/recipe-studio/dialogs/preview-dialog";
import { GithubCrawlerEasyView } from "@/features/recipe-studio/easy/github-crawler-easy-view";
import { ProcessorsDialog } from "@/features/recipe-studio/dialogs/processors-dialog";
import { RecipeConfigsContext } from "@/features/recipe-studio/stores/recipe-configs-context";
import type {
  NodeConfig,
  RecipeProcessorConfig,
} from "@/features/recipe-studio/types";
import { buildDialogOptions } from "@/features/recipe-studio/utils/recipe-studio-view";
import { Dialog, DialogTitle } from "@/components/ui/dialog";
import { RecipeDialogContent } from "@/features/recipe-studio/dialogs/shared/recipe-dialog-content";
import { useRecipeStudioStore } from "@/features/recipe-studio/stores/recipe-studio";
import { useEffect, useMemo, useRef, useState } from "react";
import { runToNode } from "../editor/node-actions";
import { newId, useDocStore } from "../stores/doc-store";
import { useRuntimeStore } from "../stores/runtime-store";
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
import { groupColumns, groupGenerate, groupPayload } from "./recipe-columns";
import { runWithSettings, settingsFromRun } from "./recipe-run-settings";

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

type Check = { valid: boolean; errors: string[]; rawDetail: string | null };

async function checkGroup(groupId: string): Promise<Check> {
  const graph = useDocStore.getState().graph;
  const group = graph.groups[groupId];
  try {
    const result = await validateRecipe({
      recipe: groupPayload(group, groupColumns(graph, groupId)).recipe,
    });
    return {
      valid: result.valid,
      errors: (result.errors ?? []).map((e) => e.message),
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

function setGenerate(groupId: string, patch: Record<string, unknown>) {
  const generate = groupGenerate(useDocStore.getState().graph, groupId);
  if (generate)
    useDocStore
      .getState()
      .updateParams(generate.id, patch, `param:${generate.id}:run`);
}

/** Recipe Studio's run dialog: rows and mode live on the Recipe Dataset step, the rest on the group's `run`. */
function RecipeRunSettings({ target }: { target: RecipeEditorTarget }) {
  const group = useDocStore((s) => s.graph.groups[target.groupId]);
  const generate = useDocStore((s) => groupGenerate(s.graph, target.groupId));
  const running = useRuntimeStore((s) => s.runStatus === "running");
  const [settings, setSettings] = useState(() => settingsFromRun(group?.run));
  const [runName, setRunName] = useState(() =>
    typeof group?.run?.run_name === "string" ? group.run.run_name : "",
  );
  const [check, setCheck] = useState<Check | null>(null);
  const [checking, setChecking] = useState(false);
  const close = () => useUiStore.getState().setRecipeEditor(null);
  if (!group || !generate) return null;
  const save = (next: typeof settings, name: string) => {
    setSettings(next);
    setRunName(name);
    const current = useDocStore.getState().graph.groups[target.groupId];
    if (current)
      useDocStore
        .getState()
        .upsertGroup({ ...current, run: runWithSettings(current.run, next, name) });
  };
  const rows = Number(generate.params.rows ?? 5) || 5;
  return (
    <RunDialog
      open={true}
      onOpenChange={(value) => !value && close()}
      kind={generate.params.mode === "full" ? "full" : "preview"}
      onKindChange={(kind) => setGenerate(target.groupId, { mode: kind })}
      rows={rows}
      onRowsChange={(next) => setGenerate(target.groupId, { rows: next })}
      fullRunName={runName}
      onFullRunNameChange={(name) => save(settings, name)}
      settings={settings}
      onSettingsChange={(patch) => save({ ...settings, ...patch }, runName)}
      loading={running}
      validateLoading={checking}
      validateResult={check}
      errors={[]}
      onValidate={() => {
        setChecking(true);
        void checkGroup(target.groupId)
          .then(setCheck)
          .finally(() => setChecking(false));
      }}
      onRun={() => {
        close();
        void runToNode(generate.id);
      }}
    />
  );
}

/** Recipe Studio's GitHub crawler form over a GitHub-seeded recipe group. */
function RecipeEasyView({ target }: { target: RecipeEditorTarget }) {
  useNoUploadCleanup();
  const [opened] = useState(() => open(target));
  const [configs, setConfigs] = useState(opened.initial);
  const latest = useRef(configs);
  const generate = useDocStore((s) => groupGenerate(s.graph, target.groupId));
  const running = useRuntimeStore((s) => s.runStatus === "running");
  const [errors, setErrors] = useState<string[]>([]);
  const close = () => useUiStore.getState().setRecipeEditor(null);
  if (!generate) return null;
  return (
    <Dialog open={true} onOpenChange={(value) => !value && close()}>
      <RecipeDialogContent className="max-w-2xl">
        <DialogTitle className="sr-only">GitHub crawler</DialogTitle>
        <GithubCrawlerEasyView
          configs={configs}
          rows={Number(generate.params.rows ?? 5) || 5}
          setRows={(rows) => setGenerate(target.groupId, { rows })}
          updateConfig={(id, patch) => {
            const next = applyConfigPatch(latest.current, id, patch);
            latest.current = next;
            setConfigs(next);
            const graph = useDocStore.getState().graph;
            const group = graph.groups[target.groupId];
            if (!group) return;
            const edit = commitStudioEdit(
              group,
              groupColumns(graph, target.groupId),
              opened.view,
              next,
            );
            commitRecipe(
              target.groupId,
              edit.nodes,
              edit.group,
              `recipe:${target.groupId}`,
            );
          }}
          onRun={() => {
            void checkGroup(target.groupId).then((result) => {
              setErrors(result.errors);
              if (!result.valid) return;
              setGenerate(target.groupId, { mode: "full" });
              close();
              void runToNode(generate.id);
            });
          }}
          runLoading={running}
          runErrors={errors}
          onSwitchToAdvanced={close}
        />
      </RecipeDialogContent>
    </Dialog>
  );
}

/** The seed dialog queues replaced upload dirs for Recipe Studio to delete; a flow can share those dirs with the
 * recipe it came from, so nothing it replaces here is queued. */
function useNoUploadCleanup() {
  useEffect(() => {
    const before = useRecipeStudioStore.getState().pendingUploadCleanups;
    return useRecipeStudioStore.subscribe((state) => {
      if (state.pendingUploadCleanups !== before)
        useRecipeStudioStore.setState({ pendingUploadCleanups: before });
    });
  }, []);
}

function RecipeEditorOpen({ target }: { target: RecipeEditorTarget }) {
  useNoUploadCleanup();
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
        toolProfileAliases={options.toolProfileAliases}
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
  if (target.runSettings)
    return <RecipeRunSettings key={JSON.stringify(target)} target={target} />;
  if (target.easy)
    return <RecipeEasyView key={JSON.stringify(target)} target={target} />;
  return <RecipeEditorOpen key={JSON.stringify(target)} target={target} />;
}
