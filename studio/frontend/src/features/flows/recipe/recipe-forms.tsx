// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Recipe Studio's field forms for one recipe block, inline in the Flows inspector. Lazy: none of Recipe Studio's
// form code loads until a recipe block is inspected.

import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { renderBlockDialog } from "@/features/recipe-studio/blocks/registry";
import { FieldLabel } from "@/features/recipe-studio/dialogs/shared/field-label";
import { ValidationBanner } from "@/features/recipe-studio/dialogs/shared/validation-banner";
import { useRecipeStudioStore } from "@/features/recipe-studio/stores/recipe-studio";
import { RecipeConfigsContext } from "@/features/recipe-studio/stores/recipe-configs-context";
import type {
  NodeConfig,
  RecipeProcessorConfig,
} from "@/features/recipe-studio/types";
import { buildDefaultSchemaTransform } from "@/features/recipe-studio/utils/processors";
import { buildDialogOptions } from "@/features/recipe-studio/utils/recipe-studio-view";
import { useEffect, useMemo, useRef, useState } from "react";
import { useDocStore } from "../stores/doc-store";
import { type RecipeSetupTarget, useUiStore } from "../stores/ui-store";
import type { FlowNode } from "../types";
import { commitRecipe } from "./recipe-actions";
import {
  type StudioView,
  applyConfigPatch,
  commitStudioEdit,
  isColumnConfig,
} from "./recipe-adapter";
import { commitConfigs, setupConfig, viewOf } from "./recipe-blocks";
import { groupColumns } from "./recipe-columns";

/** The seed form queues replaced upload dirs for Recipe Studio to delete; a flow can share those dirs with the
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

function DropRow({
  checked,
  onChange,
}: { checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <div className="corner-squircle flex items-center justify-between gap-3 rounded-xl border border-border/60 px-3 py-2">
      <div className="min-w-0">
        <p className="text-sm font-semibold">Keep out of final dataset</p>
        <p className="text-xs text-muted-foreground">
          Used while generating, left out of the rows.
        </p>
      </div>
      <Switch checked={checked} onCheckedChange={onChange} />
    </div>
  );
}

function BlockForm({
  groupId,
  find,
  onRename,
}: {
  groupId: string;
  find: (view: StudioView) => string | undefined;
  onRename?: (name: string) => void;
}) {
  useNoUploadCleanup();
  const [opened] = useState(() => {
    const view = viewOf(groupId);
    return view ? { view, id: find(view) } : null;
  });
  const [configs, setConfigs] = useState(opened?.view.configs ?? {});
  const latest = useRef(configs);
  const options = useMemo(
    () => buildDialogOptions(Object.values(configs)),
    [configs],
  );
  const config = opened?.id ? configs[opened.id] : undefined;
  if (!opened || !config) {
    return (
      <p className="text-xs text-muted-foreground">
        {opened?.view.errors[0] ?? "This block is no longer in the recipe."}
      </p>
    );
  }
  const update = (id: string, patch: Partial<NodeConfig>) => {
    const next = applyConfigPatch(latest.current, id, patch);
    latest.current = next;
    setConfigs(next);
    commitConfigs(groupId, opened.view, next);
    if (onRename && typeof patch.name === "string") onRename(next[id].name);
  };
  const canDrop =
    isColumnConfig(config) ||
    (config.kind === "seed" && config.seed_source_type === "unstructured");
  return (
    <RecipeConfigsContext.Provider value={configs}>
      <div className="flex min-w-0 flex-col gap-4" data-testid="recipe-block-form">
        <ValidationBanner config={config} />
        {canDrop && (
          <DropRow
            checked={"drop" in config && config.drop === true}
            onChange={(drop) => update(config.id, { drop })}
          />
        )}
        {renderBlockDialog(
          config,
          true,
          options.categoryOptions,
          options.modelConfigAliases,
          options.modelProviderOptions,
          options.localProviderNames,
          options.toolProfileAliases,
          options.datetimeOptions,
          update,
        )}
      </div>
    </RecipeConfigsContext.Provider>
  );
}

/** A column's settings: the column's own Recipe Studio form. */
export function ColumnForm({ node }: { node: FlowNode }) {
  const groupId = node.group ?? "";
  return (
    <BlockForm
      groupId={groupId}
      find={(view) => view.byNode[node.id]}
    />
  );
}

/** A seed, provider connection, model preset or tool profile's settings. */
export function SetupForm({ target }: { target: RecipeSetupTarget }) {
  return (
    <BlockForm
      groupId={target.groupId}
      find={(view) => setupConfig(view, target)?.id}
      onRename={
        target.kind === "seed"
          ? undefined
          : (name) =>
              useUiStore.getState().setRecipeSetup({ ...target, name }, true)
      }
    />
  );
}

/** The recipe's schema transform: reshape every final row with a Jinja JSON template. */
export function ProcessorsForm({ groupId }: { groupId: string }) {
  const [view] = useState(() => viewOf(groupId));
  const [processors, setProcessors] = useState(view?.processors ?? []);
  if (!view) return null;
  const index = processors.findIndex(
    (p) => p.processor_type === "schema_transform",
  );
  const schema = index >= 0 ? processors[index] : null;
  const commit = (next: RecipeProcessorConfig[]) => {
    setProcessors(next);
    const graph = useDocStore.getState().graph;
    const group = graph.groups[groupId];
    if (!group) return;
    const edit = commitStudioEdit(
      group,
      groupColumns(graph, groupId),
      view,
      view.configs,
      next,
    );
    commitRecipe(groupId, edit.nodes, edit.group, `recipe:${groupId}:processors`);
  };
  const patch = (change: Partial<RecipeProcessorConfig>) => {
    if (!schema) return;
    const next = [...processors];
    next[index] = { ...schema, ...change };
    commit(next);
  };
  return (
    <div className="flex flex-col gap-3" data-testid="recipe-processors">
      <div className="corner-squircle flex items-center justify-between gap-3 rounded-xl border border-border/60 px-3 py-2">
        <div className="min-w-0">
          <p className="text-sm font-semibold">Schema transform</p>
          <p className="text-xs text-muted-foreground">
            Reshape every final row with a JSON template.
          </p>
        </div>
        <Switch
          checked={schema !== null}
          onCheckedChange={(on) =>
            commit(
              on
                ? [...processors, buildDefaultSchemaTransform()]
                : processors.filter(
                    (p) => p.processor_type !== "schema_transform",
                  ),
            )
          }
        />
      </div>
      {schema && (
        <>
          <div className="grid gap-1.5">
            <FieldLabel label="Name" />
            <Input
              value={schema.name}
              onChange={(e) => patch({ name: e.target.value })}
            />
          </div>
          <div className="grid gap-1.5">
            <FieldLabel
              label="Template (JSON)"
              hint="Use Jinja references like {{ column_name }} in values."
            />
            <Textarea
              className="min-h-40 text-xs"
              value={schema.template}
              onChange={(e) => patch({ template: e.target.value })}
            />
          </div>
        </>
      )}
    </div>
  );
}
