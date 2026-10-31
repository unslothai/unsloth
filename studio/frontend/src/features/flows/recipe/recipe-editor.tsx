// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Recipe Studio's run dialog and GitHub crawler form over a Flows recipe group. Lazy: loads when one opens.

import { Dialog, DialogTitle } from "@/components/ui/dialog";
import { validateRecipe } from "@/features/recipe-studio/api";
import { RunDialog } from "@/features/recipe-studio/dialogs/preview-dialog";
import { RecipeDialogContent } from "@/features/recipe-studio/dialogs/shared/recipe-dialog-content";
import { GithubCrawlerEasyView } from "@/features/recipe-studio/easy/github-crawler-easy-view";
import { useRef, useState } from "react";
import { runToNode } from "../editor/node-actions";
import { useDocStore } from "../stores/doc-store";
import { useRuntimeStore } from "../stores/runtime-store";
import { type RecipeEditorTarget, useUiStore } from "../stores/ui-store";
import { commitRecipe } from "./recipe-actions";
import { applyConfigPatch, commitStudioEdit, studioView } from "./recipe-adapter";
import { groupColumns, groupGenerate } from "./recipe-columns";
import { type RecipeCheck, checkRecipeGroupDetail } from "./recipe-group-actions";
import { fullRunName, runWithSettings, settingsFromRun } from "./recipe-run-settings";

// The same check as the frame's Check: disabled columns left out, a bare Hub seed resolved.
const checkGroup = checkRecipeGroupDetail;

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
  const [runName, setRunName] = useState(() => fullRunName(group));
  const [check, setCheck] = useState<RecipeCheck | null>(null);
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
  const [view] = useState(() => {
    const graph = useDocStore.getState().graph;
    return studioView(
      graph.groups[target.groupId],
      groupColumns(graph, target.groupId),
    );
  });
  const [configs, setConfigs] = useState(view.configs);
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
              view,
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

export default function RecipeEditor() {
  const target = useUiStore((s) => s.recipeEditor);
  if (!target) return null;
  if (target.runSettings)
    return <RecipeRunSettings key={JSON.stringify(target)} target={target} />;
  if (target.easy)
    return <RecipeEasyView key={JSON.stringify(target)} target={target} />;
  return null;
}
