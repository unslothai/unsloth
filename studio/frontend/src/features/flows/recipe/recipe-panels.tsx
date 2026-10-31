// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Inspector sections for a recipe group: what a column reads and made, and the Recipe Dataset step that stands
// for the whole recipe.

import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { CollapsibleSectionTriggerButton } from "@/features/recipe-studio/dialogs/shared/collapsible-section-trigger";
import { FieldLabel } from "@/features/recipe-studio/dialogs/shared/field-label";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Suspense, lazy, useState } from "react";
import { toast } from "sonner";
import { exportRecipe } from "../api";
import { NodeIconTile } from "../editor/node-card";
import { downloadBlob, runToNode } from "../editor/node-actions";
import { documentFromState, useDocStore } from "../stores/doc-store";
import { IDLE_RUNTIME, useRuntimeStore } from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import type { FlowNode } from "../types";
import {
  checkRecipeGroup,
  openAddBlock,
  runRecipeGroup,
} from "./recipe-group-actions";
import { recipeFileName } from "./recipe-io";
import {
  columnName,
  columnOf,
  columnReferences,
  groupColumns,
  groupGenerate,
} from "./recipe-columns";
import { columnBlock, setupLook } from "./recipe-kinds";
import { recipeSetup } from "./recipe-setup";

const ProcessorsForm = lazy(() =>
  import("./recipe-forms").then((m) => ({ default: m.ProcessorsForm })),
);

function Heading({ children }: { children: string }) {
  return (
    <h3 className="text-xs font-semibold uppercase text-muted-foreground">
      {children}
    </h3>
  );
}

/** Column values from the latest run of its group, so a column shows what it made. */
export function useColumnSamples(node: FlowNode, limit = 3): string[] {
  const generateId = useDocStore((s) =>
    node.group ? groupGenerate(s.graph, node.group)?.id : undefined,
  );
  const runtime =
    useRuntimeStore((s) => (generateId ? s.nodes[generateId] : undefined)) ??
    IDLE_RUNTIME;
  const rows = runtime.outputs?.preview;
  if (!Array.isArray(rows)) return [];
  const name = columnName(node);
  return rows.slice(0, limit).flatMap((row) => {
    const value =
      row && typeof row === "object"
        ? (row as Record<string, unknown>)[name]
        : undefined;
    if (value === undefined || value === null) return [];
    return [typeof value === "string" ? value : JSON.stringify(value)];
  });
}

/** What a column reads and is read by, and what it made in the last run. */
export function ColumnPanel({ node }: { node: FlowNode }) {
  const graph = useDocStore((s) => s.graph);
  const columns = node.group ? groupColumns(graph, node.group) : [];
  const all = new Set(columns.map(columnName));
  const reads = columnReferences(columnOf(node), all);
  const readBy = columns
    .filter((other) =>
      columnReferences(columnOf(other), all).includes(columnName(node)),
    )
    .map(columnName);
  const samples = useColumnSamples(node);
  if (!reads.length && !readBy.length && !samples.length) return null;
  return (
    <div className="flex flex-col gap-2 border-border/60 border-t pt-3">
      {reads.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Reads <span className="text-foreground">{reads.join(", ")}</span>
        </p>
      )}
      {readBy.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Read by <span className="text-foreground">{readBy.join(", ")}</span>
        </p>
      )}
      {samples.length > 0 && (
        <>
          <Heading>Last run</Heading>
          {samples.map((value, i) => (
            <p
              // biome-ignore lint/suspicious/noArrayIndexKey: rows have no id
              key={i}
              className="line-clamp-3 whitespace-pre-wrap rounded-md border border-border/60 bg-muted/40 px-2 py-1 text-xs text-foreground"
            >
              {value}
            </p>
          ))}
        </>
      )}
    </div>
  );
}

async function exportGroup(groupId: string, how: "copy" | "download") {
  try {
    const recipe = await exportRecipe(
      documentFromState(useDocStore.getState()),
      groupId,
    );
    const text = JSON.stringify(recipe, null, 2);
    if (how === "download") {
      downloadBlob(
        new Blob([text], { type: "application/json" }),
        recipeFileName(useDocStore.getState().graph.groups[groupId]?.label),
      );
    } else if (await copyToClipboard(text)) {
      toast.success("Recipe JSON copied", {
        description: "Paste it into Data Recipes' import or another flow.",
      });
    } else {
      throw new Error("Clipboard not available");
    }
  } catch (error) {
    toast.error(
      error instanceof Error ? error.message : "Could not export the recipe",
    );
  }
}

type Check =
  | { state: "idle" | "checking" }
  | { state: "done"; errors: string[] };

function ListRow({
  icon,
  tone,
  title,
  detail,
  onClick,
}: {
  icon: Parameters<typeof NodeIconTile>[0]["icon"];
  tone: string;
  title: string;
  detail: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex w-full items-center gap-2.5 rounded-lg px-2 py-1 text-left transition hover:bg-muted/50"
    >
      <NodeIconTile icon={icon} tone={tone} />
      <span className="min-w-0 flex-1 truncate text-sm">{title}</span>
      <span className="shrink-0 text-ui-11 text-muted-foreground">
        {detail}
      </span>
    </button>
  );
}

/** The Recipe Dataset step's inspector: the whole recipe at a glance, and how to run and share it. */
export function RecipePanel({ node }: { node: FlowNode }) {
  const groupId = node.group ?? "";
  const graph = useDocStore((s) => s.graph);
  const group = graph.groups[groupId];
  const [check, setCheck] = useState<Check>({ state: "idle" });
  const [advancedOpen, setAdvancedOpen] = useState(false);
  if (group?.kind !== "recipe") {
    return (
      <p className="mx-4 my-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
        Put this step inside a Data Recipe group; dropped on an empty spot it
        makes one.
      </p>
    );
  }
  const columns = groupColumns(graph, groupId);
  const setup = recipeSetup(graph, group).cards;
  const seedType = (
    group.config?.seed_config as { source?: { seed_type?: string } } | undefined
  )?.source?.seed_type;
  const ui = useUiStore.getState();
  const runCheck = async () => {
    setCheck({ state: "checking" });
    setCheck({ state: "done", errors: await checkRecipeGroup(groupId) });
  };
  return (
    <div
      className="flex flex-col gap-4 border-border/60 border-t px-4 py-3"
      data-testid="recipe-panel"
    >
      <div className="grid gap-1.5">
        <FieldLabel label="Recipe name" />
        <Input
          value={group.label}
          onChange={(e) =>
            useDocStore
              .getState()
              .upsertGroup({ ...group, label: e.target.value || "Data Recipe" })
          }
        />
      </div>
      <div className="flex flex-col gap-1">
        <Heading>{`Columns (${columns.length})`}</Heading>
        {columns.map((column) => {
          const block = columnBlock(columnOf(column));
          return (
            <ListRow
              key={column.id}
              icon={block.icon}
              tone={block.tone}
              title={columnName(column)}
              detail={block.label}
              onClick={() => {
                ui.setSelection([column.id], []);
                ui.openInspector(column.id);
              }}
            />
          );
        })}
        <Button
          type="button"
          variant="outline"
          size="xs"
          className="mt-1 self-start"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            openAddBlock(groupId, { x: rect.left, y: rect.bottom + 4 });
          }}
          data-testid="recipe-add-column"
        >
          + Add to recipe
        </Button>
      </div>
      {setup.length > 0 && (
        <div className="flex flex-col gap-1">
          <Heading>Seed, models and tools</Heading>
          {setup.map((card) => (
            <ListRow
              key={card.id}
              {...setupLook(card.kind, seedType)}
              title={card.title}
              detail={card.subtitle}
              onClick={() =>
                ui.setRecipeSetup({
                  groupId,
                  kind: card.kind,
                  name: card.name,
                })
              }
            />
          ))}
        </div>
      )}
      <div className="flex flex-col gap-2">
        <Heading>Run</Heading>
        <div className="flex flex-wrap gap-1.5">
          <Button
            type="button"
            variant="dark"
            size="xs"
            onClick={() => runRecipeGroup(groupId, runToNode)}
          >
            Run
          </Button>
          <Button
            type="button"
            variant="outline"
            size="xs"
            onClick={() => ui.setRecipeEditor({ groupId, runSettings: true })}
            data-testid="recipe-run-settings"
          >
            Run settings
          </Button>
          <Button
            type="button"
            variant="outline"
            size="xs"
            disabled={check.state === "checking"}
            onClick={() => void runCheck()}
          >
            {check.state === "checking" ? "Checking…" : "Check"}
          </Button>
          {(["preview", "runs"] as const).map((view) => (
            <Button
              key={view}
              type="button"
              variant="outline"
              size="xs"
              className="capitalize"
              onClick={() => ui.setRecipeRuns({ groupId, view })}
            >
              {view}
            </Button>
          ))}
        </div>
        {check.state === "done" && check.errors.length === 0 && (
          <p className="text-xs text-emerald-700 dark:text-emerald-300">
            Ready to run
          </p>
        )}
        {check.state === "done" && check.errors.length > 0 && (
          <ul className="flex flex-col gap-1 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
            {check.errors.slice(0, 6).map((message) => (
              <li key={message}>{message}</li>
            ))}
          </ul>
        )}
      </div>
      <div className="flex flex-col gap-2">
        <Heading>Share</Heading>
        <div className="flex flex-wrap gap-1.5">
          <Button
            type="button"
            variant="outline"
            size="xs"
            onClick={() => void exportGroup(groupId, "copy")}
            data-testid="recipe-copy-json"
          >
            Copy recipe JSON
          </Button>
          <Button
            type="button"
            variant="outline"
            size="xs"
            onClick={() => void exportGroup(groupId, "download")}
          >
            Download
          </Button>
        </div>
        <p className="text-ui-11 text-muted-foreground">
          Sharing drops keys, tokens and local tool servers.
        </p>
      </div>
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger asChild={true}>
          <CollapsibleSectionTriggerButton
            label="Advanced"
            open={advancedOpen}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3">
          <Suspense fallback={null}>
            <ProcessorsForm groupId={groupId} />
          </Suspense>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
