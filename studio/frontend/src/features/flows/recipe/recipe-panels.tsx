// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Inspector sections for a recipe group: a column, and the Recipe Dataset step that stands for the whole recipe.

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { useState } from "react";
import { toast } from "sonner";
import { exportRecipe } from "../api";
import { downloadBlob } from "../editor/node-actions";
import { documentFromState, useDocStore } from "../stores/doc-store";
import { IDLE_RUNTIME, useRuntimeStore } from "../stores/runtime-store";
import { type RecipeEditorTarget, useUiStore } from "../stores/ui-store";
import type { FlowNode } from "../types";
import { commitRecipe } from "./recipe-actions";
import { recipeFileName } from "./recipe-io";
import {
  type RecipeColumn,
  columnName,
  columnOf,
  columnReferences,
  columnSummary,
  groupColumns,
  groupGenerate,
  groupPayload,
} from "./recipe-columns";

const ADD_COLUMN: { kind: string; type: string; label: string }[] = [
  { kind: "llm", type: "text", label: "AI text" },
  { kind: "llm", type: "structured", label: "AI structured data" },
  { kind: "llm", type: "code", label: "AI code" },
  { kind: "llm", type: "judge", label: "AI scorer" },
  { kind: "sampler", type: "category", label: "Category" },
  { kind: "sampler", type: "subcategory", label: "Subcategory" },
  { kind: "sampler", type: "uniform", label: "Random number" },
  { kind: "sampler", type: "gaussian", label: "Bell-curve number" },
  { kind: "sampler", type: "bernoulli", label: "Yes/no value" },
  { kind: "sampler", type: "datetime", label: "Date and time" },
  { kind: "sampler", type: "timedelta", label: "Time offset" },
  { kind: "sampler", type: "uuid", label: "Unique ID" },
  { kind: "sampler", type: "person", label: "Synthetic person" },
  { kind: "expression", type: "expression", label: "Formula" },
  { kind: "validator", type: "validator_python", label: "Python check" },
  { kind: "validator", type: "validator_sql", label: "SQL check" },
  { kind: "validator", type: "validator_json", label: "JSON check" },
];

const ADD_SEED: { type: string; label: string }[] = [
  { type: "seed_hf", label: "Hugging Face dataset" },
  { type: "seed_local", label: "CSV or JSON file" },
  { type: "seed_unstructured", label: "Document file" },
  { type: "seed_github", label: "GitHub repositories" },
];

const SEED_LABEL: Record<string, string> = {
  hf: "Hugging Face seed",
  local: "CSV or JSON seed",
  unstructured: "Document seed",
  github_repo: "GitHub seed",
};

const BUTTON =
  "rounded-md border border-white/10 px-2 py-1 text-[12px] hover:bg-white/5";
const HEADING =
  "text-[11px] font-semibold uppercase tracking-wider text-zinc-500";

function names(list: unknown, key: string): string[] {
  return Array.isArray(list)
    ? list.flatMap((item) =>
        item &&
        typeof item === "object" &&
        typeof (item as Record<string, unknown>)[key] === "string"
          ? [(item as Record<string, string>)[key]]
          : [],
      )
    : [];
}

function edit(target: RecipeEditorTarget) {
  useUiStore.getState().setRecipeEditor(target);
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

export function ColumnPanel({ node }: { node: FlowNode }) {
  const column = columnOf(node);
  const graph = useDocStore((s) => s.graph);
  const columns = node.group ? groupColumns(graph, node.group) : [];
  const all = new Set(columns.map(columnName));
  const reads = columnReferences(column, all);
  const readBy = columns
    .filter((other) =>
      columnReferences(columnOf(other), all).includes(columnName(node)),
    )
    .map(columnName);
  const { badge, line } = columnSummary(column);
  const samples = useColumnSamples(node);
  const setDrop = (drop: boolean) => {
    const next: RecipeColumn = { ...column, drop };
    commitRecipe(
      node.group as string,
      [{ ...node, params: { ...node.params, column: next } }],
      null,
    );
  };
  return (
    <section className="flex flex-col gap-3 px-4 py-3">
      <div className="flex items-center gap-2">
        <span className="rounded bg-sky-500/15 px-1.5 py-0.5 text-[11px] text-sky-300">
          {badge}
        </span>
        <span className="truncate font-mono text-[12px] text-zinc-300">
          {columnName(node)}
        </span>
      </div>
      {line && (
        <p className="line-clamp-4 whitespace-pre-wrap text-[12px] text-zinc-400">
          {line}
        </p>
      )}
      <button
        type="button"
        onClick={() =>
          node.group && edit({ groupId: node.group, nodeId: node.id })
        }
        className="rounded-md bg-sky-600/80 px-2 py-1.5 text-[12px] hover:bg-sky-500"
        data-testid="recipe-column-edit"
      >
        Edit column
      </button>
      <label className="flex items-center justify-between gap-2 text-[12px] text-zinc-300">
        Keep out of the final dataset
        <input
          type="checkbox"
          checked={column.drop === true}
          onChange={(e) => setDrop(e.target.checked)}
        />
      </label>
      {(reads.length > 0 || readBy.length > 0) && (
        <div className="flex flex-col gap-1 text-[12px] text-zinc-400">
          {reads.length > 0 && <p>Reads {reads.join(", ")}</p>}
          {readBy.length > 0 && <p>Read by {readBy.join(", ")}</p>}
        </div>
      )}
      {samples.length > 0 && (
        <div className="flex flex-col gap-1">
          <h3 className={HEADING}>Last run</h3>
          {samples.map((value, i) => (
            <p
              // biome-ignore lint/suspicious/noArrayIndexKey: rows have no id
              key={i}
              className="line-clamp-3 whitespace-pre-wrap rounded bg-white/5 px-2 py-1 text-[11.5px] text-zinc-300"
            >
              {value}
            </p>
          ))}
        </div>
      )}
    </section>
  );
}

function ConfigList({
  groupId,
  title,
  items,
  addLabel,
  add,
}: {
  groupId: string;
  title: string;
  items: string[];
  addLabel: string;
  add: { kind: string; type: string };
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11.5px] font-medium text-zinc-400">{title}</span>
      {items.map((name) => (
        <button
          key={name}
          type="button"
          onClick={() => edit({ groupId, configName: name })}
          className="truncate rounded-md bg-white/5 px-2 py-1 text-left text-[12px] hover:bg-white/10"
        >
          {name}
        </button>
      ))}
      <button
        type="button"
        onClick={() => edit({ groupId, add })}
        className={`${BUTTON} self-start`}
      >
        + {addLabel}
      </button>
    </div>
  );
}

/** The seed's preview from its dialog: row count and columns, so the panel shows what the run starts from. */
function SeedSummary({ ui }: { ui: Record<string, unknown> }) {
  const rows = Array.isArray(ui.seed_preview_rows) ? ui.seed_preview_rows : [];
  const columns = Array.isArray(ui.seed_columns)
    ? ui.seed_columns.filter((c): c is string => typeof c === "string")
    : [];
  const total =
    typeof ui.seed_total_rows === "number" ? ui.seed_total_rows : null;
  const file =
    typeof ui.local_file_name === "string" ? ui.local_file_name : "";
  const docs = Array.isArray(ui.unstructured_file_names)
    ? ui.unstructured_file_names.length
    : 0;
  if (!rows.length && !columns.length && !file && !docs) return null;
  return (
    <div
      className="flex flex-col gap-0.5 text-[11px] text-zinc-400"
      data-testid="recipe-seed-summary"
    >
      {file && <span className="truncate">{file}</span>}
      {docs > 0 && (
        <span>
          {docs} document{docs === 1 ? "" : "s"}
        </span>
      )}
      {columns.length > 0 && (
        <span className="truncate font-mono">{columns.join(", ")}</span>
      )}
      {(total ?? rows.length) > 0 && (
        <span>
          {total !== null ? `${total} rows` : `${rows.length} preview rows`}
        </span>
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

export function RecipePanel({ node }: { node: FlowNode }) {
  const groupId = node.group ?? "";
  const group = useDocStore((s) => s.graph.groups[groupId]);
  const graph = useDocStore((s) => s.graph);
  const columns = groupColumns(graph, groupId);
  const [check, setCheck] = useState<Check>({ state: "idle" });
  if (group?.kind !== "recipe") {
    return (
      <p className="mx-4 my-3 rounded bg-amber-500/10 px-2 py-1.5 text-[12px] text-amber-300">
        Put this step inside a Data Recipe group; dropped on an empty spot it
        makes one.
      </p>
    );
  }
  const config = group.config ?? {};
  const seed = config.seed_config as Record<string, unknown> | undefined;
  const seedType = (seed?.source as Record<string, unknown> | undefined)
    ?.seed_type;
  const runCheck = async () => {
    setCheck({ state: "checking" });
    try {
      const { validateRecipe } = await import("@/features/recipe-studio/api");
      const result = await validateRecipe({
        recipe: groupPayload(group, columns).recipe,
      });
      setCheck({
        state: "done",
        errors: (result.errors ?? []).map((e) => e.message),
      });
    } catch (error) {
      setCheck({
        state: "done",
        errors: [error instanceof Error ? error.message : String(error)],
      });
    }
  };
  return (
    <section
      className="flex flex-col gap-3 border-white/10 border-t px-4 py-3"
      data-testid="recipe-panel"
    >
      <div className="flex flex-col gap-1">
        <span className="text-[11.5px] font-medium text-zinc-400">
          Recipe name
        </span>
        <input
          value={group.label}
          onChange={(e) =>
            useDocStore
              .getState()
              .upsertGroup({ ...group, label: e.target.value || "Data Recipe" })
          }
          className="rounded-md border border-white/10 bg-zinc-900 px-2 py-1 text-[12.5px] outline-none"
        />
      </div>
      <div className="flex flex-col gap-1">
        <h3 className={HEADING}>Columns</h3>
        {columns.map((column) => (
          <button
            key={column.id}
            type="button"
            onClick={() => {
              useUiStore.getState().setSelection([column.id], []);
              edit({ groupId, nodeId: column.id });
            }}
            className="flex items-center justify-between gap-2 rounded-md bg-white/5 px-2 py-1 text-left text-[12px] hover:bg-white/10"
          >
            <span className="truncate font-mono">{columnName(column)}</span>
            <span className="shrink-0 text-[10.5px] text-zinc-500">
              {columnSummary(columnOf(column)).badge}
            </span>
          </button>
        ))}
        <select
          value=""
          onChange={(e) => {
            const pick = ADD_COLUMN[Number(e.target.value)];
            if (pick)
              edit({ groupId, add: { kind: pick.kind, type: pick.type } });
          }}
          className="mt-1 rounded-md border border-white/10 bg-zinc-900 px-2 py-1 text-[12px]"
          aria-label="Add a column"
        >
          <option value="">+ Add a column…</option>
          {ADD_COLUMN.map((item, i) => (
            <option key={`${item.kind}:${item.type}`} value={i}>
              {item.label}
            </option>
          ))}
        </select>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {(["preview", "runs"] as const).map((view) => (
          <button
            key={view}
            type="button"
            onClick={() =>
              useUiStore.getState().setRecipeRuns({ groupId, view })
            }
            className={`${BUTTON} capitalize`}
          >
            {view}
          </button>
        ))}
        <button
          type="button"
          onClick={() => void runCheck()}
          className={BUTTON}
          disabled={check.state === "checking"}
        >
          {check.state === "checking" ? "Checking…" : "Check recipe"}
        </button>
        <button
          type="button"
          onClick={() => edit({ groupId, runSettings: true })}
          className={BUTTON}
          data-testid="recipe-run-settings"
        >
          Run settings
        </button>
        {check.state === "done" && check.errors.length === 0 && (
          <span className="text-[12px] text-emerald-300">Ready to run</span>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <button
          type="button"
          onClick={() => void exportGroup(groupId, "copy")}
          className={BUTTON}
          data-testid="recipe-copy-json"
        >
          Copy recipe JSON
        </button>
        <button
          type="button"
          onClick={() => void exportGroup(groupId, "download")}
          className={BUTTON}
        >
          Download
        </button>
      </div>
      {check.state === "done" && check.errors.length > 0 && (
        <ul className="flex flex-col gap-1 rounded bg-red-500/10 px-2 py-1.5 text-[12px] text-red-300">
          {check.errors.slice(0, 6).map((message) => (
            <li key={message}>{message}</li>
          ))}
        </ul>
      )}
      <details>
        <summary className="cursor-pointer select-none text-[11.5px] font-medium text-zinc-400 hover:text-zinc-200">
          Models, seed and processors
        </summary>
        <div className="mt-3 flex flex-col gap-3">
          <ConfigList
            groupId={groupId}
            title="Providers"
            items={names(config.model_providers, "name")}
            addLabel="Provider connection"
            add={{ kind: "llm", type: "model_provider" }}
          />
          <ConfigList
            groupId={groupId}
            title="Models"
            items={names(config.model_configs, "alias")}
            addLabel="Model preset"
            add={{ kind: "llm", type: "model_config" }}
          />
          <ConfigList
            groupId={groupId}
            title="Tool profiles"
            items={names(config.tool_configs, "tool_alias")}
            addLabel="Tool profile"
            add={{ kind: "llm", type: "tool_config" }}
          />
          <div className="flex flex-col gap-1">
            <span className="text-[11.5px] font-medium text-zinc-400">
              Seed data
            </span>
            {seed ? (
              <>
                <button
                  type="button"
                  onClick={() => edit({ groupId, seed: true })}
                  className="rounded-md bg-white/5 px-2 py-1 text-left text-[12px] hover:bg-white/10"
                  data-testid="recipe-seed-edit"
                >
                  {SEED_LABEL[String(seedType)] ?? "Seed data"}
                </button>
                <SeedSummary ui={group.ui ?? {}} />
                {seedType === "github_repo" && (
                  <button
                    type="button"
                    onClick={() => edit({ groupId, easy: true })}
                    className={`${BUTTON} self-start`}
                    data-testid="recipe-github-easy"
                  >
                    GitHub crawler (easy view)
                  </button>
                )}
              </>
            ) : (
              <select
                value=""
                onChange={(e) =>
                  e.target.value &&
                  edit({ groupId, add: { kind: "seed", type: e.target.value } })
                }
                className="rounded-md border border-white/10 bg-zinc-900 px-2 py-1 text-[12px]"
                aria-label="Add seed data"
              >
                <option value="">+ Start from data…</option>
                {ADD_SEED.map((item) => (
                  <option key={item.type} value={item.type}>
                    {item.label}
                  </option>
                ))}
              </select>
            )}
            <span className="text-[11px] text-zinc-500">
              A dataset wired into Seed dataset replaces it.
            </span>
          </div>
          <button
            type="button"
            onClick={() => edit({ groupId, processors: true })}
            className={`${BUTTON} self-start`}
          >
            Processors
          </button>
          <p className="text-[11px] leading-snug text-zinc-500">
            Local (stdio) tool servers only run from the Data Recipes page.
            Exporting or sharing drops keys and local tool servers.
          </p>
        </div>
      </details>
    </section>
  );
}
