// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useNativeFileDrop } from "@/features/native-intents/use-native-file-drop";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import {
  type ImportReport,
  copyExample,
  createFlow,
  deleteFlow,
  importFlow,
  importAllRecipes,
  importNotebook,
  listExamples,
  listFlows,
} from "../api";
import { ModelRequirementList } from "../editor/dialogs";
import {
  BookOpen01Icon,
  Delete02Icon,
  FileImportIcon,
  FlowConnectionIcon,
  GameController01Icon,
  GridViewIcon,
  PlusSignIcon,
  TestTube01Icon,
} from "../editor/icon-data";
import { MediaThumb } from "../editor/media";
import { starterRecipeGraph } from "../recipe/recipe-actions";
import { COLUMN_TYPE, referenceEdges } from "../recipe/recipe-columns";
import { RecipeImportDialog } from "../recipe/recipe-import-dialog";
import type { FlowKind, FlowRecord, FlowSummary } from "../types";
import {
  type GalleryCategory,
  categoriesOf,
  needsLine,
  visibleCategories,
} from "../ux-rules";
import { NotebooksSection } from "./notebooks-section";
import { notebookName } from "./notebooks";
import { templateDocument } from "./templates";

const KIND_STYLE: Record<
  FlowKind,
  { label: string; className: string; icon: typeof FlowConnectionIcon }
> = {
  pipeline: {
    label: "Pipeline",
    className: "bg-sky-500/15 text-sky-600 dark:text-sky-300",
    icon: FlowConnectionIcon,
  },
  story: {
    label: "Story game",
    className: "bg-violet-500/15 text-violet-600 dark:text-violet-300",
    icon: BookOpen01Icon,
  },
  world: {
    label: "World game",
    className: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300",
    icon: GameController01Icon,
  },
};

const COVER_GRADIENT: Record<FlowKind, string> = {
  pipeline: "from-sky-500/30 via-indigo-500/20 to-transparent",
  story: "from-violet-500/35 via-fuchsia-500/20 to-transparent",
  world: "from-emerald-500/35 via-lime-500/15 to-transparent",
};

function relativeTime(ms: number): string {
  if (!ms) return "";
  const diff = Date.now() - ms;
  const minutes = Math.round(diff / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days} d ago`;
  return new Date(ms).toLocaleDateString();
}

function FlowCard({
  flow,
  onOpen,
  action,
  onDelete,
  onApp,
}: {
  flow: FlowSummary;
  onOpen: () => void;
  action?: string;
  onDelete?: () => void;
  onApp?: () => void;
}) {
  const kind = KIND_STYLE[flow.kind] ?? KIND_STYLE.pipeline;
  return (
    <div
      className="group relative flex flex-col overflow-hidden rounded-xl border bg-card transition-shadow hover:shadow-lg"
      data-testid="flows-card"
    >
      <button
        type="button"
        onClick={onOpen}
        className="flex flex-col text-left"
      >
        <div
          className={`relative flex aspect-[16/9] items-center justify-center overflow-hidden bg-gradient-to-br ${COVER_GRADIENT[flow.kind] ?? ""}`}
        >
          {flow.cover ? (
            <MediaThumb
              media={{
                kind: "image",
                ref: flow.cover,
                url: flow.cover.startsWith("/") ? flow.cover : null,
              }}
              className="h-full w-full object-cover"
            />
          ) : (
            <HugeiconsIcon
              icon={kind.icon}
              size={42}
              strokeWidth={1.2}
              className="text-foreground/40"
            />
          )}
        </div>
        <div className="flex flex-col gap-1 p-3">
          <div className="flex items-center gap-2">
            <span className="min-w-0 flex-1 truncate font-medium text-sm">
              {flow.name}
            </span>
            <span
              className={`rounded px-1.5 py-0.5 text-[10.5px] ${kind.className}`}
            >
              {kind.label}
            </span>
          </div>
          {flow.description && (
            <p className="line-clamp-2 text-muted-foreground text-xs">
              {flow.description}
            </p>
          )}
          {flow.example && needsLine(flow.needs) && (
            <p
              className="text-[11px] text-foreground/70"
              data-testid="flows-card-needs"
            >
              {needsLine(flow.needs)}
            </p>
          )}
          <p className="text-[11px] text-muted-foreground">
            {flow.nodeCount} nodes
            {flow.updatedAt && !flow.example
              ? ` · ${relativeTime(flow.updatedAt)}`
              : ""}
          </p>
        </div>
      </button>
      {action && (
        <span className="pointer-events-none absolute top-2 right-2 rounded-md bg-background/90 px-2 py-1 text-xs font-medium opacity-0 shadow transition-opacity group-hover:opacity-100">
          {action}
        </span>
      )}
      {onApp && (
        <button
          type="button"
          aria-label={`Open ${flow.name} as an app`}
          title="Open as app"
          onClick={onApp}
          className="absolute top-2 right-10 rounded-md bg-background/80 p-1.5 text-muted-foreground opacity-0 transition-opacity hover:text-foreground group-hover:opacity-100"
          data-testid="flows-card-app"
        >
          <HugeiconsIcon icon={GridViewIcon} size={14} />
        </button>
      )}
      {onDelete && (
        <button
          type="button"
          aria-label={`Delete ${flow.name}`}
          title="Delete"
          onClick={onDelete}
          className="absolute top-2 right-2 rounded-md bg-background/80 p-1.5 text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
        >
          <HugeiconsIcon icon={Delete02Icon} size={14} />
        </button>
      )}
    </div>
  );
}

function ImportDialog({
  open,
  onOpenChange,
  onImported,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImported: (flow: FlowRecord) => void;
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{
    flow: FlowRecord | null;
    report: ImportReport;
  } | null>(null);
  const {
    ref: dropRef,
    dragging,
    dragHandlers,
  } = useNativeFileDrop({
    accept: ".unslothflow,.json,.ipynb",
    multiple: false,
    disabled: busy,
    onFiles: (files) => {
      const file = files[0];
      if (file) return submit(file);
    },
  });
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setText("");
      setResult(null);
      setBusy(false);
    }
  }, [open]);

  const submit = useCallback(async (data: Blob | string, name?: string) => {
    setBusy(true);
    try {
      const notebook = notebookName(data, name);
      setResult(
        notebook
          ? await importNotebook(data, notebook)
          : await importFlow(data, name),
      );
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Import failed");
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>
            {result
              ? result.flow
                ? `Imported “${result.flow.name}”`
                : "Open this notebook in Colab"
              : "Import a flow"}
          </DialogTitle>
          <DialogDescription>
            {result
              ? result.flow
                ? "Nothing ran or downloaded. Here is what it needs on this computer."
                : "Studio cannot train this notebook yet. Nothing ran or downloaded."
              : "Drop a .unslothflow, .json or Unsloth notebook (.ipynb) file, or paste the flow's JSON. Imports never run anything by themselves."}
          </DialogDescription>
        </DialogHeader>
        {result ? (
          <div className="flex flex-col gap-3">
            {(result.report.author ||
              result.report.license ||
              result.report.content_rating) && (
              <p className="text-muted-foreground text-sm">
                {[
                  result.report.author && `By ${result.report.author}`,
                  result.report.license && `License ${result.report.license}`,
                  result.report.content_rating &&
                    `Rated ${result.report.content_rating}`,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
            )}
            <ModelRequirementList models={result.report.missing_models} />
            {result.report.baked_nodes.length > 0 && (
              <p className="text-muted-foreground text-sm">
                {result.report.baked_nodes.length} step
                {result.report.baked_nodes.length === 1 ? " comes" : "s come"}{" "}
                pre-generated, so those parts work without models.
              </p>
            )}
            {result.report.live_note && (
              <p className="text-muted-foreground text-sm">
                {result.report.live_note}
              </p>
            )}
            {(result.report.replication ?? []).map((line) => (
              <p
                key={line}
                className="text-amber-600 text-sm dark:text-amber-400"
              >
                {line}
              </p>
            ))}
            {(result.report.unsupported ?? []).map((line) => (
              <p key={line} className="text-destructive text-sm">
                {line}
              </p>
            ))}
            {result.report.warnings.map((warning) => (
              <p
                key={warning}
                className="text-amber-600 text-sm dark:text-amber-400"
              >
                {warning}
              </p>
            ))}
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              ref={dropRef}
              {...dragHandlers}
              className={`flex flex-col items-center gap-2 rounded-xl border-2 border-dashed px-4 py-8 text-sm transition-colors ${
                dragging
                  ? "border-primary bg-primary/5"
                  : "border-border hover:bg-muted/50"
              }`}
              disabled={busy}
            >
              <HugeiconsIcon
                icon={FileImportIcon}
                size={26}
                className="text-muted-foreground"
              />
              {busy ? "Importing…" : "Drop a file here or click to choose"}
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".unslothflow,.json,.ipynb,application/json,application/zip"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                e.target.value = "";
                if (file) void submit(file);
              }}
            />
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder='…or paste JSON: {"format": "unsloth-flow", …}'
              rows={5}
              className="rounded-md border bg-transparent px-2 py-1.5 font-mono text-xs"
            />
          </div>
        )}
        <DialogFooter>
          {result?.flow ? (
            <Button onClick={() => result.flow && onImported(result.flow)}>
              Open flow
            </Button>
          ) : result ? (
            result.report.colab_url && (
              <Button asChild={true}>
                <a
                  href={result.report.colab_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open in Colab
                </a>
              </Button>
            )
          ) : (
            <Button
              disabled={busy || !text.trim()}
              onClick={() => void submit(text.trim())}
            >
              Import
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function FlowsGalleryPage(): ReactElement {
  const navigate = useNavigate();
  const [flows, setFlows] = useState<FlowSummary[] | null>(null);
  const [examples, setExamples] = useState<FlowSummary[]>([]);
  const [importOpen, setImportOpen] = useState(false);
  const [category, setCategory] = useState<GalleryCategory | null>(null);
  const [creating, setCreating] = useState(false);
  const [migrating, setMigrating] = useState(false);
  const [recipeImportOpen, setRecipeImportOpen] = useState(false);

  const refresh = useCallback(() => {
    listFlows()
      .then((r) => setFlows(r.flows))
      .catch((e: unknown) => {
        setFlows([]);
        toast.error(e instanceof Error ? e.message : "Could not load flows");
      });
    listExamples()
      .then((r) => setExamples(r.examples))
      .catch(() => setExamples([]));
  }, []);

  useEffect(() => refresh(), [refresh]);

  const open = useCallback(
    (flowId: string) =>
      void navigate({ to: "/flows/$flowId", params: { flowId } }),
    [navigate],
  );

  const createFrom = async (kind: FlowKind) => {
    setCreating(true);
    try {
      const record = await createFlow(templateDocument(kind));
      open(record.id);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Could not create the flow",
      );
    } finally {
      setCreating(false);
    }
  };

  const newRecipe = async () => {
    setCreating(true);
    try {
      const doc = templateDocument("pipeline");
      const { group, nodes } = starterRecipeGraph([0, 0], {
        rows: 5,
        mode: "preview",
      });
      const record = await createFlow({
        ...doc,
        name: "New recipe",
        description: "A Data Recipe: columns that generate a dataset.",
        nodes,
        edges: referenceEdges(nodes.filter((n) => n.type === COLUMN_TYPE)),
        groups: [group],
        notes: [],
      });
      open(record.id);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Could not create the recipe",
      );
    } finally {
      setCreating(false);
    }
  };

  const bringRecipes = async () => {
    setMigrating(true);
    try {
      const result = await importAllRecipes();
      const refreshed = result.created.filter((c) => c.replaces).length;
      const made = result.created.length - refreshed;
      const kept = result.skipped.length;
      if (made + refreshed === 0 && kept === 0 && result.failed.length === 0)
        toast.info("No saved Data Recipes yet");
      else if (made + refreshed === 0 && kept === 0)
        toast.error("Could not bring in your Data Recipes", {
          description: result.failed.map((f) => `${f.name}: ${f.error}`).join("; "),
        });
      else
        toast.success(
          made + refreshed
            ? [
                made && `Brought ${made} Data Recipe${made === 1 ? "" : "s"} into Flows`,
                refreshed &&
                  `${refreshed} edited recipe${refreshed === 1 ? "" : "s"} brought in again`,
                kept && `${kept} already here`,
              ]
                .filter(Boolean)
                .join("; ")
            : `All ${kept} Data Recipe${kept === 1 ? " is" : "s are"} already in Flows`,
          {
            description: result.failed.length
              ? `Could not bring in: ${result.failed.map((f) => f.name).join(", ")}`
              : "Your recipes stay on the Data Recipes page too.",
          },
        );
      const notices = result.created.flatMap((c) =>
        (c.notices ?? []).map((notice) => `${c.name}: ${notice}`),
      );
      for (const notice of notices.slice(0, 4)) toast.info(notice);
      if (notices.length > 4)
        toast.info(`${notices.length - 4} more notes on the brought-in recipes`);
      refresh();
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Could not bring in the recipes",
      );
    } finally {
      setMigrating(false);
    }
  };

  return (
    <div
      className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-6 py-8"
      data-testid="flows-gallery"
    >
      <header className="flex flex-wrap items-end gap-4">
        <div className="flex-1">
          <h1 className="font-heading font-semibold text-2xl tracking-tight">
            Flows
          </h1>
          <p className="mt-1 max-w-2xl text-muted-foreground text-sm">
            Connect blocks for language models, images, video, audio and
            training into pipelines and games. Everything runs on this computer,
            and any flow can be shared as a single file.
          </p>
        </div>
        <Button
          variant="outline"
          disabled={creating}
          onClick={() => void newRecipe()}
          title="A new flow with a starter Data Recipe in it"
          data-testid="flows-new-recipe"
        >
          <HugeiconsIcon icon={PlusSignIcon} size={15} /> New recipe
        </Button>
        <Button
          variant="outline"
          disabled={migrating}
          onClick={() => void bringRecipes()}
          title="One flow per saved Data Recipe, each an editable recipe group. Safe to run again."
          data-testid="flows-bring-recipes"
        >
          <HugeiconsIcon icon={TestTube01Icon} size={15} />{" "}
          {migrating ? "Bringing in…" : "Bring in my Data Recipes"}
        </Button>
        <Button
          variant="outline"
          onClick={() => setRecipeImportOpen(true)}
          data-testid="flows-import-recipe"
        >
          <HugeiconsIcon icon={FileImportIcon} size={15} /> Import recipe JSON
        </Button>
        <Button
          variant="outline"
          onClick={() => setImportOpen(true)}
          data-testid="flows-import"
        >
          <HugeiconsIcon icon={FileImportIcon} size={15} /> Import
        </Button>
      </header>

      <section className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {(["pipeline", "story", "world"] as const).map((kind) => {
          const style = KIND_STYLE[kind];
          const copy = {
            pipeline: "Chain models together, like text → image → video.",
            story: "A branching visual novel with generated art.",
            world: "A little world to walk around and talk to its people.",
          }[kind];
          return (
            <button
              key={kind}
              type="button"
              disabled={creating}
              onClick={() => void createFrom(kind)}
              className="flex items-center gap-3 rounded-xl border bg-card px-4 py-3 text-left transition-colors hover:bg-muted/60 disabled:opacity-60"
              data-testid={`flows-new-${kind}`}
            >
              <span
                className={`flex size-10 shrink-0 items-center justify-center rounded-lg ${style.className}`}
              >
                <HugeiconsIcon icon={style.icon} size={20} />
              </span>
              <span className="flex flex-col">
                <span className="flex items-center gap-1 font-medium text-sm">
                  <HugeiconsIcon icon={PlusSignIcon} size={12} /> New{" "}
                  {style.label.toLowerCase()}
                </span>
                <span className="text-muted-foreground text-xs">{copy}</span>
              </span>
            </button>
          );
        })}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="font-semibold text-sm">My flows</h2>
        {flows === null ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="aspect-[4/3] animate-pulse rounded-xl bg-muted"
              />
            ))}
          </div>
        ) : flows.length === 0 ? (
          <p className="rounded-xl border border-dashed px-4 py-8 text-center text-muted-foreground text-sm">
            No flows yet. Start from a template above or remix an example below.
          </p>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {flows.map((flow) => (
              <FlowCard
                key={flow.id}
                flow={flow}
                onOpen={() => open(flow.id)}
                onApp={
                  flow.kind === "pipeline"
                    ? () =>
                        void navigate({
                          to: "/flows/$flowId/app",
                          params: { flowId: flow.id },
                        })
                    : undefined
                }
                onDelete={() => {
                  if (!window.confirm(`Delete “${flow.name}”?`)) return;
                  deleteFlow(flow.id)
                    .then(refresh)
                    .catch((e: unknown) =>
                      toast.error(
                        e instanceof Error ? e.message : "Delete failed",
                      ),
                    );
                }}
              />
            ))}
          </div>
        )}
      </section>

      {examples.length > 0 && (
        <section className="flex flex-col gap-3">
          <h2 className="font-semibold text-sm">Examples</h2>
          <div
            className="flex flex-wrap gap-1.5"
            role="tablist"
            aria-label="Example categories"
          >
            {[
              { id: null, label: "All", count: examples.length },
              ...visibleCategories(examples),
            ].map((chip) => (
              <button
                key={chip.id ?? "all"}
                type="button"
                role="tab"
                aria-selected={category === chip.id}
                onClick={() => setCategory(chip.id)}
                className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                  category === chip.id
                    ? "border-foreground/30 bg-foreground text-background"
                    : "text-muted-foreground hover:bg-muted"
                }`}
                data-testid="flows-category"
              >
                {chip.label} <span className="opacity-60">{chip.count}</span>
              </button>
            ))}
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {examples
              .filter(
                (example) =>
                  category === null || categoriesOf(example).includes(category),
              )
              .map((example) => (
                <FlowCard
                  key={example.id}
                  flow={example}
                  action="Use this"
                  onOpen={() => {
                    copyExample(example.id)
                      .then((record) => open(record.id))
                      .catch((e: unknown) =>
                        toast.error(
                          e instanceof Error
                            ? e.message
                            : "Could not copy the example",
                        ),
                      );
                  }}
                />
              ))}
          </div>
        </section>
      )}

      <NotebooksSection
        onUseTemplate={(exampleId) =>
          copyExample(exampleId)
            .then((record) => open(record.id))
            .catch((e: unknown) =>
              toast.error(
                e instanceof Error ? e.message : "Could not copy the template",
              ),
            )
        }
        onImport={() => setImportOpen(true)}
      />

      <RecipeImportDialog
        open={recipeImportOpen}
        onOpenChange={setRecipeImportOpen}
        onImported={(record) => {
          setRecipeImportOpen(false);
          open(record.id);
        }}
      />
      <ImportDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        onImported={(record) => {
          setImportOpen(false);
          open(record.id);
        }}
      />
    </div>
  );
}
