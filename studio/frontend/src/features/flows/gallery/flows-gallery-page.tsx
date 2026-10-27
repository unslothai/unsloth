// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useNativeFileDrop } from "@/features/native-intents/use-native-file-drop";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
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
} from "../editor/icon-data";
import { MediaThumb } from "../editor/media";
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
  { label: string; icon: typeof FlowConnectionIcon; copy: string }
> = {
  pipeline: {
    label: "Pipeline",
    icon: FlowConnectionIcon,
    copy: "Chain models together, like text → image → video.",
  },
  story: {
    label: "Story game",
    icon: BookOpen01Icon,
    copy: "A branching visual novel with generated art.",
  },
  world: {
    label: "World game",
    icon: GameController01Icon,
    copy: "A little world to walk around and talk to its people.",
  },
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
    <Card
      className="group relative gap-0 py-0 transition-transform hover:-translate-y-0.5"
      data-testid="flows-card"
    >
      <button
        type="button"
        onClick={onOpen}
        className="flex flex-col text-left"
      >
        <div className="relative flex aspect-[16/9] items-center justify-center overflow-hidden border-border/60 border-b bg-muted/30">
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
            <div className="inline-flex size-10 items-center justify-center rounded-md border border-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)] bg-background/80">
              <HugeiconsIcon
                icon={kind.icon}
                className="size-5 text-foreground/90"
              />
            </div>
          )}
        </div>
        <div className="flex flex-col gap-1 p-4">
          <div className="flex items-center gap-2">
            <p className="min-w-0 flex-1 truncate text-sm font-semibold text-foreground">
              {flow.name}
            </p>
            <Badge variant="outline" className="shrink-0">
              {kind.label}
            </Badge>
          </div>
          {flow.description && (
            <p className="line-clamp-2 text-xs text-muted-foreground">
              {flow.description}
            </p>
          )}
          {flow.example && needsLine(flow.needs) && (
            <p
              className="text-xs text-foreground/70"
              data-testid="flows-card-needs"
            >
              {needsLine(flow.needs)}
            </p>
          )}
          <p className="text-xs text-muted-foreground">
            {flow.nodeCount} nodes
            {flow.updatedAt && !flow.example
              ? ` · ${relativeTime(flow.updatedAt)}`
              : ""}
          </p>
        </div>
      </button>
      {action && (
        <Badge
          variant="secondary"
          className="pointer-events-none absolute top-2 right-2 opacity-0 shadow-sm transition-opacity group-hover:opacity-100"
        >
          {action}
        </Badge>
      )}
      {onApp && (
        <Button
          type="button"
          variant="secondary"
          size="icon-sm"
          aria-label={`Open ${flow.name} as an app`}
          title="Open as app"
          onClick={onApp}
          className="absolute top-2 right-11 opacity-0 transition-opacity group-hover:opacity-100"
          data-testid="flows-card-app"
        >
          <HugeiconsIcon icon={GridViewIcon} size={14} />
        </Button>
      )}
      {onDelete && (
        <Button
          type="button"
          variant="secondary"
          size="icon-sm"
          aria-label={`Delete ${flow.name}`}
          title="Delete"
          onClick={onDelete}
          className="absolute top-2 right-2 opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
        >
          <HugeiconsIcon icon={Delete02Icon} size={14} />
        </Button>
      )}
    </Card>
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
              className={`flex flex-col items-center gap-2 rounded-xl border-2 border-dashed px-4 py-8 text-center text-muted-foreground text-sm transition-colors ${
                dragging
                  ? "border-ring-strong bg-wash-5"
                  : "border-muted-foreground/25 hover:border-muted-foreground/50"
              }`}
              disabled={busy}
            >
              <HugeiconsIcon
                icon={FileImportIcon}
                className="size-8 text-muted-foreground"
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
            <Textarea
              fieldSizing="fixed"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder='…or paste JSON: {"format": "unsloth-flow", …}'
              rows={5}
              className="font-mono text-xs"
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

  return (
    <div
      className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-5 py-8 max-sm:px-4 sm:px-9"
      data-testid="flows-gallery"
    >
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div className="min-w-0 flex-1">
          <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
            Flows
          </h1>
          <p className="mt-0.5 max-w-2xl text-sm text-muted-foreground">
            Connect blocks for language models, images, video, audio and
            training into pipelines and games. Everything runs on this computer,
            and any flow can be shared as a single file.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => setImportOpen(true)}
            data-testid="flows-import"
          >
            <HugeiconsIcon icon={FileImportIcon} className="size-4" /> Import
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild={true}>
              <Button variant="secondary" type="button" disabled={creating}>
                <HugeiconsIcon icon={PlusSignIcon} className="size-4" />
                New flow
                <HugeiconsIcon
                  icon={ChevronDownStandardIcon}
                  className="size-4"
                />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-72">
              {(["pipeline", "story", "world"] as const).map((kind) => (
                <DropdownMenuItem
                  key={kind}
                  onSelect={() => void createFrom(kind)}
                  className="items-start"
                  data-testid={`flows-new-${kind}`}
                >
                  <HugeiconsIcon
                    icon={KIND_STYLE[kind].icon}
                    className="mt-0.5 size-4"
                  />
                  <span className="flex flex-col">
                    <span>New {KIND_STYLE[kind].label.toLowerCase()}</span>
                    <span className="text-xs text-muted-foreground">
                      {KIND_STYLE[kind].copy}
                    </span>
                  </span>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      <section className="flex flex-col gap-3">
        <h2 className="font-semibold text-sm">My flows</h2>
        {flows === null ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="aspect-[4/3] rounded-2xl" />
            ))}
          </div>
        ) : flows.length === 0 ? (
          <Empty className="border border-dashed border-border/70 p-8 dark:border-none">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <HugeiconsIcon icon={FlowConnectionIcon} className="size-5" />
              </EmptyMedia>
              <EmptyTitle>No flows yet</EmptyTitle>
              <EmptyDescription>
                Start one with New flow or remix an example below.
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
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
              <Button
                key={chip.id ?? "all"}
                type="button"
                size="xs"
                variant={category === chip.id ? "secondary" : "ghost"}
                role="tab"
                aria-selected={category === chip.id}
                onClick={() => setCategory(chip.id)}
                className={`rounded-full ${category === chip.id ? "" : "text-muted-foreground"}`}
                data-testid="flows-category"
              >
                {chip.label} <span className="opacity-60">{chip.count}</span>
              </Button>
            ))}
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
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
