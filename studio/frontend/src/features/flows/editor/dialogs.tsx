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
import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  type ExportMode,
  type FlowPlan,
  type PlanModel,
  exportFlow,
  planFlow,
} from "../api";
import { documentFromState, useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import type { FlowParam } from "../types";
import { downloadBytes, modelSignature, modelsToDownload } from "../ux-rules";
import { downloadBlob } from "./node-actions";
import { ParamField } from "./param-fields";
import { runFlow } from "./run-controller";

export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes || bytes <= 0) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

/** Flow inputs: the `input.param` nodes plus any declared doc params. */
export function collectFlowParams(): FlowParam[] {
  const { graph, meta } = useDocStore.getState();
  const params = new Map<string, FlowParam>();
  for (const param of meta.params) params.set(param.id, param);
  for (const id of graph.nodeOrder) {
    const node = graph.nodes[id];
    if (node.type !== "input.param") continue;
    const name = String(node.params.name ?? "").trim();
    if (!name || params.has(name)) continue;
    params.set(name, {
      id: name,
      type: (node.params.type as FlowParam["type"]) ?? "text",
      label: node.label ?? name,
      default: node.params.default ?? "",
    });
  }
  return [...params.values()];
}

export function RunDialog({
  open,
  onOpenChange,
}: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const [params, setParams] = useState<FlowParam[]>([]);
  const [values, setValues] = useState<Record<string, unknown>>({});
  useEffect(() => {
    if (!open) return;
    const next = collectFlowParams();
    setParams(next);
    setValues(Object.fromEntries(next.map((p) => [p.id, p.default ?? ""])));
  }, [open]);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Run flow</DialogTitle>
          <DialogDescription>
            Set the flow's inputs for this run.
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-3">
          {params.map((param) => (
            <div key={param.id} className="flex flex-col gap-1 text-sm">
              <span className="font-medium">{param.label ?? param.id}</span>
              {param.type === "image" || param.type === "audio" ? (
                <ParamField
                  spec={{
                    id: param.id,
                    kind: "asset",
                    default: null,
                    label: param.label ?? param.id,
                    accept: param.type,
                  }}
                  value={values[param.id] ?? null}
                  onChange={(value) =>
                    setValues((v) => ({ ...v, [param.id]: value }))
                  }
                />
              ) : param.type === "bool" ? (
                <input
                  type="checkbox"
                  aria-label={param.label ?? param.id}
                  checked={!!values[param.id]}
                  onChange={(e) =>
                    setValues((v) => ({ ...v, [param.id]: e.target.checked }))
                  }
                />
              ) : (
                <textarea
                  aria-label={param.label ?? param.id}
                  rows={param.type === "text" ? 3 : 1}
                  value={String(values[param.id] ?? "")}
                  onChange={(e) =>
                    setValues((v) => ({
                      ...v,
                      [param.id]:
                        param.type === "number"
                          ? Number(e.target.value)
                          : e.target.value,
                    }))
                  }
                  className="rounded-md border bg-transparent px-2 py-1.5 text-sm"
                />
              )}
            </div>
          ))}
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => {
              onOpenChange(false);
              void runFlow({ params: values });
            }}
          >
            Run
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function ModelRequirementList({ models }: { models: PlanModel[] }) {
  if (!models.length)
    return <p className="text-muted-foreground text-sm">No models needed.</p>;
  return (
    <ul className="flex flex-col divide-y rounded-md border">
      {models.map((model) => (
        <li
          key={`${model.kind}:${model.repo}:${model.variant ?? ""}`}
          className="flex items-center gap-3 px-3 py-2 text-sm"
        >
          <span className="w-16 shrink-0 rounded bg-muted px-1.5 py-0.5 text-center text-[11px] uppercase">
            {model.kind}
          </span>
          <span
            className="min-w-0 flex-1 truncate font-mono text-[12px]"
            title={model.repo}
          >
            {model.repo}
            {model.variant ? ` · ${model.variant}` : ""}
          </span>
          <span className="text-[11px] text-muted-foreground">
            {formatBytes(model.size_bytes)}
          </span>
          <span
            className={`text-[11px] ${
              model.loaded
                ? "text-emerald-500"
                : model.downloaded
                  ? "text-sky-500"
                  : model.fits === false
                    ? "text-red-500"
                    : "text-amber-500"
            }`}
          >
            {model.loaded
              ? "loaded"
              : model.downloaded
                ? "downloaded"
                : model.fits === false
                  ? "too big"
                  : "will download"}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function PlanDialog({
  open,
  onOpenChange,
}: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const flowId = useDocStore((s) => s.flowId);
  const [plan, setPlan] = useState<FlowPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!open || !flowId) return;
    setPlan(null);
    setError(null);
    // Plan what is on screen, not the last autosave.
    planFlow(flowId, {}, documentFromState(useDocStore.getState()))
      .then(setPlan)
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : "Could not plan the flow"),
      );
  }, [open, flowId]);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>What this flow needs</DialogTitle>
          <DialogDescription>
            Models, downloads and GPU steps, checked against this computer.
          </DialogDescription>
        </DialogHeader>
        {error && <p className="text-destructive text-sm">{error}</p>}
        {!plan && !error && (
          <p className="text-muted-foreground text-sm">Checking…</p>
        )}
        {plan && (
          <div className="flex flex-col gap-3">
            <ModelRequirementList models={plan.models} />
            <p className="text-muted-foreground text-sm">
              {plan.gpu_ops} GPU step{plan.gpu_ops === 1 ? "" : "s"} ·{" "}
              {plan.cached_nodes} cached
            </p>
            {[...plan.errors, ...plan.warnings].map((issue, i) => (
              <p
                // biome-ignore lint/suspicious/noArrayIndexKey: issues have no id
                key={i}
                className={`text-sm ${i < plan.errors.length ? "text-destructive" : "text-amber-600 dark:text-amber-400"}`}
              >
                {issue.message}
              </p>
            ))}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

/** Shown instead of starting a run that would first download models: what, how big, one button to go ahead. */
export function DownloadRunDialog() {
  const pending = useUiStore((s) => s.pendingRun);
  const close = () => useUiStore.getState().setPendingRun(null);
  const missing = modelsToDownload(pending?.plan.models);
  const total = downloadBytes(missing);
  return (
    <Dialog open={pending !== null} onOpenChange={(open) => !open && close()}>
      <DialogContent className="sm:max-w-xl" data-testid="flows-download-run">
        <DialogHeader>
          <DialogTitle>Download models to run this flow?</DialogTitle>
          <DialogDescription>
            {missing.length === 1
              ? "One model this flow uses is not on this computer yet."
              : `${missing.length} models this flow uses are not on this computer yet.`}{" "}
            {total ? `About ${formatBytes(total)} to download. ` : ""}
            They download once, then the run starts.
          </DialogDescription>
        </DialogHeader>
        {pending && <ModelRequirementList models={pending.plan.models} />}
        <DialogFooter>
          <Button variant="outline" onClick={close}>
            Not now
          </Button>
          <Button
            onClick={() => {
              const options = pending?.options ?? {};
              // A pick changed while the dialog was open (undo): plan again rather than confirm unseen models.
              const same =
                pending?.models ===
                modelSignature(useDocStore.getState().graph.nodes);
              close();
              void runFlow({ ...options, downloadsConfirmed: same });
            }}
            data-testid="flows-download-and-run"
          >
            Download and run
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function slugify(name: string): string {
  return (
    name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60) || "flow"
  );
}

const EXPORT_OPTIONS: {
  mode: ExportMode;
  title: string;
  description: string;
  ext: string;
}[] = [
  {
    mode: "json",
    title: "Flow only (.json)",
    description:
      "The graph and settings. Small; the receiver generates everything with their own models.",
    ext: "json",
  },
  {
    mode: "package",
    title: "Flow + files (.unslothflow)",
    description: "Includes uploaded images and audio the flow uses.",
    ext: "unslothflow",
  },
  {
    mode: "baked",
    title: "Playable (.unslothflow, baked)",
    description:
      "Also embeds every generated output, so it plays on a computer with no models.",
    ext: "unslothflow",
  },
];

export function ShareDialog({
  open,
  onOpenChange,
}: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const flowId = useDocStore((s) => s.flowId);
  const name = useDocStore((s) => s.meta.name);
  const [busy, setBusy] = useState<ExportMode | null>(null);
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Share this flow</DialogTitle>
          <DialogDescription>
            API keys, provider settings and local file paths are never included.
            Anyone with Unsloth can import the file.
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-2">
          {EXPORT_OPTIONS.map((option) => (
            <button
              key={option.mode}
              type="button"
              disabled={!flowId || busy !== null}
              onClick={async () => {
                if (!flowId) return;
                setBusy(option.mode);
                try {
                  const blob = await exportFlow(flowId, option.mode);
                  downloadBlob(blob, `${slugify(name)}.${option.ext}`);
                } catch (error) {
                  toast.error(
                    error instanceof Error ? error.message : "Export failed",
                  );
                } finally {
                  setBusy(null);
                }
              }}
              className="flex flex-col items-start gap-0.5 rounded-lg border px-3 py-2.5 text-left hover:bg-muted disabled:opacity-60"
            >
              <span className="font-medium text-sm">
                {busy === option.mode ? "Preparing…" : option.title}
              </span>
              <span className="text-muted-foreground text-xs">
                {option.description}
              </span>
            </button>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
