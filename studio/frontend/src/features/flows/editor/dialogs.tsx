// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
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
import { FieldLabel } from "@/features/recipe-studio/dialogs/shared/field-label";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
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
            <div key={param.id} className="grid gap-1.5">
              <FieldLabel label={param.label ?? param.id} />
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
                <Checkbox
                  aria-label={param.label ?? param.id}
                  checked={!!values[param.id]}
                  onCheckedChange={(checked) =>
                    setValues((v) => ({ ...v, [param.id]: checked === true }))
                  }
                />
              ) : (
                <Textarea
                  fieldSizing="fixed"
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
                  className="min-h-0"
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
    <ul className="flex flex-col divide-y divide-border/60 rounded-lg border border-border/60">
      {models.map((model) => (
        <li
          key={`${model.kind}:${model.repo}:${model.variant ?? ""}`}
          className="flex items-center gap-3 px-3 py-2 text-sm"
        >
          <Badge variant="secondary" className="w-16 shrink-0 text-ui-10 uppercase">
            {model.kind}
          </Badge>
          <span
            className="min-w-0 flex-1 truncate font-mono text-xs"
            title={model.repo}
          >
            {model.repo}
            {model.variant ? ` · ${model.variant}` : ""}
          </span>
          <span className="text-xs text-muted-foreground">
            {formatBytes(model.size_bytes)}
          </span>
          <Badge
            variant="outline"
            className={
              model.loaded
                ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                : model.downloaded
                  ? "border-border bg-muted text-foreground"
                  : model.fits === false
                    ? "border-destructive/25 bg-destructive/10 text-destructive"
                    : "border-amber-500/25 bg-amber-500/10 text-amber-700 dark:text-amber-300"
            }
          >
            {model.loaded
              ? "loaded"
              : model.downloaded
                ? "downloaded"
                : model.fits === false
                  ? "too big"
                  : "will download"}
          </Badge>
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
  const callsFlows = useDocStore((s) =>
    Object.values(s.graph.nodes).some(
      (n) => n?.type === "flow.call" && !n.disabled,
    ),
  );
  const [plan, setPlan] = useState<FlowPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!open || !flowId) return;
    setPlan(null);
    setError(null);
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
          <p className="flex items-center gap-2 text-muted-foreground text-sm">
            <Spinner /> Checking…
          </p>
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
            {(plan.hints ?? []).map((hint) => (
              <div
                key={hint.id}
                className="flex items-start gap-3 rounded-lg border border-border/60 bg-muted/10 px-3 py-2 text-sm"
                data-testid={`flow-plan-hint-${hint.id}`}
              >
                <p className="min-w-0 flex-1 text-muted-foreground">
                  {hint.message}
                </p>
                {hint.id === "multi_model" && (
                  <Button
                    size="sm"
                    variant="outline"
                    className="shrink-0"
                    onClick={() => {
                      onOpenChange(false);
                      useSettingsDialogStore
                        .getState()
                        .openDialog("resources", {
                          scrollTarget: "resources-model-memory",
                        });
                    }}
                  >
                    Open setting
                  </Button>
                )}
              </div>
            ))}
            {(plan.models.length > 0 || callsFlows) && <UnloadAfterRunToggle />}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

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

function UnloadAfterRunToggle() {
  const extra = useDocStore((s) => s.meta.meta);
  const setMeta = useDocStore((s) => s.setMeta);
  const on = extra?.unload_after_run === true;
  return (
    <div className="flex items-center gap-3 text-sm">
      <Switch
        id="flow-unload-after-run"
        checked={on}
        onCheckedChange={(checked) =>
          setMeta({ meta: { ...(extra ?? {}), unload_after_run: checked } })
        }
      />
      <label htmlFor="flow-unload-after-run" className="flex-1">
        Unload models when the run ends
        <span className="block text-muted-foreground text-xs">
          Frees the memory of models this flow loaded. Models loaded before the
          run stay.
        </span>
      </label>
    </div>
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
      "The recipe only: blocks, settings and seeds. The receiver makes everything with their own models.",
    ext: "json",
  },
  {
    mode: "package",
    title: "Flow + files (.unslothflow)",
    description:
      "The recipe plus the images, audio and documents you added to it.",
    ext: "unslothflow",
  },
  {
    mode: "baked",
    title: "Playable (.unslothflow, baked)",
    description:
      "Also every output it made, so it plays with no models. Parts made while playing, like characters' replies, still need one.",
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
            Anyone with Unsloth can import the file; it records the Studio
            version and model revisions so they can see if theirs differ.
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
              className="flex flex-col items-start gap-0.5 rounded-lg border border-border/60 px-3 py-3 text-left transition hover:bg-muted/25 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <span className="font-medium text-foreground text-sm">
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
