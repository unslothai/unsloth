// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import {
  AlertCircleIcon,
  CheckmarkCircle02Icon,
  Delete02Icon,
  Download01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import {
  type NpuModel,
  type NpuStatus,
  deleteNpuModel,
  downloadNpuModel,
  enableNpu,
  isNpuRuntimeReady,
  listNpuModels,
} from "./api";

const TAGS: Array<[keyof NpuModel, string]> = [
  ["supports_reasoning", "Reasoning"],
  ["supports_tools", "Tools"],
  ["supports_vision", "Vision"],
];

function sizeLabel(sizeGb: number | null): string | null {
  if (sizeGb == null) return null;
  return sizeGb < 1
    ? `${Math.round(sizeGb * 1000)} MB`
    : `${sizeGb.toFixed(1)} GB`;
}

/** NPU setup and model management; selection uses the normal load flow. */
export function NpuModelPanel({
  status,
  onStatusChange,
  value,
  loadedModelId,
  onPick,
  sectionToggle,
}: {
  status: NpuStatus;
  onStatusChange: (status: NpuStatus) => void;
  value?: string;
  loadedModelId?: string;
  onPick: (modelPath: string) => void;
  sectionToggle: ReactNode;
}) {
  const [models, setModels] = useState<NpuModel[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [enabling, setEnabling] = useState(false);
  const [downloads, setDownloads] = useState<Record<string, number | null>>({});
  const ready = isNpuRuntimeReady(status);

  const refreshModels = useCallback(async () => {
    try {
      setModels(await listNpuModels());
      setListError(null);
    } catch (error) {
      setListError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    if (!status.runtime_installed || status.state === "failed") return;
    let live = true;
    listNpuModels().then(
      (next) => {
        if (!live) return;
        setModels(next);
        setListError(null);
      },
      (error) => {
        if (live) {
          setListError(error instanceof Error ? error.message : String(error));
        }
      },
    );
    return () => {
      live = false;
    };
  }, [status.runtime_installed, status.state]);

  const enable = async () => {
    setEnabling(true);
    try {
      onStatusChange(await enableNpu());
      await refreshModels();
    } catch (error) {
      toast.error("Could not enable the NPU", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setEnabling(false);
    }
  };

  const download = async (model: NpuModel) => {
    setDownloads((current) => ({ ...current, [model.id]: null }));
    try {
      await downloadNpuModel(model.id, (event) => {
        if (typeof event.percent === "number") {
          setDownloads((current) => ({
            ...current,
            [model.id]: event.percent ?? null,
          }));
        }
      });
      await refreshModels();
    } catch (error) {
      toast.error(`Could not download ${model.id}`, {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setDownloads((current) => {
        const next = { ...current };
        delete next[model.id];
        return next;
      });
    }
  };

  const remove = async (model: NpuModel) => {
    try {
      await deleteNpuModel(model.id);
      await refreshModels();
    } catch (error) {
      toast.error(`Could not delete ${model.id}`, {
        description: error instanceof Error ? error.message : String(error),
      });
    }
  };

  const problems = status.validation?.problems ?? [];
  return (
    <div className="flex flex-col gap-3 pr-2">
      <div className="flex flex-wrap items-center gap-2">{sectionToggle}</div>
      <p className="text-xs text-muted-foreground">
        {status.hardware.name ?? "AMD NPU"}: models run on FastFlowLM through
        Lemonade. They are FastFlowLM's own NPU builds; GGUF models run on the
        GPU.
      </p>
      {status.error || problems.length > 0 ? (
        <div className="flex gap-2 rounded-lg bg-destructive/10 p-2.5 text-xs text-destructive">
          <HugeiconsIcon
            icon={AlertCircleIcon}
            className="mt-0.5 size-3.5 shrink-0"
          />
          <div className="flex flex-col gap-1">
            <span>{status.error ?? problems.join(" ")}</span>
            {status.help_url ? (
              <a
                href={status.help_url}
                target="_blank"
                rel="noreferrer"
                className="underline underline-offset-2"
              >
                NPU driver setup
              </a>
            ) : null}
          </div>
        </div>
      ) : null}
      {!ready || !status.runtime_installed ? (
        <div className="flex items-center justify-between gap-3 rounded-lg bg-muted/50 p-3 text-xs">
          <span className="text-muted-foreground">
            Enabling downloads Lemonade (about 7 MB) and FastFlowLM (about 40
            MB), then checks the NPU driver.
          </span>
          <Button size="sm" onClick={() => void enable()} disabled={enabling}>
            {enabling ? <Spinner className="size-3.5" /> : null}
            {status.state === "failed"
              ? "Try again"
              : status.runtime_installed
                ? "Start NPU runtime"
                : "Enable NPU"}
          </Button>
        </div>
      ) : null}
      {listError ? (
        <p className="text-xs text-destructive">{listError}</p>
      ) : null}
      {models ? (
        <div className="model-list-scroll max-h-[calc(335px*var(--ui-space-scale,1))] overflow-y-auto px-0.5 pb-3">
          {models.map((model) => {
            const progress = downloads[model.id];
            const downloading = model.id in downloads;
            const selected = value === model.model_path;
            const loaded = loadedModelId === model.model_path;
            return (
              <div
                key={model.id}
                className={cn(
                  "flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm",
                  model.downloaded && "hover:bg-muted/60",
                  selected && "bg-muted/60",
                )}
              >
                <button
                  type="button"
                  className="flex min-w-0 flex-1 flex-col text-left disabled:cursor-default"
                  disabled={!model.downloaded || downloading}
                  aria-current={selected ? "true" : undefined}
                  title={
                    model.downloaded
                      ? undefined
                      : "Download this model to load it"
                  }
                  onClick={() => onPick(model.model_path)}
                  data-model-picker-option={true}
                >
                  <span className="truncate font-medium">{model.id}</span>
                  <span className="flex flex-wrap gap-x-2 text-xs text-muted-foreground">
                    {sizeLabel(model.size_gb) ? (
                      <span>{sizeLabel(model.size_gb)}</span>
                    ) : null}
                    {TAGS.filter(([key]) => model[key]).map(([, label]) => (
                      <span key={label}>{label}</span>
                    ))}
                  </span>
                </button>
                {loaded ? (
                  <HugeiconsIcon
                    icon={CheckmarkCircle02Icon}
                    className="size-4 shrink-0 text-primary"
                    aria-label="Loaded"
                  />
                ) : null}
                {downloading ? (
                  <span className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Spinner className="size-3.5" />
                    {progress != null ? `${Math.round(progress)}%` : null}
                  </span>
                ) : model.downloaded ? (
                  <Button
                    size="icon-xs"
                    variant="ghost"
                    aria-label={`Delete ${model.id}`}
                    disabled={loaded}
                    onClick={() => void remove(model)}
                  >
                    <HugeiconsIcon icon={Delete02Icon} />
                  </Button>
                ) : (
                  <Button
                    size="icon-xs"
                    variant="ghost"
                    aria-label={`Download ${model.id}`}
                    onClick={() => void download(model)}
                  >
                    <HugeiconsIcon icon={Download01Icon} />
                  </Button>
                )}
              </div>
            );
          })}
        </div>
      ) : ready && status.runtime_installed && !listError ? (
        <div className="flex justify-center py-4">
          <Spinner className="size-4" />
        </div>
      ) : null}
      <p className="pb-2 text-[11px] text-muted-foreground">
        Powered by{" "}
        <a
          href="https://github.com/ROCm/FastFlowLM"
          target="_blank"
          rel="noreferrer"
          className="underline underline-offset-2"
        >
          FastFlowLM
        </a>{" "}
        and{" "}
        <a
          href="https://github.com/lemonade-sdk/lemonade"
          target="_blank"
          rel="noreferrer"
          className="underline underline-offset-2"
        >
          Lemonade
        </a>
        .
      </p>
    </div>
  );
}
