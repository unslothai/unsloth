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
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { type TranslationKey, useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Copy01Icon, RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import {
  type ColabCapability,
  type ColabGpu,
  type ColabLaunchJob,
  type ColabStage,
  cancelColabLaunch,
  fetchColabCapability,
  fetchColabLaunch,
  startColabLaunch,
} from "../api/linked-instances";

const GPUS: { id: ColabGpu; hint: TranslationKey }[] = [
  { id: "T4", hint: "settings.apiKeys.linkedInstances.colab.gpuT4" },
  { id: "L4", hint: "settings.apiKeys.linkedInstances.colab.gpuL4" },
  { id: "A100", hint: "settings.apiKeys.linkedInstances.colab.gpuA100" },
  { id: "H100", hint: "settings.apiKeys.linkedInstances.colab.gpuH100" },
];

export const COLAB_STAGES: { id: ColabStage; label: TranslationKey }[] = [
  {
    id: "allocating",
    label: "settings.apiKeys.linkedInstances.colab.stageAllocating",
  },
  {
    id: "installing",
    label: "settings.apiKeys.linkedInstances.colab.stageInstalling",
  },
  {
    id: "starting",
    label: "settings.apiKeys.linkedInstances.colab.stageStarting",
  },
  {
    id: "linking",
    label: "settings.apiKeys.linkedInstances.colab.stageLinking",
  },
  { id: "ready", label: "settings.apiKeys.linkedInstances.colab.stageReady" },
];

const POLL_MS = 2000;

function SetupCommands({
  commands,
  wsl,
  distro,
}: {
  commands: string[];
  wsl: boolean;
  distro: string | null;
}) {
  const t = useT();
  if (commands.length === 0) return null;
  return (
    <div className="flex flex-col gap-2">
      <p className="text-ui-11 text-muted-foreground">
        {wsl
          ? t("settings.apiKeys.linkedInstances.colab.setupWsl", {
              command: distro ? `wsl -d ${distro}` : "wsl",
            })
          : t("settings.apiKeys.linkedInstances.colab.setupNative")}
      </p>
      <div className="relative rounded-md border border-border/60 bg-muted/30">
        <pre className="overflow-x-auto p-3 pr-10 font-mono text-ui-11 leading-relaxed text-foreground">
          {commands.join("\n")}
        </pre>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="absolute top-1.5 right-1.5 size-7 p-0 text-muted-foreground hover:text-foreground"
          aria-label={t("settings.apiKeys.linkedInstances.colab.copyCommands")}
          title={t("settings.apiKeys.linkedInstances.colab.copyCommands")}
          onClick={async () => {
            if (await copyToClipboard(commands.join("\n"))) {
              toast.success(t("settings.apiKeys.copied"));
            }
          }}
        >
          <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
        </Button>
      </div>
      {commands.some((c) => c.startsWith("colab --auth")) ? (
        <p className="text-ui-11 leading-snug text-muted-foreground">
          {t("settings.apiKeys.linkedInstances.colab.signInNote")}
        </p>
      ) : null}
    </div>
  );
}

function StageList({ job }: { job: ColabLaunchJob }) {
  const t = useT();
  const current = COLAB_STAGES.findIndex((s) => s.id === job.stage);
  return (
    <ol className="flex flex-col gap-2">
      {COLAB_STAGES.map((stage, index) => {
        const done =
          index < current || (job.state === "ready" && index === current);
        const active = index === current && job.state === "running";
        const failed =
          index === current &&
          (job.state === "failed" || job.state === "cancelled");
        return (
          <li
            key={stage.id}
            className={cn(
              "flex items-center gap-2.5 text-xs",
              done || active || failed
                ? "text-foreground"
                : "text-muted-foreground",
            )}
          >
            <span className="flex size-4 shrink-0 items-center justify-center">
              {active ? (
                <Spinner className="size-3.5" />
              ) : (
                <span
                  aria-hidden={true}
                  className={cn(
                    "size-2 rounded-full",
                    done
                      ? "bg-emerald-500"
                      : failed
                        ? "bg-red-500"
                        : "bg-muted-foreground/30",
                  )}
                />
              )}
            </span>
            {t(stage.label)}
          </li>
        );
      })}
    </ol>
  );
}

export function ColabLaunchDialog({
  open,
  onOpenChange,
  onFinished,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called once when a launch leaves the running state, so the card can reload. */
  onFinished: (job: ColabLaunchJob) => void;
}) {
  const t = useT();
  const id = useId();
  const [capability, setCapability] = useState<ColabCapability | null>(null);
  const [checking, setChecking] = useState(false);
  const [gpu, setGpu] = useState<ColabGpu>("L4");
  const [name, setName] = useState("colab-l4");
  const [nameEdited, setNameEdited] = useState(false);
  const [job, setJob] = useState<ColabLaunchJob | null>(null);
  const [showForm, setShowForm] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLPreElement>(null);
  const lastState = useRef<string | null>(null);

  const check = useCallback(async () => {
    setChecking(true);
    try {
      setCapability(await fetchColabCapability());
    } catch (e) {
      setCapability(null);
      setError(e instanceof Error ? e.message : null);
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    setError(null);
    void (async () => {
      const current = await fetchColabLaunch().catch(() => null);
      setJob(current);
      lastState.current = current?.state ?? null;
      const running = current?.state === "running";
      setShowForm(!running);
      if (!running) void check();
    })();
  }, [open, check]);

  useEffect(() => {
    if (!open || job?.state !== "running") return;
    const timer = window.setInterval(async () => {
      const next = await fetchColabLaunch().catch(() => null);
      if (!next) return;
      setJob(next);
      if (lastState.current === "running" && next.state !== "running") {
        onFinished(next);
      }
      lastState.current = next.state;
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [open, job?.state, onFinished]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [job?.log.length]);

  const pickGpu = (next: ColabGpu) => {
    setGpu(next);
    if (!nameEdited) setName(`colab-${next.toLowerCase()}`);
  };

  const launch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!capability?.ready || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const started = await startColabLaunch({ gpu, name: name.trim() });
      setJob(started);
      lastState.current = started.state;
      setShowForm(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : null);
    } finally {
      setSubmitting(false);
    }
  };

  const cancel = async () => {
    setCancelling(true);
    try {
      await cancelColabLaunch();
    } catch (err) {
      setError(err instanceof Error ? err.message : null);
    } finally {
      setCancelling(false);
    }
  };

  const wsl = capability?.runner === "wsl";
  const running = job?.state === "running";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>
            {t("settings.apiKeys.linkedInstances.colab.title")}
          </DialogTitle>
          <DialogDescription>
            {t("settings.apiKeys.linkedInstances.colab.description")}
          </DialogDescription>
        </DialogHeader>

        {showForm || !job ? (
          <form
            id={`${id}-form`}
            onSubmit={launch}
            className="flex min-w-0 flex-col gap-4"
          >
            {capability === null ? (
              <p className="flex items-center gap-2 text-xs text-muted-foreground">
                <Spinner className="size-3.5" />
                {t("settings.apiKeys.linkedInstances.colab.checking")}
              </p>
            ) : !capability.ready ? (
              <div className="flex flex-col gap-3 rounded-md border border-amber-500/40 bg-amber-500/5 p-3">
                <div className="flex items-start justify-between gap-3">
                  <p className="text-xs font-medium text-foreground">
                    {capability.message}
                  </p>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-7 shrink-0 gap-1.5 px-2 text-ui-11"
                    onClick={() => void check()}
                    disabled={checking}
                  >
                    <HugeiconsIcon
                      icon={RefreshIcon}
                      className={cn("size-3", checking && "animate-spin")}
                    />
                    {t("settings.apiKeys.linkedInstances.colab.checkAgain")}
                  </Button>
                </div>
                <SetupCommands
                  commands={capability.setup}
                  wsl={wsl}
                  distro={capability.distro}
                />
              </div>
            ) : null}

            <div className="flex flex-col gap-1.5">
              <span className="text-ui-11 font-medium text-muted-foreground">
                {t("settings.apiKeys.linkedInstances.colab.gpu")}
              </span>
              <div
                role="radiogroup"
                aria-label={t("settings.apiKeys.linkedInstances.colab.gpu")}
                className="grid grid-cols-2 gap-2 sm:grid-cols-4"
              >
                {GPUS.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    role="radio"
                    aria-checked={gpu === option.id}
                    onClick={() => pickGpu(option.id)}
                    className={cn(
                      "flex flex-col items-start gap-0.5 rounded-md border px-3 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                      gpu === option.id
                        ? "border-foreground/60 bg-accent/40"
                        : "border-border/70 hover:border-border hover:bg-accent/20",
                    )}
                  >
                    <span className="font-mono text-sm font-medium text-foreground">
                      {option.id}
                    </span>
                    <span className="text-ui-11 text-muted-foreground">
                      {t(option.hint)}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            <div className="flex flex-col gap-1">
              <label
                htmlFor={`${id}-name`}
                className="text-ui-11 font-medium text-muted-foreground"
              >
                {t("settings.apiKeys.linkedInstances.name")}
              </label>
              <Input
                id={`${id}-name`}
                value={name}
                onChange={(e) => {
                  setName(e.target.value);
                  setNameEdited(true);
                }}
                autoComplete="off"
                spellCheck={false}
                className="h-8 font-mono text-xs"
              />
            </div>

            <p
              className={cn(
                "text-ui-11 leading-snug",
                error ? "text-destructive" : "text-muted-foreground",
              )}
              role={error ? "alert" : undefined}
            >
              {error ?? t("settings.apiKeys.linkedInstances.colab.billingNote")}
            </p>
          </form>
        ) : (
          <div className="flex min-w-0 flex-col gap-4">
            <div className="flex items-baseline justify-between gap-3">
              <span className="font-mono text-sm font-medium text-foreground">
                @{job.name}
              </span>
              <span className="text-ui-11 text-muted-foreground">
                {t("settings.apiKeys.linkedInstances.colab.badge", {
                  gpu: job.gpu,
                })}
              </span>
            </div>
            <StageList job={job} />
            {job.state === "failed" ? (
              <div className="flex flex-col gap-3 rounded-md border border-destructive/40 bg-destructive/5 p-3">
                <p className="text-xs text-destructive" role="alert">
                  {job.error ??
                    t("settings.apiKeys.linkedInstances.colab.failed")}
                </p>
                <SetupCommands
                  commands={job.setup}
                  wsl={wsl}
                  distro={capability?.distro ?? null}
                />
              </div>
            ) : job.state === "cancelled" ? (
              <p className="text-xs text-muted-foreground">
                {t("settings.apiKeys.linkedInstances.colab.cancelled")}
              </p>
            ) : job.state === "ready" ? (
              <p className="text-xs text-foreground">
                {t("settings.apiKeys.linkedInstances.colab.ready", {
                  name: job.name,
                })}
              </p>
            ) : null}
            {job.log.length > 0 ? (
              <pre
                ref={logRef}
                aria-label={t("settings.apiKeys.linkedInstances.colab.log")}
                className="max-h-48 overflow-auto rounded-md border border-border/60 bg-muted/30 p-3 font-mono text-ui-11 leading-relaxed whitespace-pre-wrap break-all text-muted-foreground"
              >
                {job.log.join("\n")}
              </pre>
            ) : null}
            {error ? (
              <p className="text-ui-11 text-destructive" role="alert">
                {error}
              </p>
            ) : null}
          </div>
        )}

        <DialogFooter>
          {showForm || !job ? (
            <>
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
              >
                {t("common.cancel")}
              </Button>
              <Button
                type="submit"
                form={`${id}-form`}
                disabled={
                  !capability?.ready || submitting || name.trim() === ""
                }
              >
                {submitting
                  ? t("settings.apiKeys.linkedInstances.colab.starting")
                  : t("settings.apiKeys.linkedInstances.colab.submit")}
              </Button>
            </>
          ) : running ? (
            <>
              <Button
                type="button"
                variant="outline"
                className="text-destructive hover:text-destructive"
                onClick={() => void cancel()}
                disabled={cancelling}
              >
                {cancelling
                  ? t("settings.apiKeys.linkedInstances.colab.cancelling")
                  : t("settings.apiKeys.linkedInstances.colab.cancelLaunch")}
              </Button>
              <Button type="button" onClick={() => onOpenChange(false)}>
                {t("settings.apiKeys.linkedInstances.colab.close")}
              </Button>
            </>
          ) : (
            <>
              {job.state !== "ready" ? (
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    setShowForm(true);
                    setError(null);
                    void check();
                  }}
                >
                  {t("settings.apiKeys.linkedInstances.colab.tryAgain")}
                </Button>
              ) : null}
              <Button type="button" onClick={() => onOpenChange(false)}>
                {t("settings.apiKeys.linkedInstances.colab.close")}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
