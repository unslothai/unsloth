// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { FolderBrowser } from "@/components/assistant-ui/model-selector/folder-browser";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  AlertCircleIcon,
  ArrowRight01Icon,
  CancelCircleIcon,
  CheckmarkCircle02Icon,
  FolderSearchIcon,
  Key01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { EXPORT_METHODS, type ExportMethod } from "../constants";
import type { ExportLogEntry } from "../api/export-api";
import { getExportLogLineClass } from "../lib/log-style";
import {
  selectExportProgressPercent,
  useExportRuntimeStore,
  type ExportDestination,
} from "../stores/export-runtime-store";

function useElapsedSeconds(startedAt: number | null, running: boolean): number {
  // `now` is only advanced by the interval (never set synchronously in the
  // effect body), so the elapsed value is derived during render. When `running`
  // flips false the interval stops and `now` freezes, holding the final time.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running || startedAt == null) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [running, startedAt]);
  if (startedAt == null) return 0;
  return Math.max(0, Math.floor((now - startedAt) / 1000));
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s.toString().padStart(2, "0")}s`;
}

function formatLogLine(entry: ExportLogEntry): string {
  return entry.line.replace(/\r+$/g, "");
}

function isAbsoluteFolderPath(path: string): boolean {
  return (
    path.startsWith("/") ||
    /^[A-Za-z]:([\\/]|$)/.test(path) ||
    /^\\\\/.test(path)
  );
}

const PHASE_LABELS: Record<string, string> = {
  idle: "Ready",
  loading: "Loading model",
  exporting: "Exporting",
  success: "Complete",
  error: "Failed",
  canceled: "Canceled",
};

// Shown in the terminal before the first worker line arrives (it can lag a few
// seconds behind, especially over a tunnel). Reflects the phase so the panel
// never looks stuck while the progress bar is already advancing.
function waitingMessage(phase: string, stage: string | null): string {
  if (stage) return stage;
  if (phase === "loading") return "Loading model into the export worker...";
  if (phase === "exporting") return "Preparing export...";
  return "Starting...";
}

// Preset shard sizes offered in the dropdown. unsloth only accepts an integer
// followed by "GB"/"MB" (unsloth/save.py:_resolve_gguf_shard_size), so every
// preset is a pre-validated string. "__custom__" reveals a free-text field.
const SHARD_SIZE_PRESETS = [
  "128MB",
  "256MB",
  "512MB",
  "1GB",
  "2GB",
  "4GB",
  "8GB",
  "16GB",
] as const;
const SHARD_SIZE_CUSTOM = "__custom__";
const SHARD_SIZE_DEFAULT = "2GB";

export interface ExportRunPanelProps {
  exportMethod: ExportMethod | null;
  quantLevels: string[];
  checkpoint: string | null;
  baseModelName: string;
  isAdapter: boolean;
  destination: ExportDestination;
  onDestinationChange: (v: ExportDestination) => void;
  saveDirectory: string;
  defaultSaveDirectory: string;
  saveDirectoryOverridden: boolean;
  onSaveDirectoryChange: (v: string | null) => void;
  hfUsername: string;
  onHfUsernameChange: (v: string) => void;
  modelName: string;
  onModelNameChange: (v: string) => void;
  hfToken: string;
  onHfTokenChange: (v: string) => void;
  privateRepo: boolean;
  onPrivateRepoChange: (v: boolean) => void;
  /** Custom GGUF shard size (e.g. "2GB"); blank keeps unsloth's default. */
  ggufShardSize: string;
  onGgufShardSizeChange: (v: string) => void;
  /** Kick off the export (the page assembles params and calls the store). */
  onStart: () => void;
  /** Collapse the panel; only offered before a run or after a terminal one. */
  onClose: () => void;
}

export function ExportRunPanel(props: ExportRunPanelProps) {
  const {
    exportMethod,
    quantLevels,
    checkpoint,
    baseModelName,
    isAdapter,
    destination,
    onDestinationChange,
    saveDirectory,
    defaultSaveDirectory,
    saveDirectoryOverridden,
    onSaveDirectoryChange,
    hfUsername,
    onHfUsernameChange,
    modelName,
    onModelNameChange,
    hfToken,
    onHfTokenChange,
    privateRepo,
    onPrivateRepoChange,
    ggufShardSize,
    onGgufShardSizeChange,
    onStart,
    onClose,
  } = props;

  // View-only state for the GGUF split toggle. The resolved value lives in the
  // page as `ggufShardSize`: "0" = single file (no split), else a size string.
  const [splitMode, setSplitMode] = useState<"single" | "split">(
    ggufShardSize.trim() && ggufShardSize.trim() !== "0" ? "split" : "single",
  );
  // True when the user picked "Custom…" so the free-text size field is shown
  // (also inferred on mount when the size isn't one of the presets).
  const [customSize, setCustomSize] = useState(
    splitMode === "split" &&
      !SHARD_SIZE_PRESETS.includes(
        ggufShardSize.trim() as (typeof SHARD_SIZE_PRESETS)[number],
      ),
  );

  const run = useExportRuntimeStore(
    useShallow((s) => ({
      phase: s.phase,
      isExporting: s.isExporting,
      logLines: s.logLines,
      connected: s.connected,
      reconnecting: s.reconnecting,
      stage: s.stage,
      quantIndex: s.quantIndex,
      quantTotal: s.quantTotal,
      startedAt: s.startedAt,
      result: s.result,
      error: s.error,
      summary: s.summary,
    })),
  );
  const requestCancel = useExportRuntimeStore((s) => s.requestCancel);
  const progress = useExportRuntimeStore(selectExportProgressPercent);

  const isExporting = run.isExporting;
  const isTerminal =
    run.phase === "success" || run.phase === "error" || run.phase === "canceled";
  const showConfig = run.phase === "idle";
  // Gate the log area on the active run's method (from the store) as well as the
  // local form selection, so it stays visible after navigating away and back
  // (the form `exportMethod` resets on remount, but the run does not).
  const panelMethod = run.summary?.method ?? exportMethod;
  const showLogPanel =
    isExporting ||
    run.logLines.length > 0 ||
    panelMethod === "merged" ||
    panelMethod === "gguf" ||
    panelMethod === "lora";

  const elapsedSeconds = useElapsedSeconds(run.startedAt, isExporting);

  const [folderBrowserOpen, setFolderBrowserOpen] = useState(false);
  const logScrollRef = useRef<HTMLDivElement | null>(null);
  const [followTail, setFollowTail] = useState(true);

  useEffect(() => {
    if (!followTail) return;
    const el = logScrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [run.logLines, followTail]);

  const handleLogScroll = () => {
    const el = logScrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    setFollowTail(nearBottom);
  };

  const methodTitle = EXPORT_METHODS.find((m) => m.value === exportMethod)?.title;
  const summary = run.summary;
  const summaryBaseModel = summary?.baseModelName ?? baseModelName;
  const summaryCheckpoint = summary?.checkpointLabel ?? checkpoint;
  const summaryMethodLabel = summary?.methodLabel ?? methodTitle;
  const summaryQuants = summary?.quantLevels ?? quantLevels;
  const summaryMethod = summary?.method ?? exportMethod;
  const showProgress = isExporting || isTerminal;

  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-border/50 bg-muted/20 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          <h3 className="text-sm font-semibold">
            {showConfig ? "Export Model" : "Export"}
          </h3>
          <p className="text-xs text-muted-foreground">
            {showConfig
              ? "Choose where to save your exported model."
              : "Runs in the background. You can keep training and chatting while it exports."}
          </p>
        </div>
        {(showConfig || isTerminal) && (
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label="Close export panel"
            onClick={onClose}
          >
            <HugeiconsIcon icon={CancelCircleIcon} className="size-4" />
          </Button>
        )}
      </div>

      {/* Destination configuration (only before a run starts) */}
      {showConfig && (
        <>
          <div className="flex gap-2">
            <Button
              variant={destination === "local" ? "dark" : "outline"}
              onClick={() => onDestinationChange("local")}
              className="flex-1"
            >
              Save Locally
            </Button>
            <Button
              variant={destination === "hub" ? "dark" : "outline"}
              onClick={() => onDestinationChange("hub")}
              className="flex-1"
            >
              Push to Hub
            </Button>
          </div>

          {destination === "local" && (
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between gap-2">
                <label className="text-xs font-medium text-muted-foreground">
                  Save folder
                </label>
                {saveDirectoryOverridden && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="xs"
                    onClick={() => onSaveDirectoryChange(null)}
                  >
                    Use default
                  </Button>
                )}
              </div>
              <div className="flex items-stretch gap-2">
                <Input
                  className="min-w-0 flex-1 font-mono text-[12px]"
                  value={saveDirectory}
                  onChange={(e) => onSaveDirectoryChange(e.target.value)}
                  spellCheck={false}
                  title={saveDirectory}
                  placeholder={defaultSaveDirectory}
                />
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <Button
                      type="button"
                      variant="outline"
                      size="icon"
                      onClick={() => setFolderBrowserOpen(true)}
                      aria-label="Browse save folder"
                    >
                      <HugeiconsIcon icon={FolderSearchIcon} className="size-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Browse</TooltipContent>
                </Tooltip>
              </div>
              <p className="text-[11px] text-muted-foreground/70">
                {saveDirectory !== defaultSaveDirectory ? (
                  <>Default: {defaultSaveDirectory}</>
                ) : (
                  <>
                    Paste an absolute path if the folder browser cannot reach the
                    drive.
                  </>
                )}
              </p>
            </div>
          )}

          {exportMethod === "gguf" && (
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                GGUF file splitting
              </label>
              <div className="flex gap-2">
                <Button
                  variant={splitMode === "single" ? "dark" : "outline"}
                  onClick={() => {
                    setSplitMode("single");
                    setCustomSize(false);
                    onGgufShardSizeChange("0");
                  }}
                  className="flex-1"
                >
                  Single file
                </Button>
                <Button
                  variant={splitMode === "split" ? "dark" : "outline"}
                  onClick={() => {
                    setSplitMode("split");
                    // Seed a sensible preset when leaving the no-split state.
                    if (!ggufShardSize.trim() || ggufShardSize.trim() === "0") {
                      setCustomSize(false);
                      onGgufShardSizeChange(SHARD_SIZE_DEFAULT);
                    }
                  }}
                  className="flex-1"
                >
                  Split into shards
                </Button>
              </div>
              {splitMode === "split" ? (
                <>
                  <Select
                    value={customSize ? SHARD_SIZE_CUSTOM : ggufShardSize.trim()}
                    onValueChange={(value) => {
                      if (value === SHARD_SIZE_CUSTOM) {
                        setCustomSize(true);
                        return;
                      }
                      setCustomSize(false);
                      onGgufShardSizeChange(value);
                    }}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Choose a shard size" />
                    </SelectTrigger>
                    <SelectContent>
                      {SHARD_SIZE_PRESETS.map((preset) => (
                        <SelectItem key={preset} value={preset}>
                          {preset.replace("GB", " GB").replace("MB", " MB")}
                        </SelectItem>
                      ))}
                      <SelectItem value={SHARD_SIZE_CUSTOM}>Custom…</SelectItem>
                    </SelectContent>
                  </Select>
                  {customSize && (
                    <Input
                      className="font-mono text-[12px]"
                      value={ggufShardSize === "0" ? "" : ggufShardSize}
                      onChange={(e) => onGgufShardSizeChange(e.target.value)}
                      spellCheck={false}
                      placeholder="e.g. 512MB, 6GB"
                    />
                  )}
                  <p className="text-[11px] text-muted-foreground/70">
                    {customSize ? (
                      <>
                        Whole number + <code className="font-mono">GB</code> or{" "}
                        <code className="font-mono">MB</code> (e.g.{" "}
                        <code className="font-mono">512MB</code>). Decimals and bare
                        numbers are not accepted.
                      </>
                    ) : (
                      "Each file will be at most this size. Quantized outputs are always a single file."
                    )}
                  </p>
                </>
              ) : (
                <p className="text-[11px] text-muted-foreground/70">
                  One GGUF file, no splitting regardless of size.
                </p>
              )}
            </div>
          )}

          {destination === "hub" && (
            <div className="flex flex-col gap-4 px-0.5">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-medium text-muted-foreground">
                        Username / Org
                      </label>
                      <Input
                        placeholder="your-username"
                        value={hfUsername}
                        onChange={(e) => onHfUsernameChange(e.target.value)}
                      />
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-medium text-muted-foreground">
                        Model Name
                      </label>
                      <Input
                        placeholder="my-model-GGUF"
                        value={modelName}
                        onChange={(e) => onModelNameChange(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-medium text-muted-foreground">
                        HF Write Token
                      </label>
                      <a
                        href="https://huggingface.co/settings/tokens"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 text-[11px] text-emerald-600 hover:text-emerald-700 transition-colors"
                      >
                        Get token
                        <HugeiconsIcon icon={ArrowRight01Icon} className="size-3" />
                      </a>
                    </div>
                    <InputGroup>
                      <InputGroupAddon>
                        <HugeiconsIcon icon={Key01Icon} className="size-4" />
                      </InputGroupAddon>
                      <InputGroupInput
                        type="password"
                        autoComplete="new-password"
                        name="hf-token"
                        placeholder="hf_..."
                        value={hfToken}
                        onChange={(e) => onHfTokenChange(e.target.value)}
                      />
                    </InputGroup>
                    <p className="text-[11px] text-muted-foreground/70">
                      Leave empty if already logged in via CLI.
                    </p>
                  </div>

                  <div className="flex items-center gap-3">
                    <Switch
                      id="export-private-repo"
                      size="sm"
                      checked={privateRepo}
                      onCheckedChange={onPrivateRepoChange}
                    />
                    <label
                      htmlFor="export-private-repo"
                      className="text-xs font-medium cursor-pointer"
                    >
                      Private Repository
                    </label>
                  </div>
                </div>
          )}
        </>
      )}

      {/* Result banners */}
      {run.phase === "success" && (
        <div className="flex items-start gap-2 rounded-lg bg-emerald-500/10 p-3 text-sm text-emerald-700 dark:text-emerald-300">
          <HugeiconsIcon
            icon={CheckmarkCircle02Icon}
            className="mt-0.5 size-4 shrink-0"
          />
          <div className="flex min-w-0 flex-col gap-1">
            <span>
              {run.result?.destination === "hub"
                ? "Export finished and pushed to Hugging Face Hub."
                : "Export finished successfully."}
            </span>
            {run.result?.outputPath ? (
              <code
                className="select-all break-all font-mono text-[12px] text-foreground/90"
                title={run.result.outputPath}
              >
                {run.result.outputPath}
              </code>
            ) : null}
          </div>
        </div>
      )}

      {run.phase === "canceled" && (
        <div className="flex items-start gap-2 rounded-lg bg-amber-500/10 p-3 text-sm text-amber-700 dark:text-amber-300">
          <HugeiconsIcon icon={CancelCircleIcon} className="mt-0.5 size-4 shrink-0" />
          <span>Export canceled. Training and inference were not affected.</span>
        </div>
      )}

      {run.phase === "error" && run.error && (
        <div className="flex items-start gap-2 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
          <HugeiconsIcon icon={AlertCircleIcon} className="size-4 mt-0.5 shrink-0" />
          <span>{run.error}</span>
        </div>
      )}

      {/* Summary */}
      <div className="rounded-xl bg-muted/50 p-3 text-xs text-muted-foreground flex flex-col gap-1">
        <div className="flex justify-between">
          <span>Base Model</span>
          <span className="font-medium text-foreground">{summaryBaseModel}</span>
        </div>
        <div className="flex justify-between">
          <span>{isAdapter ? "Checkpoint" : "Model"}</span>
          <span className="font-medium text-foreground">{summaryCheckpoint}</span>
        </div>
        <div className="flex justify-between">
          <span>Export Method</span>
          <span className="font-medium text-foreground">{summaryMethodLabel}</span>
        </div>
        {summaryMethod === "gguf" && summaryQuants.length > 0 && (
          <div className="flex justify-between">
            <span>Quantizations</span>
            <span className="font-medium text-foreground">
              {summaryQuants.join(", ")}
            </span>
          </div>
        )}
      </div>

      {/* Progress */}
      {showProgress && (
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full bg-foreground/10 px-2.5 py-1 text-[10px] font-semibold">
              {PHASE_LABELS[run.phase] ?? run.phase}
            </span>
            {summaryMethod === "gguf" && run.quantTotal > 1 && (
              <span className="text-[10px] tabular-nums text-muted-foreground">
                Quant {Math.min(run.quantIndex + (isExporting ? 1 : 0), run.quantTotal)} of {run.quantTotal}
              </span>
            )}
            <span className="rounded-full border border-border/60 px-2.5 py-1 text-[10px] font-medium tabular-nums text-muted-foreground">
              {progress}%
            </span>
            <span className="text-[10px] tabular-nums text-muted-foreground/70">
              {formatElapsed(elapsedSeconds)}
            </span>
          </div>
          <Progress
            value={progress}
            className="h-2 bg-foreground/[0.05]"
            indicatorClassName={
              run.phase === "error"
                ? "bg-destructive"
                : run.phase === "canceled"
                  ? "bg-amber-500"
                  : undefined
            }
          />
          {run.stage && (
            <p className="truncate text-[11px] text-muted-foreground/80" title={run.stage}>
              {run.stage}
            </p>
          )}
        </div>
      )}

      {/* Live export output */}
      {showLogPanel && (isExporting || run.logLines.length > 0) && (
        <div className="flex flex-col gap-1.5 pt-1">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-muted-foreground">
                  Export output
                </label>
                <div className="flex items-center gap-2 text-[11px] text-muted-foreground/80">
                  <span
                    className={
                      run.reconnecting
                        ? "inline-block size-1.5 rounded-full bg-amber-500"
                        : run.connected
                          ? "inline-block size-1.5 rounded-full bg-emerald-500"
                          : "inline-block size-1.5 rounded-full bg-muted-foreground/40"
                    }
                  />
                  <span>
                    {run.reconnecting
                      ? "reconnecting..."
                      : run.connected
                        ? "streaming"
                        : isExporting
                          ? "connecting..."
                          : "idle"}
                  </span>
                </div>
              </div>
              <div
                ref={logScrollRef}
                onScroll={handleLogScroll}
                className="h-56 w-full overflow-auto rounded-lg border border-border/40 bg-black/85 p-3 font-mono text-[11px] leading-[1.45] text-emerald-200/90"
              >
                {run.logLines.length === 0 ? (
                  <div className="flex h-full items-center justify-center text-muted-foreground/70">
                    <span className="flex items-center gap-2">
                      <Spinner className="size-3" />
                      {waitingMessage(run.phase, run.stage)}
                    </span>
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap break-words">
                    {run.logLines.map((entry, idx) => (
                      <div
                        key={idx}
                        className={getExportLogLineClass(entry)}
                      >
                        {formatLogLine(entry)}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
        )}

      {/* Footer actions */}
      <div className="flex justify-end gap-2">
        {showConfig && (
          <>
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={onStart}>Start Export</Button>
          </>
        )}
        {isExporting && (
          <Button variant="destructive" onClick={() => void requestCancel()}>
            Cancel Export
          </Button>
        )}
        {isTerminal && <Button onClick={onClose}>Done</Button>}
      </div>

      <FolderBrowser
        open={folderBrowserOpen}
        onOpenChange={setFolderBrowserOpen}
        initialPath={
          isAbsoluteFolderPath(saveDirectory) ? saveDirectory : undefined
        }
        onSelect={(path) => onSaveDirectoryChange(path)}
      />
    </div>
  );
}
