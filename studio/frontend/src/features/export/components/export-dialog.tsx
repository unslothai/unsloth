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
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { AlertCircleIcon, ArrowRight01Icon, CheckmarkCircle02Icon, Key01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { streamExportLogs, type ExportLogEntry } from "../api/export-api";
import { collapseAnim } from "../anim";
import { EXPORT_METHODS, type ExportMethod } from "../constants";

// Max log lines kept in the dialog's local state. Matches the backend
// ring buffer's maxlen so the UI shows the full scrollback captured
// server side.
const MAX_LOG_LINES = 4000;

interface UseExportLogsResult {
  lines: ExportLogEntry[];
  connected: boolean;
  error: string | null;
}

/**
 * Subscribe to the live export log SSE stream while `exporting` is
 * true, and accumulate lines in local state. Lines from a previous
 * action are cleared:
 *
 *   - when a new export starts (`exporting` flips to true), and
 *   - when the user switches export method, dialog opens fresh, or
 *     the dialog closes — so re-opening into a different action's
 *     screen doesn't show the prior screen's saved output.
 */
function useExportLogs(
  exporting: boolean,
  exportMethod: ExportMethod | null,
  open: boolean,
): UseExportLogsResult {
  const [lines, setLines] = useState<ExportLogEntry[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset log state whenever the user moves to a different screen --
  // either by switching export method or by reopening the dialog -- so
  // each (open × method) tuple shows only its own run history. The
  // streaming effect below additionally clears on new export start.
  useEffect(() => {
    setLines([]);
    setError(null);
    setConnected(false);
  }, [exportMethod, open]);

  useEffect(() => {
    if (!exporting) return;

    setLines([]);
    setError(null);

    const abortCtrl = new AbortController();
    let cancelled = false;

    streamExportLogs({
      signal: abortCtrl.signal,
      onOpen: () => {
        if (!cancelled) setConnected(true);
      },
      onEvent: (event) => {
        if (cancelled) return;
        if (event.event === "log" && event.entry) {
          const entry = event.entry;
          setLines((prev) => {
            const next = prev.length >= MAX_LOG_LINES
              ? prev.slice(prev.length - MAX_LOG_LINES + 1)
              : prev.slice();
            next.push(entry);
            return next;
          });
        } else if (event.event === "error" && event.error) {
          setError(event.error);
        }
      },
    }).catch((err: unknown) => {
      if (cancelled) return;
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof Error ? err.message : String(err));
    }).finally(() => {
      if (!cancelled) setConnected(false);
    });

    return () => {
      cancelled = true;
      abortCtrl.abort();
      setConnected(false);
    };
  }, [exporting]);

  return { lines, connected, error };
}

/**
 * Tick every second while `exporting` is true and report elapsed
 * seconds. Powers the "Working… 27s" badge in the log header so the
 * panel doesn't look frozen during long single-step phases (cache
 * file copy, GGUF conversion) when no new lines are arriving.
 */
function useElapsedSeconds(exporting: boolean): number {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!exporting) {
      setElapsed(0);
      return;
    }
    const startedAt = Date.now();
    setElapsed(0);
    const id = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => window.clearInterval(id);
  }, [exporting]);
  return elapsed;
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s.toString().padStart(2, "0")}s`;
}

function formatLogLine(entry: ExportLogEntry): string {
  // Strip trailing carriage returns that tqdm-style progress leaves
  // in the stream so the scrollback doesn't render funky boxes.
  return entry.line.replace(/\r+$/g, "");
}

type Destination = "local" | "hub";

interface ExportDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  checkpoint: string | null;
  exportMethod: ExportMethod | null;
  quantLevels: string[];
  estimatedSize: string;
  baseModelName: string;
  isAdapter: boolean;
  destination: Destination;
  onDestinationChange: (v: Destination) => void;
  hfUsername: string;
  onHfUsernameChange: (v: string) => void;
  modelName: string;
  onModelNameChange: (v: string) => void;
  hfToken: string;
  onHfTokenChange: (v: string) => void;
  privateRepo: boolean;
  onPrivateRepoChange: (v: boolean) => void;
  onExport: () => void;
  exporting: boolean;
  exportError: string | null;
  exportSuccess: boolean;
  /**
   * Resolved on-disk realpath of the most recent successful export.
   * Surfaced on the Export Complete screen so users can find their
   * model. Null when the export only pushed to the Hub.
   */
  exportOutputPath: string | null;
}

export function ExportDialog({
  open,
  onOpenChange,
  checkpoint,
  exportMethod,
  quantLevels,
  estimatedSize: _estimatedSize,
  baseModelName,
  isAdapter,
  destination,
  onDestinationChange,
  hfUsername,
  onHfUsernameChange,
  modelName,
  onModelNameChange,
  hfToken,
  onHfTokenChange,
  privateRepo,
  onPrivateRepoChange,
  onExport,
  exporting,
  exportError,
  exportSuccess,
  exportOutputPath,
}: ExportDialogProps) {
  // Live log capture is only meaningful for export methods that run
  // a slow subprocess operation with interesting stdout: merged and
  // gguf. LoRA adapter export is a fast disk write and would just
  // show a blank panel, so we hide it there.
  const showLogPanel =
    exportMethod === "merged" || exportMethod === "gguf";

  const { lines: logLines, connected: logConnected, error: logError } =
    useExportLogs(exporting && showLogPanel, exportMethod, open);
  const elapsedSeconds = useElapsedSeconds(exporting && showLogPanel);

  const logScrollRef = useRef<HTMLDivElement | null>(null);
  // Auto-scroll to bottom whenever a new line arrives, unless the
  // user has scrolled up to read earlier output.
  const [followTail, setFollowTail] = useState(true);

  useEffect(() => {
    if (!followTail) return;
    const el = logScrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logLines, followTail]);

  const handleLogScroll = () => {
    const el = logScrollRef.current;
    if (!el) return;
    const nearBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    setFollowTail(nearBottom);
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        if (exporting) return;
        onOpenChange(v);
      }}
    >
      <DialogContent
        className={showLogPanel ? "sm:max-w-2xl" : "sm:max-w-lg"}
        onInteractOutside={(e) => { if (exporting) e.preventDefault(); }}
      >
        {exportSuccess ? (
          <>
            <div className="flex flex-col items-center gap-3 py-6">
              <div className="flex size-12 items-center justify-center rounded-full bg-emerald-500/10">
                <HugeiconsIcon icon={CheckmarkCircle02Icon} className="size-6 text-emerald-500" />
              </div>
              <div className="flex flex-col items-center gap-2 text-center">
                <h3 className="text-lg font-semibold">Export Complete</h3>
                <p className="text-sm text-muted-foreground">
                  {destination === "hub"
                    ? "Model successfully pushed to Hugging Face Hub."
                    : "Model saved locally."}
                </p>
                {exportOutputPath ? (
                  <div className="mt-1 flex w-full max-w-md flex-col items-stretch gap-1 rounded-lg border border-border/40 bg-muted/40 px-3 py-2 text-left">
                    <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                      Saved to
                    </span>
                    <code
                      className="select-all break-all font-mono text-[12px] text-foreground"
                      title={exportOutputPath}
                    >
                      {exportOutputPath}
                    </code>
                  </div>
                ) : null}
              </div>
            </div>
            <DialogFooter>
              <Button onClick={() => onOpenChange(false)}>Done</Button>
            </DialogFooter>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>Export Model</DialogTitle>
              <DialogDescription>
                Choose where to save your exported model.
              </DialogDescription>
            </DialogHeader>

            <div className="flex gap-2">
              <Button
                variant={destination === "local" ? "dark" : "outline"}
                onClick={() => onDestinationChange("local")}
                disabled={exporting}
                className="flex-1"
              >
                Save Locally
              </Button>
              <Button
                variant={destination === "hub" ? "dark" : "outline"}
                onClick={() => onDestinationChange("hub")}
                disabled={exporting}
                className="flex-1"
              >
                Push to Hub
              </Button>
            </div>

            <AnimatePresence>
              {destination === "hub" && (
                <motion.div {...collapseAnim} className="overflow-hidden">
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
                          disabled={exporting}
                        />
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <label className="text-xs font-medium text-muted-foreground">
                          Model Name
                        </label>
                        <Input
                          placeholder="my-model-gguf"
                          value={modelName}
                          onChange={(e) => onModelNameChange(e.target.value)}
                          disabled={exporting}
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
                          <HugeiconsIcon
                            icon={ArrowRight01Icon}
                            className="size-3"
                          />
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
                          disabled={exporting}
                        />
                      </InputGroup>
                      <p className="text-[11px] text-muted-foreground/70">
                        Leave empty if already logged in via CLI.
                      </p>
                    </div>

                    <div className="flex items-center gap-3">
                      <Switch
                        id="private-repo"
                        size="sm"
                        checked={privateRepo}
                        onCheckedChange={onPrivateRepoChange}
                        disabled={exporting}
                      />
                      <label
                        htmlFor="private-repo"
                        className="text-xs font-medium cursor-pointer"
                      >
                        Private Repository
                      </label>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Error banner */}
            {exportError && (
              <div className="flex items-start gap-2 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
                <HugeiconsIcon icon={AlertCircleIcon} className="size-4 mt-0.5 shrink-0" />
                <span>{exportError}</span>
              </div>
            )}

            {/* Summary */}
            <div className="rounded-xl bg-muted/50 p-3 text-xs text-muted-foreground flex flex-col gap-1">
              <div className="flex justify-between">
                <span>Base Model</span>
                <span className="font-medium text-foreground">{baseModelName}</span>
              </div>
              <div className="flex justify-between">
                <span>{isAdapter ? "Checkpoint" : "Model"}</span>
                <span className="font-medium text-foreground">{checkpoint}</span>
              </div>
              <div className="flex justify-between">
                <span>Export Method</span>
                <span className="font-medium text-foreground">
                  {EXPORT_METHODS.find((m) => m.value === exportMethod)?.title}
                </span>
              </div>
              {exportMethod === "gguf" && quantLevels.length > 0 && (
                <div className="flex justify-between">
                  <span>Quantizations</span>
                  <span className="font-medium text-foreground">
                    {quantLevels.join(", ")}
                  </span>
                </div>
              )}
              {/* TODO: unhide once estimated size comes from the backend API */}
              {/* <div className="flex justify-between">
            <span>Est. size</span>
            <span className="font-medium text-foreground">{estimatedSize}</span>
          </div> */}
            </div>

            {/* Live export output panel */}
            <AnimatePresence>
              {showLogPanel && (exporting || logLines.length > 0) && (
                <motion.div {...collapseAnim} className="overflow-hidden">
                  <div className="flex flex-col gap-1.5 pt-1">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-medium text-muted-foreground">
                        Export output
                      </label>
                      <div className="flex items-center gap-2 text-[11px] text-muted-foreground/80">
                        <span
                          className={
                            logConnected
                              ? "inline-block size-1.5 rounded-full bg-emerald-500"
                              : "inline-block size-1.5 rounded-full bg-muted-foreground/40"
                          }
                        />
                        <span>
                          {logConnected
                            ? "streaming"
                            : exporting
                              ? "connecting..."
                              : "idle"}
                        </span>
                        {exporting && elapsedSeconds > 0 ? (
                          <span className="tabular-nums text-muted-foreground/70">
                            · {formatElapsed(elapsedSeconds)}
                          </span>
                        ) : null}
                      </div>
                    </div>
                    <div
                      ref={logScrollRef}
                      onScroll={handleLogScroll}
                      className="h-56 w-full overflow-auto rounded-lg border border-border/40 bg-black/85 p-3 font-mono text-[11px] leading-[1.45] text-emerald-200/90"
                    >
                      {logLines.length === 0 ? (
                        <div className="flex h-full items-center justify-center text-muted-foreground/70">
                          <span className="flex items-center gap-2">
                            <Spinner className="size-3" />
                            Waiting for worker output...
                          </span>
                        </div>
                      ) : (
                        <pre className="whitespace-pre-wrap break-words">
                          {logLines.map((entry, idx) => (
                            <div
                              key={idx}
                              className={
                                entry.stream === "stderr"
                                  ? "text-rose-300/90"
                                  : entry.stream === "status"
                                    ? "text-sky-300/90"
                                    : ""
                              }
                            >
                              {formatLogLine(entry)}
                            </div>
                          ))}
                        </pre>
                      )}
                    </div>
                    {logError && (
                      <p className="text-[11px] text-destructive/80">
                        Log stream: {logError}
                      </p>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => onOpenChange(false)}
                disabled={exporting}
              >
                Cancel
              </Button>
              <Button onClick={onExport} disabled={exporting}>
                {exporting ? (
                  <span className="flex items-center gap-2">
                    <Spinner className="size-4" />
                    Exporting…
                  </span>
                ) : (
                  "Start Export"
                )}
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
