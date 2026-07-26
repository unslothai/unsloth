// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { InfoHint } from "@/components/ui/info-hint";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { FolderBrowser } from "@/features/model-picker";
import { loadCheckpointLocation } from "@/features/settings/api/checkpoint-location";
import { useEffect, useId, useState } from "react";
import { inspectCheckpoint } from "../api/train-api";
import type { CheckpointInspection } from "../types/api";

type Source = "none" | "browse";

export interface CheckpointResumePickerProps {
  disabled?: boolean;
  onInspectionChange: (inspection: CheckpointInspection | null, confirmed: boolean, resumeSelected: boolean) => void;
}

export function isAbsoluteFilesystemPath(path: string): boolean {
  const value = path.trim();
  return (/^\//.test(value) || /^[A-Za-z]:[\\/]/.test(value) || /^\\\\/.test(value)) &&
    !/^[a-z][a-z\d+.-]*:\/\//i.test(value);
}

function Status({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return <li className={ok ? "text-emerald-600" : "text-amber-600"}>{ok ? "✓" : "—"} {children}</li>;
}

export function CheckpointResumePicker({ disabled, onInspectionChange }: CheckpointResumePickerProps) {
  const id = useId();
  const [source, setSource] = useState<Source>("none");
  const [configuredPath, setConfiguredPath] = useState("");
  const [selectedPath, setSelectedPath] = useState("");
  const [browserOpen, setBrowserOpen] = useState(false);
  const [inspection, setInspection] = useState<CheckpointInspection | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void loadCheckpointLocation().then((value) => setConfiguredPath(value.path)).catch(() => undefined);
  }, []);

  async function inspect(selectedPath: string) {
    setInspection(null);
    setConfirmed(false);
    onInspectionChange(null, false, source !== "none");
    if (!isAbsoluteFilesystemPath(selectedPath)) {
      setError("Enter an absolute mounted filesystem path, not a web URL.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await inspectCheckpoint(selectedPath.trim());
      setInspection(result);
      onInspectionChange(result, !result.external, true);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Checkpoint inspection failed");
    } finally {
      setLoading(false);
    }
  }

  function choose(next: Source) {
    setSource(next);
    setInspection(null);
    setConfirmed(false);
    setError(null);
    onInspectionChange(null, false, next !== "none");
  }

  function setConfirmation(value: boolean) {
    setConfirmed(value);
    onInspectionChange(inspection, value, true);
  }

  const blocked = inspection && (
    !inspection.optimizerComplete || !inspection.schedulerComplete ||
    !inspection.trainerStateComplete || !inspection.bundledConfigurationFound ||
    inspection.incompatibilities.length > 0 ||
    inspection.missingDatasets.length > 0
  );

  return (
    <div className="space-y-3 rounded-xl border border-border/60 bg-muted/10 p-3" data-testid="checkpoint-resume-picker">
      <div className="flex items-center gap-1.5">
        <Label className="text-sm font-medium">How should training start?</Label>
        <InfoHint>
          Resume restores model, optimizer, scheduler, and step state from a complete checkpoint. Use New training when you do not need previous trainer state.
        </InfoHint>
      </div>

      <div className="grid grid-cols-2 gap-1 rounded-lg bg-muted/60 p-1" role="group" aria-label="Training start mode">
        <Button
          type="button"
          size="sm"
          variant={source === "none" ? "secondary" : "ghost"}
          className={source === "none" ? "bg-background shadow-sm hover:bg-background" : "text-muted-foreground"}
          aria-pressed={source === "none"}
          disabled={disabled}
          onClick={() => choose("none")}
        >
          New training
        </Button>
        <Button
          type="button"
          size="sm"
          variant={source !== "none" ? "secondary" : "ghost"}
          className={source !== "none" ? "bg-background shadow-sm hover:bg-background" : "text-muted-foreground"}
          aria-pressed={source !== "none"}
          disabled={disabled}
          onClick={() => choose("browse")}
        >
          Resume checkpoint
        </Button>
      </div>

      {source === "browse" && <div className="space-y-2 animate-in fade-in-0 slide-in-from-top-1 duration-150">
        <p className="text-xs leading-relaxed text-muted-foreground">
          Browse any local or mounted folder, including Google Drive on Colab and persistent Kaggle storage. Studio will find the newest complete checkpoint inside it.
        </p>
        <Button type="button" variant="outline" className="w-full" disabled={disabled || loading} onClick={() => setBrowserOpen(true)}>
          Browse checkpoint folder
        </Button>
        {selectedPath && <p className="truncate rounded-md bg-muted/50 px-2.5 py-2 font-mono text-[11px] text-muted-foreground" title={selectedPath}>{selectedPath}</p>}
      </div>}
      {loading && <div className="flex items-center gap-2 text-xs text-muted-foreground"><Spinner className="size-3" />Inspecting checkpoint…</div>}
      {error && <p role="alert" className="text-xs text-destructive">{error}</p>}

      {inspection && <div className="space-y-2 rounded-md bg-muted/40 p-3 text-xs" data-testid="checkpoint-inspection">
        <div className="font-medium">{inspection.checkpointName} · global step {inspection.globalStep.toLocaleString()}</div>
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
          <dt className="text-muted-foreground">Model</dt><dd>{inspection.modelIdentity ?? "Not detected"}</dd>
          <dt className="text-muted-foreground">Adapter</dt><dd>{inspection.adapterIdentity ?? "None detected"}</dd>
          <dt className="text-muted-foreground">Backend</dt><dd>{inspection.trainingBackend ?? "Not detected"}</dd>
        </dl>
        <ul>
          <Status ok={inspection.optimizerComplete}>Optimizer state complete</Status>
          <Status ok={inspection.schedulerComplete}>Scheduler state complete</Status>
          <Status ok={inspection.trainerStateComplete}>Trainer state complete</Status>
          <Status ok={inspection.bundledConfigurationFound}>{inspection.bundledConfigurationFound ? "Bundled training configuration found" : "Bundled training configuration not found"}</Status>
        </ul>
        {inspection.incompatibilities.length > 0 && <div className="text-destructive"><strong>Configuration incompatibilities:</strong><ul className="list-disc pl-4">{inspection.incompatibilities.map((item) => <li key={item}>{item}</li>)}</ul></div>}
        {inspection.missingDatasets.length > 0 && <div className="text-destructive"><strong>Missing datasets:</strong><ul className="list-disc pl-4">{inspection.missingDatasets.map((item) => <li key={item}>{item}</li>)}</ul></div>}
        {blocked && <p className="text-amber-600">A complete portable training configuration and dataset state are required before this checkpoint can be resumed.</p>}
        {inspection.external && <label htmlFor={`${id}-confirm`} className="flex items-start gap-2 border-t pt-2 font-medium"><Checkbox id={`${id}-confirm`} checked={confirmed} onCheckedChange={(value) => setConfirmation(value === true)} />I understand this checkpoint was imported from outside Studio and want to resume it.</label>}
      </div>}

      <FolderBrowser open={browserOpen} onOpenChange={setBrowserOpen} initialPath={configuredPath || undefined} title="Select checkpoint directory" confirmLabel="Inspect this directory" showModelHints={false} onSelect={(selected) => { setSelectedPath(selected); void inspect(selected); }} />
    </div>
  );
}
