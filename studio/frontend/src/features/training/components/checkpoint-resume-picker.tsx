// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Spinner } from "@/components/ui/spinner";
import { FolderBrowser } from "@/features/model-picker";
import { loadCheckpointLocation } from "@/features/settings/api/checkpoint-location";
import { useEffect, useId, useState } from "react";
import { inspectCheckpoint } from "../api/train-api";
import type { CheckpointInspection } from "../types/api";

type Source = "none" | "latest" | "browse" | "mounted";

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
  const [path, setPath] = useState("");
  const [browserOpen, setBrowserOpen] = useState(false);
  const [inspection, setInspection] = useState<CheckpointInspection | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void loadCheckpointLocation().then((value) => setConfiguredPath(value.path)).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (source === "latest" && configuredPath && !inspection && !loading) void inspect(configuredPath);
    // `inspect` deliberately stays event-like; re-run only when the setting arrives.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [configuredPath, source]);

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
    if (next === "latest" && configuredPath) void inspect(configuredPath);
    if (next === "browse") setBrowserOpen(true);
  }

  function setConfirmation(value: boolean) {
    setConfirmed(value);
    onInspectionChange(inspection, value, true);
  }

  const blocked = inspection && (
    !inspection.optimizerComplete || !inspection.schedulerComplete ||
    !inspection.trainerStateComplete || inspection.incompatibilities.length > 0 ||
    inspection.missingDatasets.length > 0
  );

  return (
    <div className="space-y-3 rounded-lg border border-border/60 p-3" data-testid="checkpoint-resume-picker">
      <Label className="text-sm font-medium">Resume from checkpoint</Label>
      <RadioGroup value={source} onValueChange={(value) => choose(value as Source)} disabled={disabled}>
        <label className="flex items-start gap-2"><RadioGroupItem value="none" /> <span className="text-sm">Start a new training run</span></label>
        <label className="flex items-start gap-2"><RadioGroupItem value="latest" /> <span className="text-sm">Latest Studio checkpoint<span className="block max-w-72 truncate font-mono text-xs text-muted-foreground" title={configuredPath}>{configuredPath || "Configured checkpoint location"}</span></span></label>
        <label className="flex items-center gap-2"><RadioGroupItem value="browse" /> <span className="text-sm">Browse directory</span></label>
        <label className="flex items-start gap-2"><RadioGroupItem value="mounted" /> <span className="text-sm">Mounted-storage path<span className="block text-xs text-muted-foreground">Use paths such as /content/drive/MyDrive/… on Colab or /kaggle/working/… on Kaggle.</span></span></label>
      </RadioGroup>

      {source === "mounted" && <form className="flex gap-2" onSubmit={(event) => { event.preventDefault(); void inspect(path); }}>
        <Input aria-label="Absolute mounted-storage path" value={path} onChange={(event) => setPath(event.target.value)} placeholder="/content/drive/MyDrive/checkpoints" />
        <Button type="submit" variant="outline" disabled={loading}>Inspect</Button>
      </form>}
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
        {blocked && <p className="text-amber-600">Resolve incomplete state, incompatibilities, and missing datasets before starting.</p>}
        {inspection.external && <label htmlFor={`${id}-confirm`} className="flex items-start gap-2 border-t pt-2 font-medium"><Checkbox id={`${id}-confirm`} checked={confirmed} onCheckedChange={(value) => setConfirmation(value === true)} />I understand this checkpoint was imported from outside Studio and want to resume it.</label>}
      </div>}

      <FolderBrowser open={browserOpen} onOpenChange={setBrowserOpen} initialPath={configuredPath || undefined} title="Select checkpoint directory" confirmLabel="Inspect this directory" showModelHints={false} onSelect={(selected) => { setPath(selected); void inspect(selected); }} />
    </div>
  );
}
