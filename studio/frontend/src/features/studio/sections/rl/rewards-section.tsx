// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  type RewardRecord,
  deleteReward,
  exportReward,
  importReward,
  listRewards,
  previewRewards,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { type ReactElement, useCallback, useEffect, useState } from "react";
import { useShallow } from "zustand/react/shallow";

const SAMPLE_COMPLETION =
  "<reasoning>\n48 + 24 = 72\n</reasoning>\n<answer>\n72\n</answer>";

function ImportRewardDialog({
  open,
  onOpenChange,
  onImported,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImported: (record: RewardRecord) => void;
}): ReactElement {
  const t = useT();
  const [markdown, setMarkdown] = useState("");
  const [overwrite, setOverwrite] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!markdown.trim()) {
      setError(t("rl.rewards.importEmpty"));
      return;
    }
    try {
      onImported(await importReward(markdown, overwrite));
      setMarkdown("");
      setError(null);
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t("rl.rewards.importTitle")}</DialogTitle>
          <DialogDescription>
            {t("rl.rewards.importDescription")}
          </DialogDescription>
        </DialogHeader>
        <Textarea
          rows={10}
          className="font-mono text-xs"
          value={markdown}
          placeholder={
            "---\nname: my-reward\nkind: rule\n---\ntype: regex\n..."
          }
          onChange={(e) => {
            setMarkdown(e.target.value);
            setError(null);
          }}
        />
        <div className="flex items-center gap-2">
          <Checkbox
            id="reward-import-overwrite"
            checked={overwrite}
            onCheckedChange={(v) => setOverwrite(v === true)}
          />
          <Label
            htmlFor="reward-import-overwrite"
            className="text-xs font-normal text-muted-foreground"
          >
            {t("rl.rewards.overwrite")}
          </Label>
        </div>
        {error && <p className="text-ui-11p5 text-destructive">{error}</p>}
        <DialogFooter>
          <Button type="button" onClick={submit}>
            {t("rl.rewards.importAction")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function RewardsSection(): ReactElement {
  const t = useT();
  const { rewards, setRewards } = useTrainingConfigStore(
    useShallow((s) => ({
      rewards: s.grpoRewards,
      setRewards: s.setGrpoRewards,
    })),
  );
  const [library, setLibrary] = useState<RewardRecord[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [importOpen, setImportOpen] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [completion, setCompletion] = useState(SAMPLE_COMPLETION);
  const [reference, setReference] = useState("72");
  const [preview, setPreview] = useState<Record<string, number> | null>(null);
  const [previewTotal, setPreviewTotal] = useState<number | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setLibrary(await listRewards());
      setLoadError(null);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Scores belong to the reward set they were computed for.
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset on any change to the selection
  useEffect(() => {
    setPreview(null);
    setPreviewTotal(null);
  }, [rewards]);

  const active = library.filter((r) => !r.shadowed);
  const byName = new Map(active.map((r) => [r.name, r]));
  const selectedNames = new Set(rewards.map((r) => r.name));
  const addable = active.filter(
    (r) => r.valid && r.kind === "rule" && !selectedNames.has(r.name),
  );

  const setWeight = (name: string, weight: number) =>
    setRewards(rewards.map((r) => (r.name === name ? { ...r, weight } : r)));

  const runPreview = async () => {
    if (rewards.length === 0) {
      return;
    }
    try {
      const res = await previewRewards(rewards, completion, reference || null);
      setPreview(
        Object.fromEntries(res.scores.map((s) => [s.name, s.weighted])),
      );
      setPreviewTotal(res.total);
      setPreviewError(null);
    } catch (err) {
      setPreviewError(err instanceof Error ? err.message : String(err));
    }
  };

  const copyReward = async (name: string) => {
    try {
      await navigator.clipboard.writeText(await exportReward(name));
      setCopied(name);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err));
    }
  };

  const removeFromLibrary = async (name: string) => {
    try {
      await deleteReward(name);
      setRewards(rewards.filter((r) => r.name !== name));
      await refresh();
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {loadError && (
        <p className="text-ui-11p5 text-destructive">
          {t("rl.rewards.loadError", { error: loadError })}
        </p>
      )}

      <div className="flex flex-col">
        {rewards.length === 0 && (
          <p className="text-ui-11p5 text-destructive">
            {t("rl.rewards.empty")}
          </p>
        )}
        {rewards.map((selection) => {
          const record = byName.get(selection.name);
          return (
            <div
              key={selection.name}
              className="grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-3 border-b border-border/60 py-2.5"
            >
              <div className="min-w-0">
                <p className="flex items-center gap-2 text-xs font-medium text-foreground">
                  <span className="truncate">{selection.name}</span>
                  <span className="rounded-full bg-muted px-1.5 text-ui-10 text-muted-foreground">
                    {record?.source === "user"
                      ? t("rl.rewards.user")
                      : t("rl.rewards.bundled")}
                  </span>
                  {preview?.[selection.name] !== undefined && (
                    <span className="font-mono text-ui-10 text-primary">
                      {preview[selection.name] > 0 ? "+" : ""}
                      {preview[selection.name].toFixed(2)}
                    </span>
                  )}
                </p>
                <p className="truncate text-ui-11p5 text-muted-foreground/85">
                  {record?.description ?? t("rl.rewards.invalid")}
                </p>
              </div>
              <Input
                type="number"
                step={0.5}
                min={-10}
                max={10}
                aria-label={t("rl.rewards.weight")}
                className="h-8 w-[calc(76px*var(--ui-space-scale,1))] text-right text-xs"
                value={selection.weight}
                onChange={(e) => {
                  const n = Number(e.target.value);
                  if (Number.isFinite(n)) {
                    setWeight(selection.name, Math.max(-10, Math.min(10, n)));
                  }
                }}
              />
              <div className="flex gap-1">
                <Button
                  type="button"
                  size="xs"
                  variant="ghost"
                  onClick={() => copyReward(selection.name)}
                >
                  {copied === selection.name
                    ? t("rl.rewards.copied")
                    : t("rl.rewards.export")}
                </Button>
                {record?.source === "user" && (
                  <Button
                    type="button"
                    size="xs"
                    variant="ghost"
                    onClick={() => removeFromLibrary(selection.name)}
                  >
                    {t("rl.rewards.delete")}
                  </Button>
                )}
                <Button
                  type="button"
                  size="xs"
                  variant="ghost"
                  aria-label={t("rl.rewards.remove", { name: selection.name })}
                  onClick={() =>
                    setRewards(rewards.filter((r) => r.name !== selection.name))
                  }
                >
                  ×
                </Button>
              </div>
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Select
          value=""
          onValueChange={(name) =>
            setRewards([...rewards, { name, weight: 1 }].slice(0, 16))
          }
          disabled={addable.length === 0}
        >
          <SelectTrigger
            size="sm"
            className="w-[calc(220px*var(--ui-space-scale,1))]"
          >
            <SelectValue placeholder={t("rl.rewards.add")} />
          </SelectTrigger>
          <SelectContent>
            {addable.map((r) => (
              <SelectItem key={r.name} value={r.name}>
                {r.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={() => setImportOpen(true)}
        >
          {t("rl.rewards.import")}
        </Button>
        <span className="text-ui-11p5 text-muted-foreground/85">
          {t("rl.rewards.pythonLocked")}
        </span>
      </div>

      <div className="flex flex-col gap-2 rounded-xl bg-muted/40 p-3">
        <p className="text-xs font-medium text-foreground">
          {t("rl.rewards.tryTitle")}
        </p>
        <Textarea
          rows={4}
          aria-label={t("rl.rewards.tryCompletion")}
          className="font-mono text-xs"
          value={completion}
          onChange={(e) => setCompletion(e.target.value)}
        />
        <div className="flex items-center gap-2">
          <Input
            aria-label={t("rl.rewards.tryReference")}
            placeholder={t("rl.rewards.tryReference")}
            className="h-8 max-w-[calc(220px*var(--ui-space-scale,1))] text-xs"
            value={reference}
            onChange={(e) => setReference(e.target.value)}
          />
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={rewards.length === 0}
            onClick={runPreview}
          >
            {t("rl.rewards.tryRun")}
          </Button>
          {previewTotal !== null && (
            <span className="text-xs font-medium text-foreground">
              {t("rl.rewards.tryTotal", { total: previewTotal.toFixed(2) })}
            </span>
          )}
        </div>
        {previewError && (
          <p className="text-ui-11p5 text-destructive">{previewError}</p>
        )}
      </div>

      <ImportRewardDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        onImported={(record) => {
          refresh();
          if (!selectedNames.has(record.name)) {
            setRewards(
              [...rewards, { name: record.name, weight: 1 }].slice(0, 16),
            );
          }
        }}
      />
    </div>
  );
}
