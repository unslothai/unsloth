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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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
  type RuleType,
  deleteReward,
  exportReward,
  importReward,
  previewCell,
  previewRewards,
  readRewardHead,
  summarizeRule,
  useRlWorkspaceStore,
  useTrainingConfigStore,
} from "@/features/training";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  Calculator01Icon,
  Cancel01Icon,
  CodeIcon,
  Copy01Icon,
  Delete02Icon,
  EqualSignIcon,
  FileImportIcon,
  InformationCircleIcon,
  MoreVerticalIcon,
  RegexIcon,
  RulerIcon,
  TestTube01Icon,
  TextIcon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";

const SAMPLE_COMPLETION =
  "<reasoning>\n48 + 24 = 72\n</reasoning>\n<answer>\n72\n</answer>";

const RULE_ICON: Record<RuleType, IconSvgElement> = {
  regex: RegexIcon,
  exact_match: EqualSignIcon,
  numeric: Calculator01Icon,
  json_schema: CodeIcon,
  length: RulerIcon,
};

function RewardSummaryLine({
  record,
}: {
  record: RewardRecord | undefined;
}): ReactElement {
  const t = useT();
  if (!record) {
    return <span className="text-destructive">{t("rl.rewards.invalid")}</span>;
  }
  const summary = summarizeRule(record.rule);
  if (!summary.type) {
    return <span>{record.description}</span>;
  }
  return (
    <span title={record.description}>
      {t(`rl.rewards.summary.${summary.key}` as TranslationKey, summary.params)}
      <span className="text-muted-foreground/60"> → </span>
      <span className="font-medium text-foreground/80">{summary.score}</span>
      {summary.otherwise && (
        <span>
          {" · "}
          {t("rl.rewards.summary.otherwise", { score: summary.otherwise })}
        </span>
      )}
    </span>
  );
}

function ScoreChip({ value }: { value: number | undefined }): ReactElement {
  if (value === undefined) {
    return <span className="w-14" />;
  }
  return (
    <span
      className={cn(
        "w-14 rounded-md px-1.5 py-0.5 text-center font-mono text-ui-11",
        value > 0
          ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300"
          : value < 0
            ? "bg-rose-100 text-rose-700 dark:bg-rose-950/50 dark:text-rose-300"
            : "bg-muted text-muted-foreground",
      )}
    >
      {value > 0 ? "+" : ""}
      {value.toFixed(2)}
    </span>
  );
}

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
  const fileInput = useRef<HTMLInputElement>(null);
  const head = readRewardHead(markdown);
  const isPython = head?.kind === "python";

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
        <div className="flex items-center gap-2">
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => fileInput.current?.click()}
          >
            <HugeiconsIcon icon={FileImportIcon} className="size-3.5" />
            {t("rl.rewards.importFile")}
          </Button>
          <span className="text-ui-11p5 text-muted-foreground">
            {t("rl.rewards.importOrPaste")}
          </span>
          <input
            ref={fileInput}
            type="file"
            accept=".md,text/markdown,text/plain"
            className="hidden"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (file) {
                setMarkdown(await file.text());
                setError(null);
              }
              e.target.value = "";
            }}
          />
        </div>
        <Textarea
          rows={9}
          className="font-mono text-xs"
          value={markdown}
          placeholder={
            "---\nname: my-reward\nkind: rule\ndescription: ...\n---\ntype: regex\n..."
          }
          onChange={(e) => {
            setMarkdown(e.target.value);
            setError(null);
          }}
        />
        {head?.name && !isPython && (
          <div className="flex items-start gap-2 rounded-lg border border-border/70 px-3 py-2 text-xs">
            <HugeiconsIcon
              icon={TextIcon}
              className="mt-0.5 size-3.5 shrink-0 text-muted-foreground"
            />
            <div className="min-w-0">
              <p className="font-medium text-foreground">
                {head.name}
                <span className="ml-2 rounded-full bg-muted px-1.5 text-ui-10 text-muted-foreground">
                  {head.kind ?? "rule"}
                </span>
              </p>
              {head.description && (
                <p className="text-ui-11p5 text-muted-foreground">
                  {head.description}
                </p>
              )}
            </div>
          </div>
        )}
        {isPython && (
          <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
            <HugeiconsIcon
              icon={InformationCircleIcon}
              className="mt-0.5 size-3.5 shrink-0"
            />
            <p>{t("rl.rewards.importPython")}</p>
          </div>
        )}
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
        {error && !isPython && (
          <p className="text-ui-11p5 text-destructive">{error}</p>
        )}
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
  const { rewards, setRewards, mapping } = useTrainingConfigStore(
    useShallow((s) => ({
      rewards: s.grpoRewards,
      setRewards: s.setGrpoRewards,
      mapping: s.rlRoleMapping,
    })),
  );
  const { library, libraryError, refreshLibrary, previewRow } =
    useRlWorkspaceStore(
      useShallow((s) => ({
        library: s.library,
        libraryError: s.libraryError,
        refreshLibrary: s.refreshLibrary,
        previewRow: s.previewRow,
      })),
    );
  const [actionError, setActionError] = useState<string | null>(null);
  const [importOpen, setImportOpen] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [completion, setCompletion] = useState(SAMPLE_COMPLETION);
  const [referenceEdit, setReferenceEdit] = useState<string | null>(null);
  const [preview, setPreview] = useState<Record<string, number> | null>(null);
  const [previewTotal, setPreviewTotal] = useState<number | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  useEffect(() => {
    refreshLibrary();
  }, [refreshLibrary]);

  const answerColumn = Object.entries(mapping).find(
    ([, role]) => role === "answer",
  )?.[0];
  const rowReference = answerColumn
    ? previewCell(previewRow?.[answerColumn])
    : "";
  const reference = referenceEdit ?? (rowReference || "72");

  const active = library.filter((r) => !r.shadowed);
  const byName = new Map(active.map((r) => [r.name, r]));
  const selectedNames = new Set(rewards.map((r) => r.name));
  const addable = active.filter(
    (r) => r.valid && r.kind === "rule" && !selectedNames.has(r.name),
  );

  const setWeight = (name: string, weight: number) =>
    setRewards(rewards.map((r) => (r.name === name ? { ...r, weight } : r)));

  const runPreview = useCallback(async () => {
    if (rewards.length === 0) {
      setPreview(null);
      setPreviewTotal(null);
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
  }, [rewards, completion, reference]);

  // Score live, a beat after the last edit.
  useEffect(() => {
    const timer = window.setTimeout(runPreview, 350);
    return () => window.clearTimeout(timer);
  }, [runPreview]);

  const copyReward = async (name: string) => {
    try {
      await navigator.clipboard.writeText(await exportReward(name));
      setCopied(name);
      window.setTimeout(() => setCopied(null), 1500);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  };

  const removeFromLibrary = async (name: string) => {
    try {
      await deleteReward(name);
      setRewards(rewards.filter((r) => r.name !== name));
      await refreshLibrary();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  };

  const loadError = actionError ?? libraryError;

  return (
    <div className="flex flex-col gap-4">
      {loadError && (
        <p className="text-ui-11p5 text-destructive">
          {t("rl.rewards.loadError", { error: loadError })}
        </p>
      )}

      <div className="flex flex-col">
        {rewards.length === 0 && (
          <p className="text-ui-11p5 text-amber-700 dark:text-amber-300">
            {t("rl.rewards.empty")}
          </p>
        )}
        {rewards.map((selection) => {
          const record = byName.get(selection.name);
          const type = summarizeRule(record?.rule).type;
          return (
            <div
              key={selection.name}
              className="grid grid-cols-[auto_minmax(0,1fr)_auto_auto_auto] items-center gap-3 border-b border-border/60 py-2.5"
            >
              <span className="flex size-8 items-center justify-center rounded-lg bg-muted/70 text-muted-foreground">
                <HugeiconsIcon
                  icon={type ? RULE_ICON[type] : TextIcon}
                  className="size-4"
                />
              </span>
              <div className="min-w-0">
                <p className="flex items-center gap-2 text-xs font-medium text-foreground">
                  <span className="truncate">{selection.name}</span>
                  <span
                    className={cn(
                      "rounded-full px-1.5 text-ui-10",
                      record?.source === "user"
                        ? "bg-primary/10 text-primary"
                        : "bg-muted text-muted-foreground",
                    )}
                  >
                    {record?.source === "user"
                      ? t("rl.rewards.user")
                      : t("rl.rewards.bundled")}
                  </span>
                </p>
                <p className="truncate text-ui-11p5 text-muted-foreground/85">
                  <RewardSummaryLine record={record} />
                </p>
              </div>
              <div className="flex items-center gap-1 text-ui-11 text-muted-foreground">
                ×
                <Input
                  type="number"
                  step={0.5}
                  min={-10}
                  max={10}
                  aria-label={t("rl.rewards.weight")}
                  className="h-7 w-[calc(60px*var(--ui-space-scale,1))] text-right text-xs"
                  value={selection.weight}
                  onChange={(e) => {
                    const n = Number(e.target.value);
                    if (Number.isFinite(n)) {
                      setWeight(selection.name, Math.max(-10, Math.min(10, n)));
                    }
                  }}
                />
              </div>
              <ScoreChip value={preview?.[selection.name]} />
              <DropdownMenu>
                <DropdownMenuTrigger asChild={true}>
                  <Button
                    type="button"
                    size="icon-xs"
                    variant="ghost"
                    aria-label={t("rl.rewards.more", { name: selection.name })}
                  >
                    <HugeiconsIcon
                      icon={MoreVerticalIcon}
                      className="size-3.5"
                    />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onSelect={() => copyReward(selection.name)}>
                    <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
                    {copied === selection.name
                      ? t("rl.rewards.copied")
                      : t("rl.rewards.export")}
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onSelect={() =>
                      setRewards(
                        rewards.filter((r) => r.name !== selection.name),
                      )
                    }
                  >
                    <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
                    {t("rl.rewards.removeFromRun")}
                  </DropdownMenuItem>
                  {record?.source === "user" && (
                    <>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        variant="destructive"
                        onSelect={() => removeFromLibrary(selection.name)}
                      >
                        <HugeiconsIcon
                          icon={Delete02Icon}
                          className="size-3.5"
                        />
                        {t("rl.rewards.delete")}
                      </DropdownMenuItem>
                    </>
                  )}
                </DropdownMenuContent>
              </DropdownMenu>
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
                <span className="flex flex-col">
                  <span>{r.name}</span>
                  <span className="text-ui-11 text-muted-foreground">
                    {r.description}
                  </span>
                </span>
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
          <HugeiconsIcon icon={Upload01Icon} className="size-3.5" />
          {t("rl.rewards.import")}
        </Button>
        <span className="text-ui-11p5 text-muted-foreground/85">
          {t("rl.rewards.pythonLocked")}
        </span>
      </div>

      <div className="flex flex-col gap-2.5 rounded-xl border border-border/70 p-3">
        <div className="flex items-center gap-2">
          <HugeiconsIcon
            icon={TestTube01Icon}
            className="size-3.5 text-muted-foreground"
          />
          <p className="text-xs font-medium text-foreground">
            {t("rl.rewards.tryTitle")}
          </p>
          <span className="text-ui-11p5 text-muted-foreground">
            {t("rl.rewards.tryHint")}
          </span>
          {previewTotal !== null && (
            <span className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground">
              {t("rl.rewards.tryTotalLabel")}
              <ScoreChip value={previewTotal} />
            </span>
          )}
        </div>
        <div className="grid gap-2 md:grid-cols-[minmax(0,1fr)_minmax(0,14rem)]">
          <Textarea
            rows={6}
            aria-label={t("rl.rewards.tryCompletion")}
            className="font-mono text-xs"
            value={completion}
            onChange={(e) => setCompletion(e.target.value)}
          />
          <div className="flex flex-col gap-1.5">
            <Label className="text-ui-11 font-normal text-muted-foreground">
              {answerColumn && referenceEdit === null && rowReference
                ? t("rl.rewards.tryReferenceFromRow", { column: answerColumn })
                : t("rl.rewards.tryReference")}
            </Label>
            <Textarea
              rows={4}
              aria-label={t("rl.rewards.tryReference")}
              className="font-mono text-xs"
              value={reference}
              onChange={(e) => setReferenceEdit(e.target.value)}
            />
          </div>
        </div>
        {previewError && (
          <p className="text-ui-11p5 text-destructive">{previewError}</p>
        )}
      </div>

      <ImportRewardDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        onImported={(record) => {
          refreshLibrary();
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
