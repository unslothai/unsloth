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
  type RewardPreviewResponse,
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
import {
  alpacaPromptCell,
  pythonRewardRow,
} from "@/features/training/lib/rl-roles";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { GrpoPythonRewardIsolation } from "@/types/training";
import {
  Calculator01Icon,
  Cancel01Icon,
  CodeIcon,
  Copy01Icon,
  Delete02Icon,
  EqualSignIcon,
  FileImportIcon,
  MoreVerticalIcon,
  PythonIcon,
  RegexIcon,
  RulerIcon,
  Shield01Icon,
  SourceCodeIcon,
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
// The template already opens <start_working_out>, so the reply starts inside it.
const REASONING_SAMPLE_COMPLETION =
  "48 + 24 = 72<end_working_out><SOLUTION>72</SOLUTION>";

const SANDBOX_NAME: Record<string, string> = {
  "mxc-processcontainer": "Windows MXC",
  bubblewrap: "bubblewrap",
  "macos-seatbelt": "macOS Seatbelt",
};

const RULE_ICON: Record<RuleType, IconSvgElement> = {
  regex: RegexIcon,
  exact_match: EqualSignIcon,
  numeric: Calculator01Icon,
  json_schema: CodeIcon,
  length: RulerIcon,
};

function RewardSummaryLine({
  record,
  loaded,
}: {
  record: RewardRecord | undefined;
  loaded: boolean;
}): ReactElement | null {
  const t = useT();
  if (!record) {
    return loaded ? (
      <span className="text-destructive">{t("rl.rewards.invalid")}</span>
    ) : null;
  }
  if (record.kind === "python") {
    return (
      <span title={record.description}>
        <span className="font-mono">
          {t("rl.rewards.summary.python", { entry: record.entry ?? "reward" })}
        </span>
        {record.description && (
          <span className="text-muted-foreground/70">
            {" · "}
            {record.description}
          </span>
        )}
      </span>
    );
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

function ScoreChip({
  value,
  error,
}: {
  value: number | null | undefined;
  error?: string | null;
}): ReactElement {
  const t = useT();
  if (error) {
    return (
      <span
        title={error}
        className="w-14 rounded-md bg-rose-100 px-1.5 py-0.5 text-center text-ui-10 text-rose-700 dark:bg-rose-950/50 dark:text-rose-300"
      >
        {t("rl.rewards.scoreError")}
      </span>
    );
  }
  if (value === undefined) {
    return <span className="w-14" />;
  }
  if (value === null) {
    return (
      <span className="w-14 rounded-md bg-muted px-1.5 py-0.5 text-center text-ui-10 text-muted-foreground">
        {t("rl.rewards.scoreNone")}
      </span>
    );
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
      setOverwrite(false);
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
          className="max-h-[40vh] font-mono text-xs"
          value={markdown}
          placeholder={
            "---\nname: my-reward\nkind: rule\ndescription: ...\n---\ntype: regex\n..."
          }
          onChange={(e) => {
            setMarkdown(e.target.value);
            setError(null);
          }}
        />
        {head?.name && (
          <div className="flex items-start gap-2 rounded-lg border border-border/70 px-3 py-2 text-xs">
            <HugeiconsIcon
              icon={isPython ? PythonIcon : TextIcon}
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
              icon={Shield01Icon}
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

function CodeDialog({
  record,
  onOpenChange,
}: {
  record: RewardRecord | null;
  onOpenChange: (open: boolean) => void;
}): ReactElement {
  const t = useT();
  return (
    <Dialog open={record !== null} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {t("rl.rewards.codeTitle", { name: record?.name ?? "" })}
          </DialogTitle>
          <DialogDescription>
            {t("rl.rewards.codeDescription")}
          </DialogDescription>
        </DialogHeader>
        <pre className="max-h-[60vh] overflow-auto rounded-lg border border-border/70 bg-muted/40 p-3 font-mono text-ui-11p5 leading-relaxed">
          {record?.code}
        </pre>
      </DialogContent>
    </Dialog>
  );
}

/** Row 1's prompt cell as stored (chat turns stay turns) and the system prompt training adds
 *  (rl.py prompt_turns), so a reward that reads prompts scores the same as in a run. */
function previewPrompt(
  row: Record<string, unknown> | null,
  mapping: Record<string, string>,
  configured: string,
): {
  prompt: string | Record<string, unknown>[] | null;
  systemPrompt: string | null;
} {
  const column = (role: string) =>
    Object.entries(mapping).find(([, r]) => r === role)?.[0];
  const promptColumn = column("prompt");
  const systemColumn = column("system");
  const cell = alpacaPromptCell(row, mapping, promptColumn);
  const turns =
    Array.isArray(cell) &&
    cell.every((m) => typeof m === "object" && m !== null);
  const own = systemColumn ? previewCell(row?.[systemColumn]) : "";
  return {
    prompt: turns
      ? (cell as Record<string, unknown>[])
      : previewCell(cell) || null,
    systemPrompt: own || (configured.trim() ? configured : null),
  };
}

export function RewardsSection(): ReactElement {
  const t = useT();
  const {
    rewards,
    setRewards,
    mapping,
    reasoning,
    grpoSystemPrompt,
    isolationMode,
    setIsolationMode,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      reasoning: s.grpoReasoningFormat,
      rewards: s.grpoRewards,
      setRewards: s.setGrpoRewards,
      mapping: s.rlRoleMapping,
      grpoSystemPrompt: s.grpoSystemPrompt,
      isolationMode: s.grpoPythonRewardIsolation,
      setIsolationMode: s.setGrpoPythonRewardIsolation,
    })),
  );
  const { library, libraryLoaded, libraryError, refreshLibrary, previewRow } =
    useRlWorkspaceStore(
      useShallow((s) => ({
        library: s.library,
        libraryLoaded: s.libraryLoaded,
        libraryError: s.libraryError,
        refreshLibrary: s.refreshLibrary,
        previewRow: s.previewRow,
      })),
    );
  const [actionError, setActionError] = useState<string | null>(null);
  const [importOpen, setImportOpen] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [completion, setCompletion] = useState(
    reasoning ? REASONING_SAMPLE_COMPLETION : SAMPLE_COMPLETION,
  );
  useEffect(() => {
    // Follow the format switch unless the sample was edited.
    setCompletion((prev) =>
      prev === SAMPLE_COMPLETION || prev === REASONING_SAMPLE_COMPLETION
        ? reasoning
          ? REASONING_SAMPLE_COMPLETION
          : SAMPLE_COMPLETION
        : prev,
    );
  }, [reasoning]);
  const [referenceEdit, setReferenceEdit] = useState<string | null>(null);
  const [preview, setPreview] = useState<Record<string, number | null> | null>(
    null,
  );
  // A Python reward that could not score; the others keep theirs.
  const [previewErrors, setPreviewErrors] = useState<Record<string, string>>(
    {},
  );
  const [isolation, setIsolation] =
    useState<RewardPreviewResponse["isolation"]>(null);
  const [codeFor, setCodeFor] = useState<RewardRecord | null>(null);
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
  const addable = active.filter((r) => r.valid && !selectedNames.has(r.name));

  const setWeight = (name: string, weight: number) =>
    setRewards(rewards.map((r) => (r.name === name ? { ...r, weight } : r)));

  // An overwrite keeps the selection but changes the rule or code, so the scores must rerun.
  const selectedRules = JSON.stringify(
    rewards.map((r) => {
      const entry = active.find((a) => a.name === r.name);
      return entry
        ? [entry.rule ?? null, entry.entry ?? null, entry.code ?? null]
        : null;
    }),
  );
  const previewRequest = useRef(0);
  const runPreview = useCallback(async () => {
    const request = ++previewRequest.current;
    if (rewards.length === 0) {
      setPreview(null);
      setPreviewErrors({});
      setPreviewTotal(null);
      return;
    }
    // The backend scores each rule against its own compare_to column from row, as in training.
    try {
      // No answer role, no answer kwarg in training; with no dataset yet, assume one.
      const row = pythonRewardRow(
        previewRow ?? {},
        answerColumn || !previewRow ? reference : null,
      );
      const { prompt, systemPrompt } = previewPrompt(
        previewRow,
        mapping,
        grpoSystemPrompt,
      );
      const res = await previewRewards(
        rewards,
        completion,
        reference || null,
        row,
        prompt,
        systemPrompt,
        isolationMode,
      );
      if (request !== previewRequest.current) return;
      setPreview(
        Object.fromEntries(res.scores.map((s) => [s.name, s.weighted])),
      );
      setPreviewErrors(
        Object.fromEntries(
          res.scores.flatMap((s) => (s.error ? [[s.name, s.error]] : [])),
        ),
      );
      setPreviewTotal(res.total);
      setIsolation(res.isolation);
      setPreviewError(null);
    } catch (err) {
      if (request !== previewRequest.current) return;
      // Stale scores next to an error read as if the failing reward had scored.
      setPreview(null);
      setPreviewErrors({});
      setPreviewTotal(null);
      setIsolation(null);
      setPreviewError(err instanceof Error ? err.message : String(err));
    }
    // selectedRules stands in for the library entries (byName is rebuilt every render).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    rewards,
    completion,
    reference,
    previewRow,
    answerColumn,
    mapping,
    grpoSystemPrompt,
    isolationMode,
    selectedRules,
  ]);

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
      // The selection may have changed while the request ran: filter the latest one.
      setRewards(
        useTrainingConfigStore
          .getState()
          .grpoRewards.filter((r) => r.name !== name),
      );
      await refreshLibrary();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  };

  const loadError = actionError ?? libraryError;
  const usesPython = rewards.some((r) => byName.get(r.name)?.kind === "python");
  // The sandbox refusal is the same for every Python reward: say it once.
  const errorTexts = Object.entries(previewErrors).reduce<string[]>(
    (out, [name, text]) => {
      const shared = Object.values(previewErrors).filter((e) => e === text);
      const line = shared.length > 1 ? text : `${name}: ${text}`;
      return out.includes(line) ? out : [...out, line];
    },
    [],
  );

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
          const isPython = record?.kind === "python";
          return (
            <div
              key={selection.name}
              className="grid grid-cols-[auto_minmax(0,1fr)_auto_auto_auto] items-center gap-3 border-b border-border/60 py-2.5"
            >
              <span className="flex size-8 items-center justify-center rounded-lg bg-muted/70 text-muted-foreground">
                <HugeiconsIcon
                  icon={
                    isPython ? PythonIcon : type ? RULE_ICON[type] : TextIcon
                  }
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
                  {isPython && (
                    <span className="rounded-full bg-sky-100 px-1.5 text-ui-10 text-sky-700 dark:bg-sky-950/50 dark:text-sky-300">
                      {t("rl.rewards.python")}
                    </span>
                  )}
                </p>
                <p className="truncate text-ui-11p5 text-muted-foreground/85">
                  <RewardSummaryLine record={record} loaded={libraryLoaded} />
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
              <ScoreChip
                value={preview?.[selection.name]}
                error={previewErrors[selection.name]}
              />
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
                  {isPython && record && (
                    <DropdownMenuItem onSelect={() => setCodeFor(record)}>
                      <HugeiconsIcon
                        icon={SourceCodeIcon}
                        className="size-3.5"
                      />
                      {t("rl.rewards.viewCode")}
                    </DropdownMenuItem>
                  )}
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
      </div>

      {usesPython && (
        <div className="flex flex-col gap-1.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <Label className="text-xs font-medium text-foreground">
              {t("rl.rewards.isolationLabel")}
            </Label>
            <Select
              value={isolationMode}
              onValueChange={(value) =>
                setIsolationMode(value as GrpoPythonRewardIsolation)
              }
            >
              <SelectTrigger
                size="sm"
                aria-label={t("rl.rewards.isolationLabel")}
                className="w-[calc(220px*var(--ui-space-scale,1))]"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="required">
                  {t("rl.rewards.isolationRequired")}
                </SelectItem>
                <SelectItem value="auto">
                  {t("rl.rewards.isolationAuto")}
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
          <p
            className={cn(
              "text-ui-11p5",
              isolationMode === "auto"
                ? "text-amber-700 dark:text-amber-300"
                : "text-muted-foreground/85",
            )}
          >
            {isolationMode === "auto"
              ? t("rl.rewards.isolationAutoWarning")
              : t("rl.rewards.isolationRequiredHint")}
          </p>
        </div>
      )}

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
        {isolation && !previewError && (
          <p
            className={cn(
              "flex items-center gap-1.5 text-ui-11p5",
              isolation.os_isolation
                ? "text-emerald-700 dark:text-emerald-300"
                : "text-amber-700 dark:text-amber-300",
            )}
          >
            <HugeiconsIcon icon={Shield01Icon} className="size-3.5 shrink-0" />
            {isolation.os_isolation
              ? t("rl.rewards.isolationOs", {
                  backend: SANDBOX_NAME[isolation.backend] ?? isolation.backend,
                })
              : t("rl.rewards.isolationSoftware")}
          </p>
        )}
        {(previewError ? [previewError] : errorTexts).map((text) => (
          <p
            key={text}
            className="whitespace-pre-wrap break-words text-ui-11p5 text-destructive"
          >
            {text}
          </p>
        ))}
      </div>

      <CodeDialog
        record={codeFor}
        onOpenChange={(open) => !open && setCodeFor(null)}
      />
      <ImportRewardDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        onImported={(record) => {
          refreshLibrary();
          // Read now, not at submit: edits made while the import was pending must survive.
          const current = useTrainingConfigStore.getState().grpoRewards;
          if (!current.some((r) => r.name === record.name)) {
            setRewards(
              [...current, { name: record.name, weight: 1 }].slice(0, 16),
            );
          }
        }}
      />
    </div>
  );
}
