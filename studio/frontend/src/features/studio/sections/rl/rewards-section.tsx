// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { NewBadge } from "@/components/new-badge";
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
  DropdownMenuCheckboxItem,
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
import { Spinner } from "@/components/ui/spinner";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  type RewardRecord,
  type RuleType,
  canEditReward,
  compareRuleScores,
  deleteReward,
  exportReward,
  formatScore,
  groupRewardPreviews,
  importRewardFiles,
  MAX_REWARD_BYTES,
  MAX_REWARD_ZIP_BYTES,
  readRewardHead,
  readRewardZip,
  RewardImportError,
  type RewardImportFile,
  rewardReference,
  rewardReferenceColumn,
  summarizeRule,
  useRewardPreview,
  useRlWorkspaceStore,
  useTrainingConfigStore,
} from "@/features/training";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import {
  Award01Icon,
  Calculator01Icon,
  Cancel01Icon,
  CodeIcon,
  Copy01Icon,
  Delete02Icon,
  Download01Icon,
  EqualSignIcon,
  FileImportIcon,
  InformationCircleIcon,
  MoreVerticalIcon,
  PencilEdit02Icon,
  PlayIcon,
  PlusSignIcon,
  RegexIcon,
  RulerIcon,
  TestTube01Icon,
  TextIcon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { type ReactElement, useEffect, useRef, useState } from "react";
import { strToU8, zipSync } from "fflate";
import { useShallow } from "zustand/react/shallow";
import { SectionBox } from "../section-box";
import { RewardEditorDialog } from "./reward-editor-dialog";
import {
  RewardScoreBadge,
  RewardScoreExplanation,
  RewardTestResult,
} from "./reward-test-result";

const SAMPLE_COMPLETION =
  "<reasoning>\n48 + 24 = 72\n</reasoning>\n<answer>\n72\n</answer>";
const ALL_REWARDS = "*";

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

function ImportRewardDialog({
  open,
  onOpenChange,
  onImported,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImported: (records: RewardRecord[]) => void;
}): ReactElement {
  const t = useT();
  const [markdown, setMarkdown] = useState("");
  const [overwrite, setOverwrite] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [archive, setArchive] = useState<RewardImportFile[] | null>(null);
  const [pending, setPending] = useState(false);
  const busy = useRef(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const head = readRewardHead(markdown);
  const isPython = head?.kind === "python";
  const errorMessage = (error: unknown) =>
    error instanceof RewardImportError
      ? t(`rl.rewards.${error.code}`, { name: error.detail })
      : error instanceof Error
        ? error.message
        : String(error);

  const chooseFile = async (file: File) => {
    if (busy.current) return;
    busy.current = true;
    setPending(true);
    setError(null);
    setArchive(null);
    setMarkdown("");
    setOverwrite(false);
    try {
      const isZip = file.name.toLowerCase().endsWith(".zip");
      if (file.size > (isZip ? MAX_REWARD_ZIP_BYTES : MAX_REWARD_BYTES))
        throw new RewardImportError("importTooLarge");
      const bytes = new Uint8Array(await file.arrayBuffer());
      if (isZip) setArchive(readRewardZip(bytes));
      else setMarkdown(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
    } catch (error) {
      setError(errorMessage(error));
    } finally {
      busy.current = false;
      setPending(false);
    }
  };

  const submit = async () => {
    if (busy.current) return;
    if (!archive && !markdown.trim()) {
      setError(t("rl.rewards.importEmpty"));
      return;
    }
    busy.current = true;
    setPending(true);
    setError(null);
    const files = archive ?? [{ path: "REWARD.md", markdown }];
    try {
      const result = await importRewardFiles(files, overwrite);
      if (result.imported.length) onImported(result.imported);
      if (result.error) {
        if (result.imported.length) {
          setArchive(result.remaining);
          setError(
            t("rl.rewards.importPartial", {
              count: result.imported.length,
              total: files.length,
              error: errorMessage(result.error),
            }),
          );
        } else setError(errorMessage(result.error));
        return;
      }
      setMarkdown("");
      setArchive(null);
      setOverwrite(false);
      onOpenChange(false);
    } finally {
      busy.current = false;
      setPending(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!busy.current) onOpenChange(next);
      }}
    >
      <DialogContent className="flex flex-col" showCloseButton={!pending}>
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
            disabled={pending}
            onClick={() => fileInput.current?.click()}
          >
            <HugeiconsIcon icon={FileImportIcon} className="size-3.5" />
            {t("rl.rewards.importFile")}
          </Button>
          {archive ? (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              disabled={pending}
              onClick={() => {
                setArchive(null);
                setError(null);
              }}
            >
              {t("rl.rewards.importPasteInstead")}
            </Button>
          ) : (
            <span className="text-ui-11p5 text-muted-foreground">
              {t("rl.rewards.importOrPaste")}
            </span>
          )}
          <input
            ref={fileInput}
            type="file"
            accept=".md,.zip,text/markdown,text/plain,application/zip,application/x-zip-compressed"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) void chooseFile(file);
            }}
          />
        </div>
        {archive ? (
          <div className="flex min-h-0 flex-col gap-2">
            <p className="text-xs text-muted-foreground">
              {t("rl.rewards.importCount", { count: archive.length })}
            </p>
            <ul className="max-h-64 overflow-y-auto divide-y divide-border/60 rounded-lg border border-border/70 px-3">
              {archive.map((file) => {
                const metadata = readRewardHead(file.markdown);
                return (
                  <li key={file.path} className="flex gap-2 py-2.5">
                    <HugeiconsIcon
                      icon={TextIcon}
                      className="mt-0.5 size-3.5 shrink-0 text-muted-foreground"
                    />
                    <div className="min-w-0">
                      <p className="break-words text-xs font-medium">
                        {metadata?.name ?? file.path}
                      </p>
                      {metadata?.description && (
                        <p className="break-words text-ui-11p5 text-muted-foreground">
                          {metadata.description}
                        </p>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        ) : (
          <Textarea
            rows={9}
            aria-label="REWARD.md"
            disabled={pending}
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
        )}
        {!archive && head?.name && !isPython && (
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
        {!archive && isPython && (
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
            disabled={pending}
            onCheckedChange={(v) => setOverwrite(v === true)}
          />
          <Label
            htmlFor="reward-import-overwrite"
            className="text-xs font-normal text-muted-foreground"
          >
            {t("rl.rewards.overwrite")}
          </Label>
        </div>
        {error && (
          <p role="alert" className="break-words text-ui-11p5 text-destructive">
            {error}
          </p>
        )}
        <DialogFooter className="max-sm:mt-auto">
          <Button type="button" onClick={submit} disabled={pending}>
            {pending && <Spinner className="size-3.5" />}
            {t(pending ? "rl.rewards.importing" : "rl.rewards.importAction")}
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
  const [exporting, setExporting] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [exportNames, setExportNames] = useState<string[] | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [completion, setCompletion] = useState(SAMPLE_COMPLETION);
  const [referenceEdits, setReferenceEdits] = useState<Record<string, string>>(
    {},
  );
  const [editor, setEditor] = useState<{ record?: RewardRecord } | null>(null);
  const editorTrigger = useRef<HTMLElement | null>(null);
  const rewardPicker = useRef<HTMLButtonElement | null>(null);
  const [testRewardName, setTestRewardName] = useState("");

  useEffect(() => {
    refreshLibrary();
  }, [refreshLibrary]);

  const active = library.filter((r) => !r.shadowed);
  const byName = new Map(active.map((r) => [r.name, r]));
  const selectedNames = new Set(rewards.map((r) => r.name));
  const sortedRewards = [...rewards].sort((a, b) =>
    compareRuleScores(byName.get(a.name)?.rule, byName.get(b.name)?.rule),
  );
  const selectedRecords = sortedRewards
    .map(({ name }) => byName.get(name))
    .filter((record): record is RewardRecord =>
      Boolean(record?.valid && record.kind === "rule"),
    );
  const exportRecords = selectedRecords.filter(
    (record) => exportNames === null || exportNames.includes(record.name),
  );
  const testReward =
    selectedRecords.find((record) => record.name === testRewardName) ??
    selectedRecords[0];
  const testAll = testRewardName === ALL_REWARDS;
  const testSelections = testAll
    ? sortedRewards
    : testReward
      ? [{ name: testReward.name, weight: 1 }]
      : [];
  const addable = active
    .filter((r) => r.valid && r.kind === "rule" && !selectedNames.has(r.name))
    .sort((a, b) => compareRuleScores(a.rule, b.rule));

  const referenceColumns = [
    ...new Set(
      testSelections
        .map(({ name }) => rewardReferenceColumn(byName.get(name)?.rule))
        .filter((column): column is string => column !== null),
    ),
  ];
  const referenceValues = Object.fromEntries(
    referenceColumns.map((column) => {
      const fromRow = rewardReference(column, previewRow, mapping);
      return [
        column,
        referenceEdits[column] ??
          (fromRow.source ? fromRow.value : column === "answer" ? "72" : null),
      ];
    }),
  );
  const {
    data: previewData,
    error: previewError,
    pending: previewPending,
    run: runPreview,
  } = useRewardPreview(
    groupRewardPreviews(testSelections, library, referenceValues),
    completion,
    JSON.stringify(testSelections.map(({ name }) => byName.get(name)?.rule)),
  );
  const openEditor = (
    record: RewardRecord | undefined,
    trigger: HTMLElement | null,
  ) => {
    editorTrigger.current = trigger;
    setEditor({ record });
  };
  const closeEditor = () => {
    setEditor(null);
    window.requestAnimationFrame(() => editorTrigger.current?.focus());
  };

  const setWeight = (name: string, weight: number) =>
    setRewards(rewards.map((r) => (r.name === name ? { ...r, weight } : r)));

  const copyReward = async (name: string) => {
    try {
      await navigator.clipboard.writeText(await exportReward(name));
      setCopied(name);
      window.setTimeout(() => setCopied(null), 1500);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  };

  const downloadRewards = async () => {
    if (exporting || exportRecords.length === 0) return;
    setExporting(true);
    try {
      const files = await Promise.all(
        exportRecords.map(async ({ name }) => ({
          name,
          markdown: await exportReward(name),
        })),
      );
      if (files.length === 1) {
        await downloadFile(
          files[0].markdown,
          `${files[0].name}.REWARD.md`,
          "text/markdown;charset=utf-8",
        );
      } else {
        const entries = Object.fromEntries(
          files.map(({ name, markdown }) => [
            `${name}/REWARD.md`,
            strToU8(markdown),
          ]),
        );
        await downloadFile(zipSync(entries), "rewards.zip", "application/zip");
      }
      setExportOpen(false);
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error(t("rl.rewards.exportFailed"), {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    } finally {
      setExporting(false);
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
    <SectionBox
      title={t("rl.rewards.title")}
      titleBadge={<NewBadge />}
      description={t("rl.rewards.description")}
      icon={Award01Icon}
      chipTint="var(--chart-2)"
      compactTitleAction
      titleAction={
        <div className="flex items-center gap-2">
          <Tooltip>
            <TooltipTrigger asChild disableClickToggle>
              <Button
                type="button"
                size="icon-sm"
                variant="outline"
                aria-label={t("rl.rewards.import")}
                onClick={() => setImportOpen(true)}
              >
                <HugeiconsIcon icon={Upload01Icon} className="size-3.5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t("rl.rewards.import")}</TooltipContent>
          </Tooltip>
          <DropdownMenu open={exportOpen} onOpenChange={setExportOpen}>
            <Tooltip>
              <TooltipTrigger asChild disableClickToggle>
                <DropdownMenuTrigger asChild>
                  <Button
                    type="button"
                    size="icon-sm"
                    variant="outline"
                    aria-label={t(
                      exporting
                        ? "rl.rewards.exportingFile"
                        : "rl.rewards.exportFile",
                    )}
                    disabled={
                      exporting ||
                      selectedRecords.length === 0 ||
                      Boolean(libraryError)
                    }
                  >
                    <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
                  </Button>
                </DropdownMenuTrigger>
              </TooltipTrigger>
              <TooltipContent>
                {t(
                  exporting
                    ? "rl.rewards.exportingFile"
                    : "rl.rewards.exportFile",
                )}
              </TooltipContent>
            </Tooltip>
            <DropdownMenuContent align="end" className="w-80">
              <DropdownMenuCheckboxItem
                checked={exportRecords.length === selectedRecords.length}
                disabled={exporting}
                onSelect={(event) => event.preventDefault()}
                onCheckedChange={(checked) =>
                  setExportNames(checked ? null : [])
                }
              >
                {t("rl.rewards.exportSelectAll")}
              </DropdownMenuCheckboxItem>
              <DropdownMenuSeparator />
              {selectedRecords.map((record) => (
                <DropdownMenuCheckboxItem
                  key={record.name}
                  checked={exportRecords.some(
                    ({ name }) => name === record.name,
                  )}
                  disabled={exporting}
                  onSelect={(event) => event.preventDefault()}
                  onCheckedChange={(checked) =>
                    setExportNames(
                      checked
                        ? [
                            ...exportRecords.map(({ name }) => name),
                            record.name,
                          ]
                        : exportRecords
                            .filter(({ name }) => name !== record.name)
                            .map(({ name }) => name),
                    )
                  }
                >
                  <HugeiconsIcon
                    icon={RULE_ICON[record.rule?.type as RuleType] ?? TextIcon}
                    className="mt-0.5 size-4 shrink-0 self-start text-muted-foreground"
                  />
                  <span className="flex min-w-0 flex-col gap-0.5">
                    <span className="text-sm font-medium leading-5">
                      {record.name}
                    </span>
                    <span className="whitespace-normal break-words text-xs leading-relaxed text-muted-foreground">
                      {record.description}
                    </span>
                  </span>
                </DropdownMenuCheckboxItem>
              ))}
              <div className="sticky bottom-0 bg-popover pt-1">
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  disabled={exporting || exportRecords.length === 0}
                  onSelect={(event) => {
                    event.preventDefault();
                    void downloadRewards();
                  }}
                  className="font-medium"
                >
                  {exporting ? (
                    <Spinner className="size-3.5" />
                  ) : (
                    <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
                  )}
                  {exporting
                    ? t("rl.rewards.exportingFile")
                    : t("rl.rewards.exportSelected", {
                        count: exportRecords.length,
                      })}
                </DropdownMenuItem>
              </div>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      }
    >
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
          {sortedRewards.map((selection) => {
            const record = byName.get(selection.name);
            const type = summarizeRule(record?.rule).type;
            return (
              <div
                key={selection.name}
                className="grid grid-cols-[auto_minmax(0,1fr)_auto_auto] items-center gap-3 border-b border-border/60 py-2.5"
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
                        setWeight(
                          selection.name,
                          Math.max(-10, Math.min(10, n)),
                        );
                      }
                    }}
                  />
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild={true}>
                    <Button
                      type="button"
                      size="icon-xs"
                      variant="ghost"
                      aria-label={t("rl.rewards.more", {
                        name: selection.name,
                      })}
                      id={"reward-menu-" + selection.name}
                    >
                      <HugeiconsIcon
                        icon={MoreVerticalIcon}
                        className="size-3.5"
                      />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent
                    align="end"
                    onCloseAutoFocus={(event) => {
                      if (editor) event.preventDefault();
                    }}
                  >
                    {record && canEditReward(record) && (
                      <DropdownMenuItem
                        onSelect={() =>
                          openEditor(
                            record,
                            document.getElementById(
                              "reward-menu-" + selection.name,
                            ),
                          )
                        }
                      >
                        <HugeiconsIcon
                          icon={PencilEdit02Icon}
                          className="size-3.5"
                        />
                        {t(
                          record.source === "user"
                            ? "rl.rewardEditor.editTitle"
                            : "rl.rewardEditor.copyTitle",
                        )}
                      </DropdownMenuItem>
                    )}
                    <DropdownMenuItem
                      onSelect={() => copyReward(selection.name)}
                    >
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
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                ref={rewardPicker}
                type="button"
                size="sm"
                variant="outline"
                className="w-full justify-between"
                disabled={rewards.length >= 16 || Boolean(libraryError)}
              >
                <HugeiconsIcon
                  icon={PlusSignIcon}
                  className="size-3.5 text-muted-foreground"
                  aria-hidden="true"
                />
                <span className="flex-1 text-start">{t("rl.rewards.add")}</span>
                <HugeiconsIcon
                  icon={ChevronDownStandardIcon}
                  className="size-4 text-muted-foreground"
                />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              align="start"
              sideOffset={6}
              onCloseAutoFocus={(event) => {
                if (editor) event.preventDefault();
              }}
            >
              {addable.map((record) => (
                <DropdownMenuItem
                  key={record.name}
                  onSelect={() => {
                    const current =
                      useTrainingConfigStore.getState().grpoRewards;
                    if (
                      current.length < 16 &&
                      !current.some(({ name }) => name === record.name)
                    )
                      setRewards([
                        ...current,
                        { name: record.name, weight: 1 },
                      ]);
                  }}
                >
                  <HugeiconsIcon
                    icon={RULE_ICON[record.rule?.type as RuleType] ?? TextIcon}
                    className="mt-0.5 size-4 shrink-0 self-start text-muted-foreground"
                  />
                  <span className="flex min-w-0 flex-col gap-0.5">
                    <span className="text-sm font-medium leading-5">
                      {record.name}
                    </span>
                    <span className="whitespace-normal break-words text-xs leading-relaxed text-muted-foreground">
                      {record.description}
                    </span>
                  </span>
                </DropdownMenuItem>
              ))}
              <DropdownMenuItem
                onSelect={() => openEditor(undefined, rewardPicker.current)}
              >
                <HugeiconsIcon
                  icon={PencilEdit02Icon}
                  className="mt-0.5 size-4 shrink-0 self-start text-muted-foreground"
                />
                <span className="flex min-w-0 flex-col gap-0.5">
                  <span className="text-sm font-medium leading-5">
                    {t("rl.rewardEditor.custom")}
                  </span>
                  <span className="whitespace-normal break-words text-xs leading-relaxed text-muted-foreground">
                    {t("rl.rewardEditor.description")}
                  </span>
                </span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        <div className="flex flex-col gap-2.5 rounded-xl border border-border/70 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <HugeiconsIcon
              icon={TestTube01Icon}
              className="size-3.5 text-muted-foreground"
            />
            <p className="text-xs font-medium text-foreground">
              {t("rl.rewards.tryTitle")}
            </p>
            <div className="ms-auto flex min-w-0 max-w-full items-center gap-2">
              <Select
                value={testAll ? ALL_REWARDS : (testReward?.name ?? "")}
                onValueChange={setTestRewardName}
                disabled={!rewards.length}
              >
                <SelectTrigger
                  size="sm"
                  aria-label={t("rl.rewards.rewardToTest")}
                  className="min-w-0 max-w-56 flex-1"
                >
                  <SelectValue placeholder={t("rl.rewards.rewardToTest")} />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={ALL_REWARDS}>
                    {t("rl.rewards.allSelected")}
                  </SelectItem>
                  {selectedRecords.map((record) => (
                    <SelectItem key={record.name} value={record.name}>
                      {record.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Tooltip>
                <TooltipTrigger asChild disableClickToggle>
                  <Button
                    type="button"
                    size="icon-sm"
                    variant="outline"
                    aria-label={t(
                      previewPending
                        ? "rl.rewardEditor.scoring"
                        : "rl.rewardEditor.run",
                    )}
                    onClick={() => void runPreview()}
                    disabled={
                      !testSelections.length ||
                      previewPending ||
                      Boolean(libraryError)
                    }
                  >
                    {previewPending ? (
                      <Spinner className="size-3.5" />
                    ) : (
                      <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {t(
                    previewPending
                      ? "rl.rewardEditor.scoring"
                      : "rl.rewardEditor.run",
                  )}
                </TooltipContent>
              </Tooltip>
            </div>
          </div>
          <div
            className={cn(
              "grid gap-3",
              referenceColumns.length > 0 &&
                "md:grid-cols-[minmax(0,1fr)_minmax(0,14rem)]",
            )}
          >
            <Textarea
              rows={6}
              aria-label={t("rl.rewards.tryCompletion")}
              className="font-mono text-xs"
              dir="auto"
              value={completion}
              onChange={(e) => setCompletion(e.target.value)}
            />
            {referenceColumns.length > 0 && (
              <div className="flex flex-col gap-3">
                {referenceColumns.map((column, index) => {
                  const fromRow = rewardReference(column, previewRow, mapping);
                  const manual = Object.hasOwn(referenceEdits, column);
                  const label =
                    fromRow.source && !manual
                      ? t("rl.rewardEditor.fromColumn", {
                          column: fromRow.source,
                        })
                      : t("rl.rewardEditor.referenceFor", { column });
                  return (
                    <div className="flex flex-col gap-1.5" key={column}>
                      <Label
                        htmlFor={"reward-reference-" + index}
                        className="text-ui-11 font-normal text-muted-foreground"
                      >
                        {label}
                      </Label>
                      <Textarea
                        id={"reward-reference-" + index}
                        rows={3}
                        dir="auto"
                        className="font-mono text-xs"
                        value={referenceValues[column] ?? ""}
                        onChange={(e) =>
                          setReferenceEdits((current) => ({
                            ...current,
                            [column]: e.target.value,
                          }))
                        }
                      />
                      {(!fromRow.source || manual) && (
                        <p className="text-xs text-muted-foreground">
                          {t("rl.rewardEditor.manualReference")}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
          {testAll && previewData && (
            <ul
              aria-label={t("rl.rewards.scoreBreakdown")}
              className="flex flex-col gap-3 border-t border-border/60 pt-3"
            >
              {[...previewData.scores]
                .sort((a, b) => b.weighted - a.weighted)
                .map((result) => {
                  const weight =
                    rewards.find(({ name }) => name === result.name)?.weight ??
                    1;
                  return (
                    <li
                      key={result.name}
                      className="flex min-w-0 flex-wrap items-center justify-between gap-x-3 gap-y-1 text-xs"
                    >
                      <span className="flex min-w-0 items-center gap-1 text-muted-foreground">
                        <span className="min-w-0 break-words">
                          {result.name}
                        </span>
                        {result.explanation && (
                          <RewardScoreExplanation
                            explanation={result.explanation}
                          />
                        )}
                      </span>
                      <span
                        dir="ltr"
                        className="ms-auto flex shrink-0 items-center gap-2 tabular-nums"
                      >
                        <span className="text-muted-foreground">
                          {formatScore(result.score)} × {weight} =
                        </span>
                        <RewardScoreBadge score={result.weighted} />
                      </span>
                    </li>
                  );
                })}
            </ul>
          )}
          <RewardTestResult
            score={previewData?.total}
            error={previewError}
            total={testAll}
            explanation={
              !testAll ? previewData?.scores[0]?.explanation : undefined
            }
          />
        </div>

        {editor && (
          <RewardEditorDialog
            record={editor.record}
            library={library}
            previewRow={previewRow}
            mapping={mapping}
            onClose={closeEditor}
            onSaved={(record) => {
              void refreshLibrary();
              if (!selectedNames.has(record.name))
                setRewards(
                  [...rewards, { name: record.name, weight: 1 }].slice(0, 16),
                );
            }}
          />
        )}

        <ImportRewardDialog
          open={importOpen}
          onOpenChange={setImportOpen}
          onImported={(records) => {
            void refreshLibrary();
            const current = useTrainingConfigStore.getState().grpoRewards;
            const selected = new Set(current.map(({ name }) => name));
            setRewards(
              [
                ...current,
                ...records
                  .filter(({ name }) => !selected.has(name))
                  .map(({ name }) => ({ name, weight: 1 })),
              ].slice(0, 16),
            );
          }}
        />
      </div>
    </SectionBox>
  );
}
