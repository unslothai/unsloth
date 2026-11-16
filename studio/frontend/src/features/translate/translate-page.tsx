// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { useNativeFileDrop } from "@/features/native-intents";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { MenuTickIcon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  ArrowDataTransferHorizontalIcon,
  ArrowRight02Icon,
  Cancel01Icon,
  ClipboardPasteIcon,
  Clock01Icon,
  Copy01Icon,
  SlidersHorizontalIcon,
  StopIcon,
  Tick02Icon,
  TranslateIcon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import {
  type ReactNode,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import { useShallow } from "zustand/react/shallow";
import { TranslateHistorySheet } from "./history-sheet";
import {
  TRANSLATE_LANGUAGES,
  findTranslateLanguage,
  translateLanguageLabel,
} from "./languages";
import {
  CHAT_TRANSLATE_MODEL,
  DEDICATED_TRANSLATE_MODELS,
  type TranslateModelId,
  dedicatedTranslateModel,
  formatModelBytes,
  isTranslateModelId,
} from "./models";
import {
  type TranslatePhase,
  translateModelDownloaded,
} from "./translate-runner";
import {
  TRANSLATE_SOURCE_MAX_CHARS,
  translationIsStale,
  useTranslateStore,
} from "./translate-store";
import { TRANSLATE_TONES, type TranslateTone } from "./translate-text";

const AUTO_TRANSLATE_DELAY_MS = 800;
const TEXT_FILE_EXTENSIONS =
  ".txt,.md,.markdown,.srt,.vtt,.csv,.tsv,.json,.html,.xml,.yaml,.yml,.log,.tex,.rst";

const IS_MAC =
  typeof navigator !== "undefined" &&
  /Mac|iPhone|iPad/.test(navigator.platform);

const TONE_LABEL: Record<TranslateTone, TranslationKey> = {
  default: "translate.options.toneDefault",
  formal: "translate.options.toneFormal",
  casual: "translate.options.toneCasual",
};

const MODEL_NOTE: Record<
  (typeof DEDICATED_TRANSLATE_MODELS)[number]["id"],
  TranslationKey
> = {
  "hy-mt2-1.8b": "translate.model.fast",
  "hy-mt2-7b": "translate.model.quality",
};

function IconButton({
  icon,
  label,
  onClick,
  disabled,
  active,
}: {
  icon: IconSvgElement;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
}) {
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      title={label}
      aria-label={label}
      aria-pressed={active}
      disabled={disabled}
      onClick={onClick}
      className="text-muted-foreground"
    >
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4" />
    </Button>
  );
}

function LanguagePicker({
  value,
  onChange,
}: {
  value: string;
  onChange: (code: string) => void;
}) {
  const t = useT();
  const locale = useLocale();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const languages = useMemo(
    () =>
      TRANSLATE_LANGUAGES.map((language) => ({
        ...language,
        label: translateLanguageLabel(language, locale),
      })),
    [locale],
  );
  const needle = query.trim().toLowerCase();
  const shown = needle
    ? languages.filter((language) =>
        [language.label, language.name, language.code].some((name) =>
          name.toLowerCase().includes(needle),
        ),
      )
    : languages;
  const current = languages.find((language) => language.code === value);

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) setQuery("");
      }}
    >
      <PopoverTrigger asChild={true}>
        <Button
          type="button"
          variant="picker"
          aria-label={t("translate.targetLanguage")}
          className="min-w-40 justify-between rounded-full border"
        >
          <span className="min-w-0 truncate">{current?.label ?? value}</span>
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={2}
            className="size-4 shrink-0 text-muted-foreground"
          />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={4}
        className="w-64 gap-0 rounded-lg p-1"
      >
        {/* Filtered here: the labels are in the interface language, the search also takes English. */}
        <Command
          shouldFilter={false}
          className="rounded-none bg-transparent p-0"
        >
          <CommandInput
            value={query}
            onValueChange={setQuery}
            aria-label={t("translate.searchLanguages")}
            placeholder={t("translate.searchLanguages")}
          />
          <CommandList>
            <CommandEmpty>{t("translate.noLanguages")}</CommandEmpty>
            {shown.map((language) => (
              <CommandItem
                key={language.code}
                value={language.code}
                onSelect={() => {
                  onChange(language.code);
                  setOpen(false);
                  setQuery("");
                }}
              >
                <span className="min-w-0 flex-1 truncate">
                  {language.label}
                </span>
                {language.code === value && (
                  <HugeiconsIcon
                    icon={MenuTickIcon}
                    strokeWidth={2}
                    className="size-4"
                  />
                )}
              </CommandItem>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

type Downloaded = Partial<Record<TranslateModelId, boolean>>;

function ModelMenu({
  value,
  onChange,
  downloaded,
}: {
  value: TranslateModelId;
  onChange: (id: TranslateModelId) => void;
  downloaded: Downloaded;
}) {
  const t = useT();
  const label =
    dedicatedTranslateModel(value)?.label ?? t("translate.model.chat");
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <Button
          type="button"
          variant="ghost"
          aria-label={t("translate.model.label")}
          title={t("translate.model.label")}
          className="max-w-56 text-muted-foreground"
        >
          <span className="min-w-0 truncate">{label}</span>
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={2}
            className="size-4 shrink-0"
          />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-80">
        <DropdownMenuLabel>{t("translate.model.label")}</DropdownMenuLabel>
        <DropdownMenuRadioGroup
          value={value}
          onValueChange={(next) => isTranslateModelId(next) && onChange(next)}
        >
          {DEDICATED_TRANSLATE_MODELS.map((model) => (
            <DropdownMenuRadioItem key={model.id} value={model.id}>
              <span className="flex min-w-0 flex-col">
                <span>{model.label}</span>
                <span className="text-xs text-muted-foreground">
                  {t(MODEL_NOTE[model.id])}
                  {" · "}
                  {downloaded[model.id]
                    ? t("translate.model.downloaded")
                    : t("translate.model.downloadSize", {
                        size: formatModelBytes(model.sizeBytes),
                      })}
                </span>
              </span>
            </DropdownMenuRadioItem>
          ))}
          <DropdownMenuRadioItem value={CHAT_TRANSLATE_MODEL}>
            <span className="flex min-w-0 flex-col">
              <span>{t("translate.model.chat")}</span>
              <span className="text-xs text-muted-foreground">
                {t("translate.model.chatDescription")}
              </span>
            </span>
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function OptionRow({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: (id: string) => ReactNode;
}) {
  const id = useId();
  return (
    <div className="flex items-center justify-between gap-4">
      <div className="flex min-w-0 flex-col">
        <label htmlFor={id} className="text-sm">
          {label}
        </label>
        {description && (
          <span className="text-xs text-muted-foreground">{description}</span>
        )}
      </div>
      {children(id)}
    </div>
  );
}

function OptionsPopover() {
  const t = useT();
  const { tone, autoTranslate, saveHistory } = useTranslateStore(
    useShallow((state) => ({
      tone: state.prefs.tone,
      autoTranslate: state.prefs.autoTranslate,
      saveHistory: state.prefs.saveHistory,
    })),
  );
  const setPrefs = useTranslateStore((state) => state.setPrefs);
  return (
    <Popover>
      <PopoverTrigger asChild={true}>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          title={t("translate.options.label")}
          aria-label={t("translate.options.label")}
          className="text-muted-foreground"
        >
          <HugeiconsIcon
            icon={SlidersHorizontalIcon}
            strokeWidth={1.75}
            className="size-4"
          />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" sideOffset={4} className="w-80">
        <OptionRow label={t("translate.options.tone")}>
          {(id) => (
            <Select
              value={tone}
              onValueChange={(next) =>
                setPrefs({ tone: next as TranslateTone })
              }
            >
              <SelectTrigger id={id} size="sm" className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TRANSLATE_TONES.map((option) => (
                  <SelectItem key={option} value={option}>
                    {t(TONE_LABEL[option])}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </OptionRow>
        <OptionRow
          label={t("translate.options.auto")}
          description={t("translate.options.autoDescription")}
        >
          {(id) => (
            <Switch
              id={id}
              checked={autoTranslate}
              onCheckedChange={(checked) =>
                setPrefs({ autoTranslate: checked })
              }
            />
          )}
        </OptionRow>
        <OptionRow
          label={t("translate.options.saveHistory")}
          description={t("translate.options.saveHistoryDescription")}
        >
          {(id) => (
            <Switch
              id={id}
              checked={saveHistory}
              onCheckedChange={(checked) => setPrefs({ saveHistory: checked })}
            />
          )}
        </OptionRow>
      </PopoverContent>
    </Popover>
  );
}

function PhaseStatus({
  phase,
  model,
}: { phase: TranslatePhase; model: string }) {
  const t = useT();
  if (phase.kind === "downloading") {
    const percent =
      phase.totalBytes > 0
        ? Math.min(100, (phase.downloadedBytes / phase.totalBytes) * 100)
        : 0;
    return (
      <div className="flex w-full max-w-sm flex-col gap-2">
        <div className="flex items-baseline justify-between gap-3 text-sm">
          <span>{t("translate.status.downloading", { model })}</span>
          <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
            {t("translate.status.downloadProgress", {
              done: formatModelBytes(phase.downloadedBytes) || "0 MB",
              total: formatModelBytes(phase.totalBytes),
            })}
          </span>
        </div>
        <Progress
          value={percent}
          indeterminate={phase.downloadedBytes <= 0}
          className="h-1.5"
        />
      </div>
    );
  }
  const text =
    phase.kind === "loading"
      ? t("translate.status.loading", { model })
      : phase.kind === "translating"
        ? phase.parts > 1
          ? t("translate.status.translatingPart", {
              part: phase.part,
              parts: phase.parts,
            })
          : t("translate.status.translating")
        : t("translate.status.preparing");
  return (
    <div className="flex items-center gap-2 text-sm text-muted-foreground">
      <Spinner className="size-3.5" />
      <span>{text}</span>
    </div>
  );
}

export function TranslatePage() {
  const t = useT();
  const locale = useLocale();
  const { prefs, source, output, phase, error, result, readyModels } =
    useTranslateStore(
      useShallow((state) => ({
        prefs: state.prefs,
        source: state.source,
        output: state.output,
        phase: state.phase,
        error: state.error,
        result: state.result,
        readyModels: state.readyModels,
      })),
    );
  const attempt = useTranslateStore((state) => state.attempt);
  const { setSource, setPrefs, setTarget, translate, stop, clear } =
    useTranslateStore(
      useShallow((state) => ({
        setSource: state.setSource,
        setPrefs: state.setPrefs,
        setTarget: state.setTarget,
        translate: state.translate,
        stop: state.stop,
        clear: state.clear,
      })),
    );
  const [historyOpen, setHistoryOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [downloaded, setDownloaded] = useState<Downloaded>({});
  const fileInput = useRef<HTMLInputElement>(null);
  const outputRef = useRef<HTMLDivElement>(null);

  const busy = phase !== null;
  const canTranslate = source.trim() !== "";
  const stale = translationIsStale({ result, source, prefs });
  const dedicated = dedicatedTranslateModel(prefs.modelId);
  const modelLabel = dedicated?.label ?? t("translate.model.chat");

  // Read again when a run ends: that is when a download has landed.
  useEffect(() => {
    if (busy) return;
    const abort = new AbortController();
    for (const model of DEDICATED_TRANSLATE_MODELS) {
      void translateModelDownloaded(model, abort.signal).then((onDisk) => {
        if (onDisk === null || abort.signal.aborted) return;
        setDownloaded((prev) => ({ ...prev, [model.id]: onDisk }));
      });
    }
    return () => abort.abort();
  }, [busy]);

  useEffect(() => {
    if (!prefs.autoTranslate || busy || !canTranslate) return;
    if (!readyModels.includes(prefs.modelId)) return;
    // Only a draft that differs from the last one sent: a stopped or failed run is not retried.
    if (
      attempt &&
      attempt.source === source &&
      attempt.target === prefs.target &&
      attempt.tone === prefs.tone &&
      attempt.modelId === prefs.modelId
    ) {
      return;
    }
    const timer = window.setTimeout(
      () => void translate(),
      AUTO_TRANSLATE_DELAY_MS,
    );
    return () => window.clearTimeout(timer);
  }, [prefs, busy, canTranslate, readyModels, attempt, source, translate]);

  // Follow the stream while it writes past the fold.
  useEffect(() => {
    const pane = outputRef.current;
    if (pane && phase?.kind === "translating")
      pane.scrollTop = pane.scrollHeight;
  }, [output, phase]);

  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(timer);
  }, [copied]);

  const loadText = (text: string) => {
    if (text.includes("\u0000")) {
      toast.error(t("translate.fileNotText"));
      return;
    }
    if (text.length > TRANSLATE_SOURCE_MAX_CHARS) {
      toast.info(
        t("translate.truncated", {
          max: TRANSLATE_SOURCE_MAX_CHARS.toLocaleString(locale),
        }),
      );
    }
    setSource(text);
  };

  const openFile = async (file: File | undefined) => {
    if (!file) return;
    try {
      // Sliced first: only the kept characters are ever read into memory.
      loadText(await file.slice(0, TRANSLATE_SOURCE_MAX_CHARS * 4 + 4).text());
    } catch {
      toast.error(t("translate.fileNotText"));
    }
  };

  const {
    ref: dropRef,
    dragging,
    dragHandlers,
  } = useNativeFileDrop({
    onFiles: (files) => void openFile(files[0]),
    accept: TEXT_FILE_EXTENSIONS,
    multiple: false,
  });

  const paste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) loadText(source + text);
    } catch {
      // Clipboard reads need a permission the user may refuse; Ctrl+V still works.
    }
  };

  const shortcut = `${IS_MAC ? "⌘" : "Ctrl"}+Enter`;

  return (
    <main className="mx-auto flex min-h-0 w-full max-w-7xl flex-1 flex-col gap-4 px-6 pb-6 pt-8 font-heading max-sm:px-4 sm:px-10">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
          {t("translate.title")}
        </h1>
        <div className="flex items-center gap-1">
          <ModelMenu
            value={prefs.modelId}
            onChange={(modelId) => setPrefs({ modelId })}
            downloaded={downloaded}
          />
          <IconButton
            icon={Clock01Icon}
            label={t("translate.history.title")}
            onClick={() => setHistoryOpen(true)}
          />
          <OptionsPopover />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full bg-muted px-3.5 py-1.5 text-sm text-muted-foreground">
          {t("translate.detectLanguage")}
        </span>
        <HugeiconsIcon
          icon={ArrowRight02Icon}
          strokeWidth={1.75}
          className="size-4 text-muted-foreground rtl:rotate-180"
          aria-hidden="true"
        />
        <LanguagePicker value={prefs.target} onChange={setTarget} />
        {prefs.recentTargets.map((code) => {
          const language = findTranslateLanguage(code);
          if (!language) return null;
          return (
            <Button
              key={code}
              type="button"
              variant="ghost"
              size="sm"
              className="rounded-full text-muted-foreground max-sm:hidden"
              onClick={() => setTarget(code)}
            >
              {translateLanguageLabel(language, locale)}
            </Button>
          );
        })}
        <div className="ml-auto">
          {busy ? (
            <Button type="button" variant="muted" onClick={stop}>
              <HugeiconsIcon
                icon={StopIcon}
                strokeWidth={1.75}
                className="size-icon"
              />
              {t("translate.stop")}
            </Button>
          ) : (
            <Button
              type="button"
              variant="dark"
              disabled={!canTranslate}
              title={`${t("translate.translate")} (${shortcut})`}
              onClick={() => void translate()}
            >
              <HugeiconsIcon
                icon={TranslateIcon}
                strokeWidth={1.75}
                className="size-icon"
              />
              {t("translate.translate")}
            </Button>
          )}
        </div>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 md:grid-cols-2">
        <section
          aria-label={t("translate.sourceLabel")}
          ref={dropRef}
          {...dragHandlers}
          className={cn(
            "relative flex min-h-64 flex-col rounded-2xl bg-muted/50 dark:bg-card",
            dragging && "ring-2 ring-ring",
          )}
        >
          <textarea
            value={source}
            onChange={(event) => setSource(event.target.value)}
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                (event.metaKey || event.ctrlKey) &&
                !busy
              ) {
                event.preventDefault();
                void translate();
              }
            }}
            maxLength={TRANSLATE_SOURCE_MAX_CHARS}
            placeholder={t("translate.sourcePlaceholder")}
            aria-label={t("translate.sourceLabel")}
            spellCheck={false}
            dir="auto"
            className="min-h-0 w-full flex-1 resize-none bg-transparent px-5 pt-4 text-ui-15 leading-relaxed outline-none placeholder:text-muted-foreground"
          />
          <div className="flex items-center gap-1 px-3 pb-2 pt-1">
            <IconButton
              icon={Upload01Icon}
              label={t("translate.openFile")}
              onClick={() => fileInput.current?.click()}
            />
            <IconButton
              icon={ClipboardPasteIcon}
              label={t("translate.paste")}
              onClick={() => void paste()}
            />
            {source !== "" && (
              <IconButton
                icon={Cancel01Icon}
                label={t("translate.clear")}
                onClick={clear}
              />
            )}
            <span className="ml-auto pr-2 text-xs tabular-nums text-muted-foreground">
              {source.length.toLocaleString(locale)} /{" "}
              {TRANSLATE_SOURCE_MAX_CHARS.toLocaleString(locale)}
            </span>
          </div>
          <input
            ref={fileInput}
            type="file"
            accept={`text/*,${TEXT_FILE_EXTENSIONS}`}
            className="hidden"
            onChange={(event) => {
              void openFile(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
          {dragging && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-2xl bg-background/80 text-sm text-muted-foreground">
              {t("translate.dropFile")}
            </div>
          )}
        </section>

        <section
          aria-label={t("translate.outputLabel")}
          aria-busy={busy}
          className="flex min-h-64 flex-col rounded-2xl bg-muted/50 dark:bg-card"
        >
          <div
            ref={outputRef}
            className="min-h-0 flex-1 overflow-y-auto px-5 pt-4"
          >
            {output !== "" ? (
              <p
                dir="auto"
                aria-live="polite"
                className={cn(
                  "whitespace-pre-wrap break-words text-ui-15 leading-relaxed [overflow-wrap:anywhere]",
                  stale && "text-muted-foreground",
                )}
              >
                {output}
              </p>
            ) : phase && phase.kind !== "translating" ? (
              <div className="flex h-full items-center justify-center pb-4">
                <PhaseStatus phase={phase} model={modelLabel} />
              </div>
            ) : error ? (
              <div className="flex h-full flex-col items-center justify-center gap-3 pb-4 text-center">
                <div className="flex max-w-md flex-col gap-1">
                  <span className="text-sm font-medium">
                    {t("translate.errors.title")}
                  </span>
                  <span className="text-sm text-muted-foreground [overflow-wrap:anywhere]">
                    {error.code === "other"
                      ? error.message
                      : t(`translate.errors.${error.code}`)}
                  </span>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => void translate()}
                >
                  {t("translate.errors.retry")}
                </Button>
              </div>
            ) : (
              <div className="flex flex-col gap-2 text-ui-15 leading-relaxed text-muted-foreground">
                <span>
                  {phase
                    ? t("translate.status.translating")
                    : t("translate.outputEmpty")}
                </span>
                {!phase && dedicated && downloaded[dedicated.id] === false && (
                  <span className="text-sm">
                    {t("translate.firstRunHint", {
                      model: dedicated.label,
                      size: formatModelBytes(dedicated.sizeBytes),
                    })}
                  </span>
                )}
              </div>
            )}
          </div>
          <div className="flex min-h-11 items-center gap-1 px-3 pb-2 pt-1">
            {output !== "" && phase && (
              <div className="pl-2">
                <PhaseStatus phase={phase} model={modelLabel} />
              </div>
            )}
            {output !== "" && !phase && stale && (
              <span className="pl-2 text-xs text-muted-foreground">
                {t("translate.stale")}
              </span>
            )}
            <div className="ml-auto flex items-center gap-1">
              {output !== "" && !busy && (
                <IconButton
                  icon={ArrowDataTransferHorizontalIcon}
                  label={t("translate.useAsSource")}
                  onClick={() => setSource(output)}
                />
              )}
              {output !== "" && (
                <IconButton
                  icon={copied ? Tick02Icon : Copy01Icon}
                  label={copied ? t("translate.copied") : t("translate.copy")}
                  onClick={() => void copyToClipboard(output).then(setCopied)}
                />
              )}
            </div>
          </div>
        </section>
      </div>

      <TranslateHistorySheet open={historyOpen} onOpenChange={setHistoryOpen} />
    </main>
  );
}
