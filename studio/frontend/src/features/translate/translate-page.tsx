// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  composerDraftKey,
  latestDraft,
  publishDraft,
  useChatRuntimeStore,
  writeComposerDraft,
} from "@/features/chat";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import {
  ArrowDataTransferHorizontalIcon,
  BubbleChatIcon,
  Copy01Icon,
  Delete02Icon,
  Download01Icon,
  TranslateIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Link, useNavigate } from "@tanstack/react-router";
import {
  type ReactNode,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";
import { TranslateLengthError, streamTranslation } from "./translate-api";
import {
  TRANSLATE_HISTORY_FILE_NAME,
  TRANSLATE_LANGUAGES,
  TRANSLATE_MODES,
  TRANSLATE_TONES,
  type TranslateMode,
  type TranslateTone,
  type TranslationEntry,
  buildChatHandoff,
  buildTranslateSystemPrompt,
  exportTranslateHistory,
  isTranslateLanguage,
  isTranslateMode,
  isTranslateSource,
  isTranslateTone,
} from "./translate-prompt";
import { useTranslateStore } from "./translate-store";

/** How many of the newest translations "Chat with translated text" carries into the chat. */
const CHAT_HANDOFF_ENTRIES = 5;
const AUTO = "auto";

const MODE_LABEL = {
  automatic: "translate.modes.automatic",
  literal: "translate.modes.literal",
  semantic: "translate.modes.semantic",
} as const satisfies Record<TranslateMode, TranslationKey>;
const TONE_LABEL = {
  balanced: "translate.tones.balanced",
  formal: "translate.tones.formal",
  casual: "translate.tones.casual",
} as const satisfies Record<TranslateTone, TranslationKey>;

// The picker reads in the UI language; the prompt always names languages in English.
function useLanguageLabel(): (code: string) => string {
  const locale = useLocale();
  return useMemo(() => {
    let names: Intl.DisplayNames | null = null;
    try {
      names = new Intl.DisplayNames([locale, "en"], { type: "language" });
    } catch {
      // Older engines without DisplayNames fall back to the English names.
    }
    return (code: string) =>
      names?.of(code) ??
      TRANSLATE_LANGUAGES.find((language) => language.code === code)?.name ??
      code;
  }, [locale]);
}

function modelName(model: string): string {
  return model.split(/[\\/]/).filter(Boolean).pop() ?? model;
}

export function TranslatePage() {
  const t = useT();
  const locale = useLocale();
  const navigate = useNavigate();
  const languageLabel = useLanguageLabel();
  const inputId = useId();
  const promptToggleId = useId();
  const {
    source,
    target,
    mode,
    tone,
    showSystemPrompt,
    history,
    setSource,
    setTarget,
    setMode,
    setTone,
    setShowSystemPrompt,
    addEntry,
    removeEntry,
    clearHistory,
  } = useTranslateStore(useShallow((state) => state));
  const { checkpoint, modelLoading } = useChatRuntimeStore(
    useShallow((state) => ({
      checkpoint: state.params.checkpoint,
      modelLoading: state.modelLoading,
    })),
  );
  const model = modelLoading ? "" : checkpoint;
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const systemPrompt = useMemo(
    () => buildTranslateSystemPrompt({ source, target, mode, tone }),
    [source, target, mode, tone],
  );
  const canTranslate = !running && !!model && input.trim().length > 0;
  const formatTime = useMemo(() => {
    const format = new Intl.DateTimeFormat(locale, {
      dateStyle: "medium",
      timeStyle: "short",
    });
    return (at: number) => format.format(at);
  }, [locale]);

  async function translate() {
    if (!canTranslate) return;
    const text = input.trim();
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setOutput("");
    try {
      const result = await streamTranslation({
        checkpoint: model,
        systemPrompt,
        text,
        signal: controller.signal,
        onText: setOutput,
      });
      setOutput(result);
      if (result) {
        addEntry({
          id: crypto.randomUUID(),
          createdAt: Date.now(),
          source,
          target,
          mode,
          tone,
          input: text,
          output: result,
          model,
        });
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      toast.error(
        t(
          error instanceof TranslateLengthError
            ? "translate.errors.cutOff"
            : "translate.errors.failed",
        ),
        {
          description:
            error instanceof TranslateLengthError
              ? undefined
              : errorText(error),
        },
      );
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setRunning(false);
    }
  }

  function swapLanguages() {
    if (source === AUTO) return;
    setSource(target);
    setTarget(source);
    if (output) {
      setInput(output);
      setOutput("");
    }
  }

  async function copyOutput() {
    if (await copyToClipboard(output)) toast.success(t("translate.copied"));
  }

  async function exportHistory() {
    try {
      await downloadFile(
        exportTranslateHistory(history),
        TRANSLATE_HISTORY_FILE_NAME,
        "text/plain",
      );
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error(t("translate.errors.exportFailed"), {
          description: errorText(error),
        });
      }
    }
  }

  function chatWithTranslations(entries: readonly TranslationEntry[]) {
    const handoff = buildChatHandoff(entries);
    if (!handoff) return;
    // The composer draft follows the user into the new chat; an unsent draft stays after it.
    const draft = latestDraft();
    const text = draft.trim() ? `${handoff}${draft}` : handoff;
    writeComposerDraft(composerDraftKey(), text);
    publishDraft(text, TRANSLATE_DRAFT_SOURCE);
    void navigate({ to: "/chat", search: { new: crypto.randomUUID() } });
  }

  return (
    <main className="mx-auto w-full max-w-5xl 4xl:max-w-6xl px-6 pb-10 pt-8 max-sm:px-4 font-heading sm:px-10">
      <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
        {t("translate.title")}
      </h1>
      <p className="mt-4 text-ui-15 text-muted-foreground">
        {t("translate.description")}
      </p>

      <div className="mt-8 flex flex-wrap items-end gap-3">
        <PickerField label={t("translate.source")}>
          <Select
            value={source}
            onValueChange={(value) =>
              isTranslateSource(value) && setSource(value)
            }
          >
            <SelectTrigger
              size="sm"
              className="w-44"
              aria-label={t("translate.source")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={AUTO}>{t("translate.autoDetect")}</SelectItem>
              {TRANSLATE_LANGUAGES.map((language) => (
                <SelectItem key={language.code} value={language.code}>
                  {languageLabel(language.code)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </PickerField>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={swapLanguages}
          disabled={source === AUTO || running}
          aria-label={t("translate.swap")}
          title={t("translate.swap")}
        >
          <HugeiconsIcon
            icon={ArrowDataTransferHorizontalIcon}
            strokeWidth={1.75}
          />
        </Button>
        <PickerField label={t("translate.target")}>
          <Select
            value={target}
            onValueChange={(value) =>
              isTranslateLanguage(value) && setTarget(value)
            }
          >
            <SelectTrigger
              size="sm"
              className="w-44"
              aria-label={t("translate.target")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TRANSLATE_LANGUAGES.map((language) => (
                <SelectItem key={language.code} value={language.code}>
                  {languageLabel(language.code)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </PickerField>
        <PickerField label={t("translate.mode")}>
          <Select
            value={mode}
            onValueChange={(value) => isTranslateMode(value) && setMode(value)}
          >
            <SelectTrigger
              size="sm"
              className="w-36"
              aria-label={t("translate.mode")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TRANSLATE_MODES.map((option) => (
                <SelectItem key={option} value={option}>
                  {t(MODE_LABEL[option])}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </PickerField>
        <PickerField label={t("translate.tone")}>
          <Select
            value={tone}
            onValueChange={(value) => isTranslateTone(value) && setTone(value)}
          >
            <SelectTrigger
              size="sm"
              className="w-36"
              aria-label={t("translate.tone")}
            >
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
        </PickerField>
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <Textarea
          id={inputId}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
              event.preventDefault();
              void translate();
            }
          }}
          placeholder={t("translate.inputPlaceholder")}
          aria-label={t("translate.inputPlaceholder")}
          fieldSizing="fixed"
          className="h-56"
        />
        <div
          aria-live="polite"
          className="relative h-56 overflow-y-auto whitespace-pre-wrap break-words rounded-xl bg-muted px-3.5 py-3 text-base md:text-sm dark:bg-card"
        >
          {output || (
            <span className="text-muted-foreground">
              {t("translate.outputPlaceholder")}
            </span>
          )}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        {running ? (
          <Button variant="outline" onClick={() => abortRef.current?.abort()}>
            <Spinner className="size-icon" />
            {t("translate.stop")}
          </Button>
        ) : (
          <Button onClick={() => void translate()} disabled={!canTranslate}>
            <HugeiconsIcon
              icon={TranslateIcon}
              strokeWidth={1.75}
              className="size-icon"
            />
            {t("translate.translate")}
          </Button>
        )}
        <Button
          variant="ghost"
          onClick={() => void copyOutput()}
          disabled={!output || running}
        >
          <HugeiconsIcon
            icon={Copy01Icon}
            strokeWidth={1.75}
            className="size-icon"
          />
          {t("translate.copy")}
        </Button>
        <span className="min-w-0 flex-1 truncate text-right text-ui-13 text-muted-foreground">
          {model ? (
            t("translate.usingModel", { model: modelName(model) })
          ) : (
            <>
              {t("translate.noModel")}{" "}
              <Link to="/chat" className="underline underline-offset-2">
                {t("translate.openChat")}
              </Link>
            </>
          )}
        </span>
      </div>

      <label
        htmlFor={promptToggleId}
        className="mt-6 flex w-fit cursor-pointer items-center gap-2.5 text-ui-14 text-muted-foreground"
      >
        <Switch
          id={promptToggleId}
          size="sm"
          checked={showSystemPrompt}
          onCheckedChange={(checked) => setShowSystemPrompt(checked === true)}
        />
        {t("translate.showSystemPrompt")}
      </label>
      {showSystemPrompt && (
        <pre className="mt-3 overflow-x-auto whitespace-pre-wrap rounded-xl bg-muted px-3.5 py-3 font-mono text-ui-13 text-foreground/80 dark:bg-card">
          {systemPrompt}
        </pre>
      )}

      <section className="mt-10">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-ui-15 font-semibold text-foreground">
            {t("translate.history")}
          </h2>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                chatWithTranslations(history.slice(0, CHAT_HANDOFF_ENTRIES))
              }
              disabled={history.length === 0}
            >
              <HugeiconsIcon
                icon={BubbleChatIcon}
                strokeWidth={1.75}
                className="size-icon"
              />
              {t("translate.chatWithTranslation")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => void exportHistory()}
              disabled={history.length === 0}
            >
              <HugeiconsIcon
                icon={Download01Icon}
                strokeWidth={1.75}
                className="size-icon"
              />
              {t("translate.exportHistory")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={clearHistory}
              disabled={history.length === 0}
            >
              <HugeiconsIcon
                icon={Delete02Icon}
                strokeWidth={1.75}
                className="size-icon"
              />
              {t("translate.clearHistory")}
            </Button>
          </div>
        </div>
        {history.length === 0 ? (
          <p className="mt-4 text-sm text-muted-foreground">
            {t("translate.historyEmpty")}
          </p>
        ) : (
          <ul className="mt-4 flex flex-col gap-2">
            {history.map((entry) => (
              <li
                key={entry.id}
                className="group/entry rounded-xl px-4 py-3 hover:bg-muted/60"
              >
                <div className="flex items-center gap-2 text-ui-13 text-muted-foreground">
                  <span className="min-w-0 flex-1 truncate">
                    {entry.source === AUTO
                      ? t("translate.autoDetect")
                      : languageLabel(entry.source)}{" "}
                    → {languageLabel(entry.target)} ·{" "}
                    {t(MODE_LABEL[entry.mode])} · {t(TONE_LABEL[entry.tone])} ·{" "}
                    {formatTime(entry.createdAt)}
                  </span>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    className="opacity-0 group-hover/entry:opacity-100 focus-visible:opacity-100"
                    onClick={() => chatWithTranslations([entry])}
                    aria-label={t("translate.chatWithTranslation")}
                    title={t("translate.chatWithTranslation")}
                  >
                    <HugeiconsIcon icon={BubbleChatIcon} strokeWidth={1.75} />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    className="opacity-0 group-hover/entry:opacity-100 focus-visible:opacity-100"
                    onClick={() => removeEntry(entry.id)}
                    aria-label={t("translate.delete")}
                    title={t("translate.delete")}
                  >
                    <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} />
                  </Button>
                </div>
                <p className="mt-1 line-clamp-2 whitespace-pre-wrap break-words text-sm text-muted-foreground">
                  {entry.input}
                </p>
                <p className="mt-1 whitespace-pre-wrap break-words text-sm text-foreground">
                  {entry.output}
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}

const TRANSLATE_DRAFT_SOURCE = Symbol("translate");

function PickerField({
  label,
  children,
}: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-ui-13 text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}

function errorText(error: unknown): string | undefined {
  return error instanceof Error ? error.message : undefined;
}
