// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronLeftStandardIcon } from "@/lib/chevron-icons";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import {
  Delete02Icon,
  PauseIcon,
  PlayIcon,
  Search01Icon,
  Tick02Icon,
  ViewIcon,
} from "@hugeicons/core-free-icons";
import { Copy01Icon } from "@/lib/copy-icon";
import { MessageCircleIcon } from "@/lib/hugeicons-derived";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  type RecentDictation,
  useVoiceSettingsStore,
} from "../stores/voice-settings-store";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

const PAGE_SIZE = 20;

type SortOrder = "newest" | "oldest" | "az";

function formatDictationDate(ms: number): string {
  return new Date(ms).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDictationDateTime(ms: number): string {
  return new Date(ms).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** `m:ss`, used for both the seek-bar position and the recording length. */
function formatElapsed(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0:00";
  const total = Math.floor(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

interface DiffToken {
  text: string;
  changed: boolean;
}

function tokenizeTranscript(text: string): string[] {
  return text.match(/\s+|[^\s]+/g) ?? [];
}

/** Word/punctuation diff used to highlight what the fixer changed. */
function diffTranscript(original: string, corrected: string): DiffToken[] {
  const before = tokenizeTranscript(original);
  const after = tokenizeTranscript(corrected);
  const rows = before.length;
  const cols = after.length;
  const lcs = Array.from({ length: rows + 1 }, () =>
    new Uint32Array(cols + 1),
  );
  for (let i = rows - 1; i >= 0; i -= 1) {
    for (let j = cols - 1; j >= 0; j -= 1) {
      lcs[i][j] =
        before[i] === after[j]
          ? lcs[i + 1][j + 1] + 1
          : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const result: DiffToken[] = [];
  let i = 0;
  let j = 0;
  while (i < rows && j < cols) {
    if (before[i] === after[j]) {
      result.push({ text: after[j], changed: false });
      i += 1;
      j += 1;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      i += 1;
    } else {
      result.push({
        text: after[j],
        changed: after[j].trim().length > 0,
      });
      j += 1;
    }
  }
  while (j < cols) {
    result.push({ text: after[j], changed: after[j].trim().length > 0 });
    j += 1;
  }
  return result;
}

/**
 * Compact recording player: play/pause, a short seek bar and elapsed/total
 * time. Replaces the native audio controls, whose chunky chrome broke the
 * settings page and hid the recording length.
 */
function DictationAudioPlayer({ src }: { src: string }) {
  const t = useT();
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    setCurrent(0);
    setDuration(0);
    setPlaying(false);
    const audio = new Audio(src);
    audio.preload = "metadata";
    audioRef.current = audio;
    const onLoaded = () => {
      if (Number.isFinite(audio.duration)) setDuration(audio.duration);
    };
    const onTime = () => setCurrent(audio.currentTime);
    const onEnded = () => {
      setPlaying(false);
      setCurrent(0);
    };
    const onPause = () => setPlaying(false);
    const onPlay = () => setPlaying(true);
    audio.addEventListener("loadedmetadata", onLoaded);
    audio.addEventListener("timeupdate", onTime);
    audio.addEventListener("ended", onEnded);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("play", onPlay);
    return () => {
      audio.pause();
      audio.removeEventListener("loadedmetadata", onLoaded);
      audio.removeEventListener("timeupdate", onTime);
      audio.removeEventListener("ended", onEnded);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("play", onPlay);
      audioRef.current = null;
    };
  }, [src]);

  const toggle = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      void audio.play().catch(() => setPlaying(false));
    } else {
      audio.pause();
    }
  };

  const seek = (value: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    const next = Math.min(Math.max(value, 0), audio.duration || value);
    audio.currentTime = next;
    setCurrent(next);
  };

  return (
    <div
      className="flex w-full items-center gap-3 rounded-xl border border-border/60 bg-muted/20 p-3"
      data-testid="dictation-audio-player"
    >
      <button
        type="button"
        onClick={toggle}
        aria-label={
          playing
            ? t("settings.voice.recents.pauseAudio")
            : t("settings.voice.recents.playAudio")
        }
        className="inline-flex size-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <HugeiconsIcon
          icon={playing ? PauseIcon : PlayIcon}
          strokeWidth={2}
          className="size-4"
        />
      </button>
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <span className="w-10 shrink-0 text-right text-ui-10 tabular-nums text-muted-foreground">
          {formatElapsed(current)}
        </span>
        <Slider
          value={[current]}
          min={0}
          max={Math.max(duration, current, 0.1)}
          step={0.05}
          onValueChange={([value]) => {
            if (value !== undefined) seek(value);
          }}
          aria-label={t("settings.voice.recents.seekLabel")}
          className="min-w-0 flex-1"
        />
        <span className="w-10 shrink-0 text-left text-ui-10 tabular-nums text-muted-foreground">
          {formatElapsed(duration)}
        </span>
      </div>
    </div>
  );
}

export function RecentDictationsView({
  selectedId,
  onSelect,
  onBack,
}: {
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onBack: () => void;
}) {
  const t = useT();
  const recentDictations = useVoiceSettingsStore((s) => s.recentDictations);
  const removeRecentDictation = useVoiceSettingsStore(
    (s) => s.removeRecentDictation,
  );
  const clearRecentDictations = useVoiceSettingsStore(
    (s) => s.clearRecentDictations,
  );
  const [clearAllPending, setClearAllPending] = useState(false);
  const [search, setSearch] = useState("");
  const [sortOrder, setSortOrder] = useState<SortOrder>("newest");
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [showOriginal, setShowOriginal] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const copyTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (copyTimerRef.current !== null) {
        window.clearTimeout(copyTimerRef.current);
      }
    };
  }, []);

  // Opening the detail view stops a row preview so only one recording plays.
  useEffect(() => {
    if (!selectedId) return;
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setPlayingId(null);
    setShowOriginal(false);
  }, [selectedId]);

  function togglePlayAudio(dictation: RecentDictation) {
    if (!dictation.audioDataUrl) return;
    if (playingId === dictation.id && audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
      setPlayingId(null);
      return;
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    const audio = new Audio(dictation.audioDataUrl);
    audioRef.current = audio;
    setPlayingId(dictation.id);
    audio.addEventListener("ended", () => {
      if (audioRef.current === audio) {
        audioRef.current = null;
        setPlayingId(null);
      }
    });
    audio.addEventListener("error", () => {
      if (audioRef.current === audio) {
        audioRef.current = null;
        setPlayingId(null);
      }
    });
    void audio.play().catch(() => {
      if (audioRef.current === audio) {
        audioRef.current = null;
        setPlayingId(null);
      }
    });
  }
  // Keyed on the inputs so a change restarts from page one without an effect.
  const pageKey = `${sortOrder}::${search}`;
  const [page, setPage] = useState({ key: pageKey, count: PAGE_SIZE });
  const visibleCount = page.key === pageKey ? page.count : PAGE_SIZE;
  const navigate = useNavigate();
  const closeSettings = useSettingsDialogStore((s) => s.closeDialog);
  const selected =
    recentDictations.find((dictation) => dictation.id === selectedId) ?? null;
  const diff = useMemo(
    () =>
      selected?.originalText
        ? diffTranscript(selected.originalText, selected.text)
        : null,
    [selected?.id, selected?.text, selected?.originalText],
  );
  const stats = useMemo(() => {
    const text = selected?.text ?? "";
    return {
      words: text.trim() ? text.trim().split(/\s+/).length : 0,
      characters: Array.from(text.replace(/\s/g, "")).length,
      spaces: (text.match(/\s/g) ?? []).length,
      symbols: (text.match(/[^\p{L}\p{N}\s]/gu) ?? []).length,
    };
  }, [selected?.text]);

  function openChat(chatId: string) {
    closeSettings();
    void navigate({ to: "/chat", search: { thread: chatId } });
  }

  const visible = useMemo(() => {
    const query = search.trim().toLowerCase();
    const filtered = query
      ? recentDictations.filter((d) => d.text.toLowerCase().includes(query))
      : recentDictations;
    const sorted = [...filtered];
    if (sortOrder === "oldest") {
      sorted.sort((a, b) => a.at - b.at);
    } else if (sortOrder === "az") {
      sorted.sort((a, b) => a.text.localeCompare(b.text));
    } else {
      sorted.sort((a, b) => b.at - a.at);
    }
    return sorted;
  }, [recentDictations, search, sortOrder]);

  function markCopied(id: string) {
    setCopiedId(id);
    if (copyTimerRef.current !== null) {
      window.clearTimeout(copyTimerRef.current);
    }
    copyTimerRef.current = window.setTimeout(() => {
      copyTimerRef.current = null;
      setCopiedId(null);
    }, 1500);
  }

  async function handleCopy(id: string, text: string) {
    if (await copyToClipboard(text)) {
      markCopied(id);
      toast.success(t("settings.voice.recents.copied"));
    } else {
      toast.error(t("settings.voice.recents.copyFailed"));
    }
  }

  function deleteOne(dictation: RecentDictation) {
    removeRecentDictation(dictation.id);
    if (dictation.id === selectedId) {
      onSelect(null);
    }
    toast.success(t("settings.voice.recents.deleted"));
  }

  function clearAll() {
    clearRecentDictations();
    onSelect(null);
    setClearAllPending(false);
  }

  return (
    <div className="settings-page">
      <header className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => (selected ? onSelect(null) : onBack())}
          aria-label={
            selected
              ? t("settings.voice.recents.backToRecents")
              : t("settings.voice.recents.backToVoice")
          }
          className="settings-back-button inline-flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <HugeiconsIcon icon={ChevronLeftStandardIcon} className="size-4 rtl:rotate-180" />
        </button>
        <h1 className="font-heading text-xl font-semibold">
          {t("settings.voice.title")}
        </h1>
      </header>

      <div className="flex flex-col gap-1">
        {selected ? (
          <div className="flex items-baseline justify-between gap-4">
            <h2 className="settings-heading text-sm font-semibold">
              {t("settings.voice.recents.detailTitle")}
            </h2>
            <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
              {formatDictationDateTime(selected.at)}
            </span>
          </div>
        ) : (
          <>
            <h2 className="settings-heading text-sm font-semibold">
              {t("settings.voice.recents.sectionTitle")}
            </h2>
            <p className="text-xs text-muted-foreground">
              {t("settings.voice.recents.pageDescription")}
            </p>
          </>
        )}
      </div>

      {selected ? (
        <div className="flex min-h-0 flex-col gap-4">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span>
              {t("settings.voice.recents.metaWords", { count: stats.words })}
            </span>
            <span>
              {t("settings.voice.recents.metaCharacters", {
                count: stats.characters,
              })}
            </span>
            <span>
              {t("settings.voice.recents.metaSpaces", { count: stats.spaces })}
            </span>
            <span>
              {t("settings.voice.recents.metaSymbols", {
                count: stats.symbols,
              })}
            </span>
            <span>
              {t("settings.voice.recents.metaFixes", {
                count: selected.fixes ?? 0,
              })}
            </span>
            {selected.chatId ? (
              <span className="inline-flex items-center gap-1">
                <HugeiconsIcon
                  icon={MessageCircleIcon}
                  className="size-3"
                />
                {t("settings.voice.recents.metaLinkedChat")}
              </span>
            ) : null}
            {selected.originalText && diff ? (
              <div className="ml-auto inline-flex items-center rounded-full border border-border/60 p-0.5 text-xs">
                <button
                  type="button"
                  onClick={() => setShowOriginal(false)}
                  className={`rounded-full px-2.5 py-1 transition-colors ${
                    !showOriginal
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {t("settings.voice.recents.corrected")}
                </button>
                <button
                  type="button"
                  onClick={() => setShowOriginal(true)}
                  className={`rounded-full px-2.5 py-1 transition-colors ${
                    showOriginal
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {t("settings.voice.recents.original")}
                </button>
              </div>
            ) : null}
          </div>
          {selected.audioDataUrl ? (
            <DictationAudioPlayer src={selected.audioDataUrl} />
          ) : null}
          <article className="max-h-72 overflow-y-auto rounded-xl border border-border/60 bg-muted/20 p-4">
            {showOriginal || !diff ? (
              <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground select-text">
                {showOriginal ? selected.originalText : selected.text}
              </p>
            ) : (
              <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground select-text">
                {diff.map((token, index) =>
                  token.changed ? (
                    <mark
                      key={index}
                      className="rounded-sm bg-emerald-500/20 px-0.5 text-emerald-800 dark:text-emerald-200"
                    >
                      {token.text}
                    </mark>
                  ) : (
                    <span key={index}>{token.text}</span>
                  ),
                )}
              </p>
            )}
          </article>
          <div className="flex justify-end gap-2">
            {selected.chatId ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => openChat(selected.chatId as string)}
              >
                <HugeiconsIcon
                  icon={MessageCircleIcon}
                  className="mr-1.5 size-3.5"
                />
                {t("settings.voice.recents.openChat")}
              </Button>
            ) : null}
            <Button
              variant="outline"
              size="sm"
              onClick={async () => {
                await handleCopy(selected.id, selected.text);
              }}
            >
              <HugeiconsIcon
                icon={copiedId === selected.id ? Tick02Icon : Copy01Icon}
                className="mr-1.5 size-3.5"
              />
              {copiedId === selected.id
                ? t("settings.voice.recents.copiedShort")
                : t("settings.voice.recents.copy")}
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => deleteOne(selected)}
              className="text-destructive hover:text-destructive"
            >
              <HugeiconsIcon icon={Delete02Icon} className="mr-1.5 size-3.5" />
              {t("settings.voice.recents.delete")}
            </Button>
          </div>
        </div>
      ) : recentDictations.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          {t("settings.voice.recents.empty")}
        </p>
      ) : (
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <InputGroup className="h-control flex-1">
              <InputGroupAddon align="inline-start">
                <HugeiconsIcon
                  icon={Search01Icon}
                  strokeWidth={2}
                  className="size-3.5 text-muted-foreground"
                />
              </InputGroupAddon>
              <InputGroupInput
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder={t("settings.voice.recents.searchPlaceholder")}
                aria-label={t("settings.voice.recents.searchPlaceholder")}
                className="text-sm"
              />
            </InputGroup>
            <Select
              value={sortOrder}
              onValueChange={(value) => setSortOrder(value as SortOrder)}
            >
              <SelectTrigger
                className="w-36 shrink-0 data-[size=default]:h-control"
                aria-label={t("settings.voice.recents.sortLabel")}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="newest">
                  {t("settings.voice.recents.sortNewest")}
                </SelectItem>
                <SelectItem value="oldest">
                  {t("settings.voice.recents.sortOldest")}
                </SelectItem>
                <SelectItem value="az">
                  {t("settings.voice.recents.sortAlpha")}
                </SelectItem>
              </SelectContent>
            </Select>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => setClearAllPending(true)}
              className="shrink-0 text-destructive hover:text-destructive"
            >
              <HugeiconsIcon icon={Delete02Icon} className="mr-1.5 size-3.5" />
              {t("settings.voice.recents.clear")}
            </Button>
          </div>

          {visible.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              {t("settings.voice.recents.noMatches")}
            </p>
          ) : (
            <div className="flex flex-col">
              <div className="flex items-center gap-2 border-b border-border/60 px-1 pb-2 text-xs font-semibold text-foreground">
                <div className="flex min-w-0 flex-1 items-center gap-4">
                  <span className="min-w-0 flex-1">
                    {t("settings.voice.recents.dictationColumn")}
                  </span>
                  <span className="hidden w-40 shrink-0 items-center gap-1.5 sm:flex">
                    {/* Spacer matching the row's chat-link icon slot so the
                        header starts exactly where the dates start. */}
                    <span className="size-3.5 shrink-0" />
                    {t("settings.voice.recents.dateColumn")}
                  </span>
                </div>
                <span className="w-[calc(7.75rem*var(--ui-space-scale,1))] shrink-0" />
              </div>
              {visible.slice(0, visibleCount).map((dictation) => (
                <div
                  key={dictation.id}
                  className="group flex items-stretch gap-2 border-b border-border/40 transition-colors last:border-0 hover:bg-muted/50"
                >
                  <button
                    type="button"
                    onClick={() => onSelect(dictation.id)}
                    aria-label={t("settings.voice.recents.view")}
                    className="flex min-w-0 flex-1 items-start gap-4 px-1 py-3 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <span className="min-w-0 flex-1">
                      <span className="line-clamp-1 whitespace-pre-wrap break-words text-sm text-foreground">
                        {dictation.text}
                      </span>
                      <span className="mt-1 flex items-center gap-1.5 text-xs text-muted-foreground sm:hidden">
                        {dictation.chatId ? (
                          <HugeiconsIcon
                            icon={MessageCircleIcon}
                            className="size-3"
                          />
                        ) : null}
                        {formatDictationDate(dictation.at)}
                      </span>
                    </span>
                    <span className="hidden w-40 shrink-0 items-center gap-1.5 text-left text-xs text-muted-foreground tabular-nums sm:flex">
                      {/* Fixed icon slot keeps every date starting at the
                          same column whether or not a chat is linked. */}
                      <span className="flex size-3.5 shrink-0 items-center justify-center">
                        {dictation.chatId ? (
                          <HugeiconsIcon
                            icon={MessageCircleIcon}
                            className="size-3.5"
                            aria-label={t("settings.voice.recents.openChat")}
                          />
                        ) : null}
                      </span>
                      <span className="whitespace-nowrap">
                        {formatDictationDate(dictation.at)}
                      </span>
                    </span>
                  </button>
                  <span className="grid w-[calc(7.75rem*var(--ui-space-scale,1))] shrink-0 grid-cols-4 items-center">
                    <span className="flex items-center justify-center">
                      {dictation.audioDataUrl ? (
                        <button
                          type="button"
                          onClick={() => togglePlayAudio(dictation)}
                          aria-label={
                            playingId === dictation.id
                              ? t("settings.voice.recents.pauseAudio")
                              : t("settings.voice.recents.playAudio")
                          }
                          title={
                            playingId === dictation.id
                              ? t("settings.voice.recents.pauseAudio")
                              : t("settings.voice.recents.playAudio")
                          }
                          className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        >
                          <HugeiconsIcon
                            icon={
                              playingId === dictation.id ? PauseIcon : PlayIcon
                            }
                            strokeWidth={1.75}
                            className="size-4"
                          />
                        </button>
                      ) : null}
                    </span>
                    <span className="flex items-center justify-center">
                      <button
                        type="button"
                        onClick={() => onSelect(dictation.id)}
                        aria-label={t("settings.voice.recents.view")}
                        title={t("settings.voice.recents.view")}
                        className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      >
                        <HugeiconsIcon
                          icon={ViewIcon}
                          strokeWidth={1.75}
                          className="size-4"
                        />
                      </button>
                    </span>
                    <span className="flex items-center justify-center">
                      <button
                        type="button"
                        onClick={async () => {
                          await handleCopy(dictation.id, dictation.text);
                        }}
                        aria-label={t("settings.voice.recents.copy")}
                        title={t("settings.voice.recents.copy")}
                        className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      >
                        <HugeiconsIcon
                          icon={copiedId === dictation.id ? Tick02Icon : Copy01Icon}
                          strokeWidth={1.75}
                          className="size-4"
                        />
                      </button>
                    </span>
                    <span className="flex items-center justify-center">
                      <button
                        type="button"
                        onClick={() => deleteOne(dictation)}
                        aria-label={t("settings.voice.recents.delete")}
                        title={t("settings.voice.recents.delete")}
                        className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      >
                        <HugeiconsIcon
                          icon={Delete02Icon}
                          strokeWidth={1.75}
                          className="size-4"
                        />
                      </button>
                    </span>
                  </span>
                </div>
              ))}
            </div>
          )}

          {visible.length > visibleCount ? (
            <div className="flex justify-center pt-1">
              <Button
                variant="outline"
                size="sm"
                onClick={() =>
                  setPage({ key: pageKey, count: visibleCount + PAGE_SIZE })
                }
              >
                {t("settings.voice.recents.showMore", {
                  count: visible.length - visibleCount,
                })}
              </Button>
            </div>
          ) : null}
        </div>
      )}

      <AlertDialog
        open={clearAllPending}
        onOpenChange={(open) => {
          if (!open) setClearAllPending(false);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("settings.voice.recents.clearTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t("settings.voice.recents.clearDescription")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={(event) => {
                event.preventDefault();
                clearAll();
              }}
            >
              {t("settings.voice.recents.clearConfirm")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
