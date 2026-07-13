// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import {
  ArrowLeft01Icon,
  Copy01Icon,
  Delete02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import {
  type RecentDictation,
  useVoiceSettingsStore,
} from "../stores/voice-settings-store";

type PendingDelete =
  | { kind: "one"; dictation: RecentDictation }
  | { kind: "all" }
  | null;

function formatDictationDate(ms: number): string {
  return new Date(ms).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
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
  const [pendingDelete, setPendingDelete] = useState<PendingDelete>(null);
  const selected =
    recentDictations.find((dictation) => dictation.id === selectedId) ?? null;

  async function handleCopy(text: string) {
    if (await copyToClipboard(text)) {
      toast.success(t("settings.voice.recents.copied"));
    } else {
      toast.error(t("settings.voice.recents.copyFailed"));
    }
  }

  function confirmDelete() {
    if (pendingDelete?.kind === "one") {
      removeRecentDictation(pendingDelete.dictation.id);
      if (pendingDelete.dictation.id === selectedId) {
        onSelect(null);
      }
    } else if (pendingDelete?.kind === "all") {
      clearRecentDictations();
      onSelect(null);
    }
    setPendingDelete(null);
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => (selected ? onSelect(null) : onBack())}
          aria-label={
            selected
              ? t("settings.voice.recents.backToRecents")
              : t("settings.voice.recents.backToVoice")
          }
          className="inline-flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <HugeiconsIcon icon={ArrowLeft01Icon} className="size-4" />
        </button>
        <h1 className="font-heading text-xl font-semibold">
          {t("settings.voice.title")}
        </h1>
      </header>

      <div className="flex flex-col gap-1">
        <h2 className="text-sm font-semibold">
          {selected
            ? t("settings.voice.recents.detailTitle")
            : t("settings.voice.recents.sectionTitle")}
        </h2>
        <p className="text-xs text-muted-foreground">
          {selected
            ? formatDictationDate(selected.at)
            : t("settings.voice.recents.pageDescription")}
        </p>
      </div>

      {selected ? (
        <div className="flex min-h-0 flex-col gap-4">
          <article className="rounded-lg border border-border/60 bg-muted/20 p-4">
            <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground select-text">
              {selected.text}
            </p>
          </article>
          <div className="flex justify-end gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={async () => {
                await handleCopy(selected.text);
              }}
            >
              <HugeiconsIcon icon={Copy01Icon} className="mr-1.5 size-3.5" />
              {t("settings.voice.recents.copy")}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                setPendingDelete({ kind: "one", dictation: selected })
              }
              className="text-destructive hover:border-destructive/60 hover:text-destructive"
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
        <div className="flex flex-col">
          <div className="flex items-center gap-4 border-b border-border/60 px-1 pb-2 text-xs font-semibold text-foreground">
            <span className="min-w-0 flex-1">
              {t("settings.voice.recents.dictationColumn")}
            </span>
            <span className="hidden w-40 shrink-0 sm:block">
              {t("settings.voice.recents.dateColumn")}
            </span>
            <span className="w-16 shrink-0" />
          </div>
          {recentDictations.map((dictation) => (
            <div
              key={dictation.id}
              className="group flex items-stretch gap-2 border-b border-border/40 last:border-0"
            >
              <button
                type="button"
                onClick={() => onSelect(dictation.id)}
                aria-label={t("settings.voice.recents.view")}
                className="flex min-w-0 flex-1 items-start gap-4 rounded-md px-1 py-3 text-left transition-colors hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <span className="min-w-0 flex-1">
                  <span className="line-clamp-3 whitespace-pre-wrap break-words text-sm text-foreground">
                    {dictation.text}
                  </span>
                  <span className="mt-1 block text-xs text-muted-foreground sm:hidden">
                    {formatDictationDate(dictation.at)}
                  </span>
                </span>
                <span className="hidden w-40 shrink-0 text-sm text-muted-foreground tabular-nums sm:block">
                  {formatDictationDate(dictation.at)}
                </span>
              </button>
              <span className="flex w-16 shrink-0 items-center justify-end gap-1">
                <button
                  type="button"
                  onClick={async () => {
                    await handleCopy(dictation.text);
                  }}
                  aria-label={t("settings.voice.recents.copy")}
                  title={t("settings.voice.recents.copy")}
                  className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <HugeiconsIcon
                    icon={Copy01Icon}
                    strokeWidth={1.75}
                    className="size-4"
                  />
                </button>
                <button
                  type="button"
                  onClick={() => setPendingDelete({ kind: "one", dictation })}
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
            </div>
          ))}
          <div className="flex justify-end border-t border-border/60 pt-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPendingDelete({ kind: "all" })}
              className="text-destructive hover:border-destructive/60 hover:text-destructive"
            >
              <HugeiconsIcon icon={Delete02Icon} className="mr-1.5 size-3.5" />
              {t("settings.voice.recents.clear")}
            </Button>
          </div>
        </div>
      )}

      <AlertDialog
        open={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open) {
            setPendingDelete(null);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {pendingDelete?.kind === "all"
                ? t("settings.voice.recents.clearTitle")
                : t("settings.voice.recents.deleteTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {pendingDelete?.kind === "all"
                ? t("settings.voice.recents.clearDescription")
                : t("settings.voice.recents.deleteDescription")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={confirmDelete}>
              {pendingDelete?.kind === "all"
                ? t("settings.voice.recents.clearConfirm")
                : t("common.delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
