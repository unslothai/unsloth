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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { Copy01Icon, Delete02Icon } from "@hugeicons/core-free-icons";
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

export function RecentDictationsDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
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

  async function handleCopy(text: string) {
    // The helper also supports browsers and local contexts where the modern
    // clipboard API is unavailable.
    if (await copyToClipboard(text)) {
      toast.success(t("settings.voice.recents.copied"));
    } else {
      toast.error(t("settings.voice.recents.copyFailed"));
    }
  }

  function confirmDelete() {
    if (pendingDelete?.kind === "one") {
      removeRecentDictation(pendingDelete.dictation.id);
    } else if (pendingDelete?.kind === "all") {
      clearRecentDictations();
    }
    setPendingDelete(null);
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t("settings.voice.recents.sectionTitle")}</DialogTitle>
          <DialogDescription>
            {t("settings.voice.recents.dialogDescription")}
          </DialogDescription>
        </DialogHeader>

        {recentDictations.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            {t("settings.voice.recents.empty")}
          </p>
        ) : (
          <>
            <div className="max-h-[60vh] overflow-y-auto">
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
                  className="group flex items-start gap-4 border-b border-border/40 px-1 py-2.5 text-sm last:border-0"
                >
                  <div className="min-w-0 flex-1">
                    <p
                      className="line-clamp-3 whitespace-pre-wrap break-words text-foreground"
                      title={dictation.text}
                    >
                      {dictation.text}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground sm:hidden">
                      {formatDictationDate(dictation.at)}
                    </p>
                  </div>
                  <span className="hidden w-40 shrink-0 text-muted-foreground tabular-nums sm:block">
                    {formatDictationDate(dictation.at)}
                  </span>
                  <span className="flex w-16 shrink-0 items-center justify-end gap-1">
                    <button
                      type="button"
                      onClick={async () => {
                        await handleCopy(dictation.text);
                      }}
                      aria-label={t("settings.voice.recents.copy")}
                      title={t("settings.voice.recents.copy")}
                      className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                    >
                      <HugeiconsIcon
                        icon={Copy01Icon}
                        strokeWidth={1.75}
                        className="size-4"
                      />
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        setPendingDelete({ kind: "one", dictation })
                      }
                      aria-label={t("settings.voice.recents.delete")}
                      title={t("settings.voice.recents.delete")}
                      className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
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
            </div>
            <div className="flex justify-end border-t border-border/60 pt-3">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPendingDelete({ kind: "all" })}
                className="text-destructive hover:border-destructive/60 hover:text-destructive"
              >
                <HugeiconsIcon
                  icon={Delete02Icon}
                  className="mr-1.5 size-3.5"
                />
                {t("settings.voice.recents.clear")}
              </Button>
            </div>
          </>
        )}
      </DialogContent>

      <AlertDialog
        open={pendingDelete !== null}
        onOpenChange={(nextOpen) => {
          if (!nextOpen) {
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
    </Dialog>
  );
}
