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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { useLocale, useT } from "@/i18n";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { withTimeFormat } from "@/lib/time-format";
import { toast } from "@/lib/toast";
import {
  Download01Icon,
  Message01Icon,
  MoreHorizontalIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { CHAT_HANDOFF_ENTRIES, chatAboutTranslations } from "./chat-handoff";
import { IconAction } from "./icon-action";
import { findTranslateLanguage, translateLanguageLabel } from "./languages";
import {
  TRANSLATE_HISTORY_FILE_NAME,
  type TranslateHistoryEntry,
  exportHistory,
  searchHistory,
} from "./translate-history";
import { useTranslateStore } from "./translate-store";

// As the Audio galleries: the row menu shows on hover, focus, while open, and always on touch.
const ROW_MENU_TRIGGER =
  "size-5 text-muted-foreground/60 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100 any-pointer-coarse:opacity-100";

/** The saved translations, listed under the panes like an Audio gallery. */
export function TranslateHistoryList() {
  const t = useT();
  const locale = useLocale();
  const history = useTranslateStore((state) => state.history);
  const restore = useTranslateStore((state) => state.restore);
  const removeHistory = useTranslateStore((state) => state.removeHistory);
  const clearHistory = useTranslateStore((state) => state.clearHistory);
  const [query, setQuery] = useState("");
  const [confirmingClear, setConfirmingClear] = useState(false);
  const shown = useMemo(() => searchHistory(history, query), [history, query]);
  const navigate = useNavigate();
  const chat = (entries: readonly TranslateHistoryEntry[]) =>
    chatAboutTranslations(entries, navigate);
  const exportAll = async () => {
    try {
      await downloadFile(
        exportHistory(history),
        TRANSLATE_HISTORY_FILE_NAME,
        "text/plain",
      );
    } catch (error) {
      if (isDownloadCancelled(error)) return;
      toast.error(t("translate.errors.exportFailed"), {
        description: error instanceof Error ? error.message : String(error),
      });
    }
  };
  const formatWhen = useMemo(() => {
    const format = new Intl.DateTimeFormat(
      locale,
      withTimeFormat(
        { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" },
        locale,
      ),
    );
    return (at: number) => format.format(at);
  }, [locale]);

  return (
    <section aria-label={t("translate.history.title")} className="grid gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="me-auto text-ui-13 font-medium text-foreground">
          {t("translate.history.title")}
        </h2>
        {history.length > 0 && (
          <>
            <div className="relative w-full min-w-0 sm:w-64">
              <span className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-muted-foreground">
                <HugeiconsIcon
                  icon={Search01Icon}
                  strokeWidth={1.75}
                  className="size-4"
                />
              </span>
              <Input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={t("translate.history.search")}
                aria-label={t("translate.history.search")}
                className="h-control w-full rounded-full border-none bg-muted pl-10 pr-4 shadow-none dark:bg-card"
              />
            </div>
            <div className="flex items-center">
              <IconAction
                icon={Message01Icon}
                label={t("translate.history.chatRecent")}
                onClick={() => chat(history.slice(0, CHAT_HANDOFF_ENTRIES))}
              />
              <IconAction
                icon={Download01Icon}
                label={t("translate.history.export")}
                onClick={() => void exportAll()}
              />
              <IconAction
                icon={TrashIcon}
                label={t("translate.history.clearAll")}
                onClick={() => setConfirmingClear(true)}
              />
            </div>
          </>
        )}
      </div>
      {shown.length === 0 ? (
        <p className="text-ui-13 leading-normal text-muted-foreground">
          {history.length === 0
            ? t("translate.history.empty")
            : t("translate.history.noMatches")}
        </p>
      ) : (
        <div className="flex flex-col gap-0.5 border-t border-wash-6 pt-2">
          {shown.map((entry) => {
            const language = findTranslateLanguage(entry.target);
            return (
              <div
                key={entry.id}
                className="group flex items-center gap-1 rounded-md ps-3 pe-1 transition-colors hover:bg-wash-3"
              >
                <button
                  type="button"
                  className="flex min-w-0 flex-1 cursor-pointer items-center gap-3 py-2.5 pr-3 text-left outline-none"
                  onClick={() => {
                    restore(entry);
                    window.scrollTo({ top: 0 });
                  }}
                >
                  <span className="grid min-w-0 flex-1 gap-0.5">
                    <span dir="auto" className="truncate text-ui-13">
                      {entry.output}
                    </span>
                    <span
                      dir="auto"
                      className="truncate text-ui-11p5 text-muted-foreground"
                    >
                      {entry.source}
                    </span>
                  </span>
                  <span className="shrink-0 text-ui-11p5 text-muted-foreground max-sm:hidden">
                    {language
                      ? translateLanguageLabel(language, locale)
                      : entry.target}
                    {entry.model ? ` · ${entry.model}` : ""}
                  </span>
                  <span className="shrink-0 font-mono text-ui-11p5 tabular-nums text-muted-foreground">
                    {formatWhen(entry.at)}
                  </span>
                </button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild={true}>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      aria-label={t("shell.navigation.more")}
                      className={ROW_MENU_TRIGGER}
                    >
                      <HugeiconsIcon
                        icon={MoreHorizontalIcon}
                        className="size-4"
                      />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onSelect={() => chat([entry])}>
                      <HugeiconsIcon icon={Message01Icon} />
                      {t("translate.chat")}
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      variant="destructive"
                      onSelect={() => removeHistory(entry.id)}
                    >
                      <HugeiconsIcon icon={TrashIcon} />
                      {t("translate.history.delete")}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            );
          })}
        </div>
      )}
      <AlertDialog open={confirmingClear} onOpenChange={setConfirmingClear}>
        <AlertDialogContent onOverlayClick={() => setConfirmingClear(false)}>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("translate.history.clearTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t("translate.history.clearDescription")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={clearHistory}>
              {t("translate.history.clearAll")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  );
}
