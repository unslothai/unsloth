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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  type DateBucket,
  dateBucket,
  dateBucketKey,
} from "@/features/library/chats/model";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { useTimeFormat, withTimeFormat } from "@/lib/time-format";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Download01Icon,
  Message01Icon,
  MoreHorizontalIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { ArrowDownIcon } from "lucide-react";
import { memo, useMemo, useState } from "react";
import { chatAboutTranslations } from "./chat-handoff";
import { IconAction } from "./icon-action";
import { findTranslateLanguage, translateLanguageLabel } from "./languages";
import {
  TRANSLATE_HISTORY_FILE_NAME,
  type TranslateHistoryEntry,
  exportHistory,
  recentForChat,
  searchHistory,
} from "./translate-history";
import { useTranslateStore } from "./translate-store";

const ROW =
  "group/row relative flex cursor-pointer items-center gap-4 rounded-md pl-4 pr-6 transition-colors hover:bg-muted dark:hover:bg-muted/60";
const CELL = "truncate text-ui-12p5 text-muted-foreground";
const ROW_MENU_TRIGGER =
  "flex size-8 shrink-0 items-center justify-center rounded-full text-muted-foreground opacity-0 outline-none transition-opacity hover:bg-accent hover:text-foreground focus-visible:opacity-100 group-hover/row:opacity-100 data-[state=open]:opacity-100 pointer-coarse:opacity-100";

const BUCKET_LABELS: Record<
  Exclude<DateBucket["kind"], "older">,
  TranslationKey
> = {
  today: "library.chats.groups.today",
  yesterday: "library.chats.groups.yesterday",
  week: "library.chats.groups.week",
  month: "library.chats.groups.month",
};

// No props, so memo keeps the page's per-keystroke and per-streamed-frame renders off these rows.
export const TranslateHistoryList = memo(function TranslateHistoryList() {
  const t = useT();
  const locale = useLocale();
  // formatWhen reads the 12h/24h setting; subscribe so a Settings change still reaches the rows.
  useTimeFormat();
  const history = useTranslateStore((state) => state.history);
  const restore = useTranslateStore((state) => state.restore);
  const removeHistory = useTranslateStore((state) => state.removeHistory);
  const clearHistory = useTranslateStore((state) => state.clearHistory);
  const [query, setQuery] = useState("");
  const [newestFirst, setNewestFirst] = useState(true);
  const [confirmingClear, setConfirmingClear] = useState(false);
  // History is stored newest first.
  const shown = useMemo(() => {
    const found = searchHistory(history, query);
    return newestFirst ? found : [...found].reverse();
  }, [history, query, newestFirst]);
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
  // Ordered by time either way, so entries of one date bucket are already adjacent.
  const groups = useMemo(() => {
    const now = Date.now();
    const out: {
      key: string;
      bucket: DateBucket;
      items: TranslateHistoryEntry[];
    }[] = [];
    for (const entry of shown) {
      const bucket = dateBucket(entry.at, now);
      const key = dateBucketKey(bucket);
      const last = out[out.length - 1];
      if (last?.key === key) last.items.push(entry);
      else out.push({ key, bucket, items: [entry] });
    }
    return out;
  }, [shown]);
  const groupLabel = (bucket: DateBucket) =>
    bucket.kind === "older"
      ? new Date(bucket.year, bucket.month).toLocaleDateString(locale, {
          month: "long",
          year: "numeric",
        })
      : t(BUCKET_LABELS[bucket.kind]);
  const formatWhen = (at: number, bucket: DateBucket["kind"]) =>
    new Date(at).toLocaleString(
      locale,
      bucket === "today" || bucket === "yesterday"
        ? withTimeFormat({ hour: "numeric", minute: "2-digit" }, locale)
        : bucket === "week"
          ? { weekday: "long" }
          : { month: "short", day: "numeric" },
    );

  return (
    <section
      aria-label={t("translate.history.title")}
      className="@container font-sans"
    >
      <div className="flex flex-wrap items-center gap-2 font-heading">
        <h2 className="me-auto text-ui-13 font-medium text-foreground">
          {t("translate.history.title")}
        </h2>
        {history.length > 0 && (
          <>
            <div className="relative w-full min-w-0 sm:w-96">
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
                onClick={() => chat(recentForChat(history))}
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
        <p className="mt-3 text-ui-13 leading-normal text-muted-foreground">
          {history.length === 0
            ? t("translate.history.empty")
            : t("translate.history.noMatches")}
        </p>
      ) : (
        groups.map((group, index) => (
          <section
            key={group.key}
            aria-label={groupLabel(group.bucket)}
            className="not-first:mt-6"
          >
            <h3 className="mb-2 mt-3 flex items-baseline gap-4 border-b border-border pb-2 pl-4 pr-6 text-ui-12">
              <span className="flex flex-1 items-baseline gap-2">
                <span className="font-medium text-muted-foreground">
                  {groupLabel(group.bucket)}
                </span>
                <span className="text-ui-11 text-muted-foreground/70">
                  {group.items.length.toLocaleString(locale)}
                </span>
              </span>
              {index === 0 && (
                <>
                  <button
                    type="button"
                    onClick={() => setNewestFirst((value) => !value)}
                    title={t(
                      newestFirst
                        ? "projectsPage.newestFirst"
                        : "projectsPage.oldestFirst",
                    )}
                    className="flex w-24 shrink-0 cursor-pointer items-center gap-1 text-left text-ui-12p5 font-medium text-muted-foreground transition-colors hover:text-foreground"
                  >
                    {t("projectsPage.updated")}
                    <ArrowDownIcon
                      strokeWidth={1.75}
                      className={cn(
                        "size-3.5 transition-transform",
                        !newestFirst && "rotate-180",
                      )}
                    />
                  </button>
                  <span className="w-8 shrink-0" />
                </>
              )}
            </h3>
            <div className="flex flex-col gap-1">
              {group.items.map((entry) => {
                const language = findTranslateLanguage(entry.target);
                return (
                  <div key={entry.id} className={ROW}>
                    <button
                      type="button"
                      className="flex min-w-0 flex-1 cursor-pointer items-center gap-4 rounded-md py-2 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      onClick={() => {
                        restore(entry);
                        window.scrollTo({ top: 0 });
                      }}
                    >
                      <span className="flex min-w-0 flex-1 flex-col">
                        <span
                          dir="auto"
                          className="truncate text-ui-13 text-foreground"
                        >
                          {entry.output}
                        </span>
                        <span
                          dir="auto"
                          className="truncate text-ui-11p5 text-muted-foreground"
                        >
                          {entry.source}
                        </span>
                      </span>
                      <span
                        className={cn(CELL, "hidden w-52 shrink-0 @3xl:block")}
                      >
                        {language
                          ? translateLanguageLabel(language, locale)
                          : entry.target}
                        {entry.model ? ` · ${entry.model}` : ""}
                      </span>
                      <span
                        className={cn(CELL, "hidden w-24 shrink-0 @xl:block")}
                      >
                        {formatWhen(entry.at, group.bucket.kind)}
                      </span>
                    </button>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild={true}>
                        <button
                          type="button"
                          aria-label={t("shell.navigation.more")}
                          className={ROW_MENU_TRIGGER}
                        >
                          <HugeiconsIcon
                            icon={MoreHorizontalIcon}
                            strokeWidth={1.75}
                            className="size-5"
                          />
                        </button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" className="w-64">
                        <DropdownMenuItem onSelect={() => chat([entry])}>
                          <HugeiconsIcon icon={Message01Icon} />
                          {t("translate.chat")}
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          variant="destructive"
                          onSelect={() => removeHistory(entry.id)}
                        >
                          <HugeiconsIcon icon={TrashIcon} />
                          {t("common.delete")}
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                );
              })}
            </div>
          </section>
        ))
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
});
