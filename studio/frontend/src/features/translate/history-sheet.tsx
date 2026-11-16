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
import { Input } from "@/components/ui/input";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { useLocale, useT } from "@/i18n";
import { withTimeFormat } from "@/lib/time-format";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState } from "react";
import { findTranslateLanguage, translateLanguageLabel } from "./languages";
import { searchHistory } from "./translate-history";
import { useTranslateStore } from "./translate-store";

export function TranslateHistorySheet({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const t = useT();
  const locale = useLocale();
  const history = useTranslateStore((state) => state.history);
  const restore = useTranslateStore((state) => state.restore);
  const removeHistory = useTranslateStore((state) => state.removeHistory);
  const clearHistory = useTranslateStore((state) => state.clearHistory);
  const [query, setQuery] = useState("");
  const [confirmingClear, setConfirmingClear] = useState(false);
  const shown = useMemo(() => searchHistory(history, query), [history, query]);
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
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="flex w-full flex-col gap-0 font-heading sm:max-w-md"
      >
        <SheetHeader>
          <SheetTitle>{t("translate.history.title")}</SheetTitle>
        </SheetHeader>
        {history.length > 0 && (
          <div className="flex items-center gap-2 px-4 pb-3">
            <div className="relative min-w-0 flex-1">
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
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setConfirmingClear(true)}
            >
              {t("translate.history.clearAll")}
            </Button>
          </div>
        )}
        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-4">
          {shown.length === 0 ? (
            <p className="px-4 py-10 text-center text-sm text-muted-foreground">
              {history.length === 0
                ? t("translate.history.empty")
                : t("translate.history.noMatches")}
            </p>
          ) : (
            <ul className="flex flex-col gap-1">
              {shown.map((entry) => {
                const language = findTranslateLanguage(entry.target);
                return (
                  <li key={entry.id} className="group/entry relative">
                    <button
                      type="button"
                      onClick={() => {
                        restore(entry);
                        onOpenChange(false);
                      }}
                      className="flex w-full cursor-pointer flex-col gap-1 rounded-xl px-3 py-2.5 text-left outline-none hover:bg-muted focus-visible:bg-muted"
                    >
                      <span className="flex items-center gap-2 pr-8 text-xs text-muted-foreground">
                        <span className="truncate">
                          {language
                            ? translateLanguageLabel(language, locale)
                            : entry.target}
                          {entry.model ? ` · ${entry.model}` : ""}
                        </span>
                        <span className="ml-auto shrink-0">
                          {formatWhen(entry.at)}
                        </span>
                      </span>
                      <span
                        dir="auto"
                        className="line-clamp-2 text-sm text-muted-foreground"
                      >
                        {entry.source}
                      </span>
                      <span dir="auto" className="line-clamp-2 text-sm">
                        {entry.output}
                      </span>
                    </button>
                    <button
                      type="button"
                      title={t("translate.history.delete")}
                      aria-label={t("translate.history.delete")}
                      onClick={() => removeHistory(entry.id)}
                      className="absolute right-2 top-1.5 flex size-7 cursor-pointer items-center justify-center rounded-md text-muted-foreground opacity-0 outline-none hover:bg-background hover:text-foreground focus-visible:opacity-100 group-hover/entry:opacity-100"
                    >
                      <HugeiconsIcon
                        icon={TrashIcon}
                        strokeWidth={1.75}
                        className="size-4"
                      />
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
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
      </SheetContent>
    </Sheet>
  );
}
