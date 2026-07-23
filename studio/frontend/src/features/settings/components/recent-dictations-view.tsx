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
import {
  deleteChatItem,
  useChatRuntimeStore,
  type SidebarItem,
} from "@/features/chat";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import {
  ArrowLeft01Icon,
  Copy01Icon,
  Delete02Icon,
  Message01Icon,
  Search01Icon,
  ViewIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import {
  type RecentDictation,
  useVoiceSettingsStore,
} from "../stores/voice-settings-store";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

/** Dictations shown per page; "Show more" reveals the next page. */
const PAGE_SIZE = 20;

type PendingDelete =
  | { kind: "one"; dictation: RecentDictation }
  | { kind: "all" }
  | null;

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
  const [search, setSearch] = useState("");
  const [sortOrder, setSortOrder] = useState<SortOrder>("newest");
  // Pagination is keyed to the search/sort inputs so changing either restarts
  // from the first page without an effect.
  const pageKey = `${sortOrder}::${search}`;
  const [page, setPage] = useState({ key: pageKey, count: PAGE_SIZE });
  const visibleCount = page.key === pageKey ? page.count : PAGE_SIZE;
  const navigate = useNavigate();
  const closeSettings = useSettingsDialogStore((s) => s.closeDialog);
  const selected =
    recentDictations.find((dictation) => dictation.id === selectedId) ?? null;

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

  async function handleCopy(text: string) {
    if (await copyToClipboard(text)) {
      toast.success(t("settings.voice.recents.copied"));
    } else {
      toast.error(t("settings.voice.recents.copyFailed"));
    }
  }

  function confirmDelete() {
    try {
      if (pendingDelete?.kind === "one") {
        removeRecentDictation(pendingDelete.dictation.id);
        if (pendingDelete.dictation.id === selectedId) {
          onSelect(null);
        }
      } else if (pendingDelete?.kind === "all") {
        clearRecentDictations();
        onSelect(null);
      }
    } finally {
      setPendingDelete(null);
    }
  }

  // Delete a linked dictation together with the chat it was used in.
  async function confirmDeleteWithChat() {
    if (pendingDelete?.kind !== "one") {
      return;
    }
    const dictation = pendingDelete.dictation;
    setPendingDelete(null);
    if (dictation.chatId) {
      const item: SidebarItem = {
        type: "single",
        id: dictation.chatId,
        threadIds: [dictation.chatId],
        title: "",
        createdAt: 0,
        updatedAt: 0,
      };
      try {
        await deleteChatItem(
          item,
          useChatRuntimeStore.getState().activeThreadId ?? undefined,
          () => {
            // The deleted chat was open; leave the user on a fresh chat.
            void navigate({
              to: "/chat",
              search: { new: crypto.randomUUID() },
            });
          },
        );
      } catch (error) {
        toast.error(t("settings.voice.recents.deleteWithChatFailed"), {
          description: error instanceof Error ? error.message : undefined,
        });
        return;
      }
    }
    removeRecentDictation(dictation.id);
    if (dictation.id === selectedId) {
      onSelect(null);
    }
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
            {selected.chatId ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => openChat(selected.chatId as string)}
              >
                <HugeiconsIcon
                  icon={Message01Icon}
                  className="mr-1.5 size-3.5"
                />
                {t("settings.voice.recents.openChat")}
              </Button>
            ) : null}
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
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <InputGroup className="h-8 flex-1">
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
                size="sm"
                className="w-36 shrink-0"
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
              variant="outline"
              size="sm"
              onClick={() => setPendingDelete({ kind: "all" })}
              className="shrink-0 text-destructive hover:border-destructive/60 hover:text-destructive"
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
              <div className="flex items-center gap-4 border-b border-border/60 px-1 pb-2 text-xs font-semibold text-foreground">
                <span className="min-w-0 flex-1">
                  {t("settings.voice.recents.dictationColumn")}
                </span>
                <span className="hidden w-40 shrink-0 items-center gap-1.5 sm:flex">
                  {/* Spacer matching the row's chat-link icon slot so the
                      header starts exactly where the dates start. */}
                  <span className="size-3.5 shrink-0" />
                  {t("settings.voice.recents.dateColumn")}
                </span>
                <span className="w-24 shrink-0" />
              </div>
              {visible.slice(0, visibleCount).map((dictation) => (
                <div
                  key={dictation.id}
                  className="group flex items-stretch gap-2 border-b border-border/40 last:border-0"
                >
                  <button
                    type="button"
                    onClick={() =>
                      dictation.chatId
                        ? openChat(dictation.chatId)
                        : onSelect(dictation.id)
                    }
                    aria-label={
                      dictation.chatId
                        ? t("settings.voice.recents.openChat")
                        : t("settings.voice.recents.view")
                    }
                    className="flex min-w-0 flex-1 items-start gap-4 rounded-md px-1 py-3 text-left transition-colors hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <span className="min-w-0 flex-1">
                      <span className="line-clamp-3 whitespace-pre-wrap break-words text-sm text-foreground">
                        {dictation.text}
                      </span>
                      <span className="mt-1 flex items-center gap-1.5 text-xs text-muted-foreground sm:hidden">
                        {dictation.chatId ? (
                          <HugeiconsIcon
                            icon={Message01Icon}
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
                            icon={Message01Icon}
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
                  <span className="flex w-24 shrink-0 items-center justify-end gap-1">
                    {dictation.chatId ? (
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
                    ) : null}
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
                      onClick={() =>
                        setPendingDelete({ kind: "one", dictation })
                      }
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
                : pendingDelete?.kind === "one" &&
                    pendingDelete.dictation.chatId
                  ? t("settings.voice.recents.deleteLinkedDescription")
                  : t("settings.voice.recents.deleteDescription")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            {pendingDelete?.kind === "one" && pendingDelete.dictation.chatId ? (
              <AlertDialogAction
                variant="outline"
                className="text-destructive hover:border-destructive/60 hover:text-destructive"
                onClick={(event) => {
                  // Close only through state so the dismissal can't race a
                  // click on whatever ends up under the pointer.
                  event.preventDefault();
                  void confirmDeleteWithChat();
                }}
              >
                {t("settings.voice.recents.deleteWithChat")}
              </AlertDialogAction>
            ) : null}
            <AlertDialogAction
              variant="destructive"
              onClick={(event) => {
                event.preventDefault();
                confirmDelete();
              }}
            >
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
