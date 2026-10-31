// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Custom spellings: a small, focused editor for the words and spoken shortcuts
// dictation should respect. One list picker, one search field, one add field and
// a clean row list; editing is inline and destructive actions ask first.

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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useUserProfileStore } from "@/features/profile/stores/user-profile-store";
import { useT } from "@/i18n";
import { ChevronLeftStandardIcon } from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import {
  ArrowRight01Icon,
  Cancel01Icon,
  Delete02Icon,
  MoreVerticalIcon,
  PencilEdit02Icon,
  PlusSignIcon,
  Search01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useState } from "react";
import {
  extractUserNameWords,
  useVoiceSettingsStore,
} from "../stores/voice-settings-store";

const SNIPPET_PATTERN = /^(.+?)\s*(?:->|=>)\s*(.+)$/;

export function DictationDictionaryView({ onBack }: { onBack: () => void }) {
  const t = useT();
  const collections = useVoiceSettingsStore((s) => s.collections);
  const activeCollectionId = useVoiceSettingsStore((s) => s.activeCollectionId);
  const setActiveCollectionId = useVoiceSettingsStore(
    (s) => s.setActiveCollectionId,
  );
  const addCollection = useVoiceSettingsStore((s) => s.addCollection);
  const removeCollection = useVoiceSettingsStore((s) => s.removeCollection);
  const renameCollection = useVoiceSettingsStore((s) => s.renameCollection);
  const addWordToCollection = useVoiceSettingsStore(
    (s) => s.addWordToCollection,
  );
  const updateWordInCollection = useVoiceSettingsStore(
    (s) => s.updateWordInCollection,
  );
  const commitWordInCollection = useVoiceSettingsStore(
    (s) => s.commitWordInCollection,
  );
  const removeWordFromCollection = useVoiceSettingsStore(
    (s) => s.removeWordFromCollection,
  );
  const syncUserNameToDictionary = useVoiceSettingsStore(
    (s) => s.syncUserNameToDictionary,
  );
  const resetDictionaryToDefaults = useVoiceSettingsStore(
    (s) => s.resetDictionaryToDefaults,
  );

  const displayName = useUserProfileStore((s) => s.displayName);
  const nickname = useUserProfileStore((s) => s.nickname);

  const [newEntry, setNewEntry] = useState("");
  const [search, setSearch] = useState("");
  const [creatingList, setCreatingList] = useState(false);
  const [newListName, setNewListName] = useState("");
  const [renamingList, setRenamingList] = useState(false);
  const [listNameDraft, setListNameDraft] = useState("");
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const [pending, setPending] = useState<"delete-list" | "reset" | null>(null);
  const [visibleLimit, setVisibleLimit] = useState(50);

  useEffect(() => {
    syncUserNameToDictionary(displayName, nickname);
  }, [displayName, nickname, syncUserNameToDictionary]);

  const activeCollection = useMemo(() => {
    return (
      collections.find((collection) => collection.id === activeCollectionId) ??
      collections[0] ?? {
        id: "default",
        name: "Default vocabulary",
        isDefault: true,
        words: [],
      }
    );
  }, [collections, activeCollectionId]);

  const userWords = extractUserNameWords(displayName, nickname);
  const isDefault = Boolean(activeCollection.isDefault);
  const needle = search.trim().toLowerCase();
  const visibleWords = useMemo(() => {
    return activeCollection.words
      .map((word, index) => ({ word, index }))
      .filter(({ word }) => !needle || word.toLowerCase().includes(needle));
  }, [activeCollection.words, needle]);

  useEffect(() => {
    setVisibleLimit(50);
  }, [activeCollection.id, needle]);

  const resetTransientState = () => {
    setSearch("");
    setNewEntry("");
    setEditingIndex(null);
    setRenamingList(false);
    setCreatingList(false);
  };

  const handleSelectList = (id: string) => {
    setActiveCollectionId(id);
    resetTransientState();
  };

  const handleAddEntry = () => {
    const trimmed = newEntry.trim();
    if (!trimmed) return;
    if (
      activeCollection.words.some(
        (word) => word.toLowerCase() === trimmed.toLowerCase(),
      )
    ) {
      toast.info(t("settings.voice.dictionary.duplicateEntry"));
      setNewEntry("");
      return;
    }
    addWordToCollection(activeCollection.id, trimmed);
    setNewEntry("");
  };

  const handleCreateList = () => {
    const trimmed = newListName.trim();
    if (!trimmed) return;
    addCollection(trimmed);
    setNewListName("");
    setCreatingList(false);
    setSearch("");
  };

  const handleRenameList = () => {
    const trimmed = listNameDraft.trim();
    if (!trimmed) {
      setRenamingList(false);
      return;
    }
    renameCollection(activeCollection.id, trimmed);
    setRenamingList(false);
  };

  const startEditing = (index: number, word: string) => {
    setEditingIndex(index);
    setEditDraft(word);
  };

  const commitEdit = () => {
    if (editingIndex === null) return;
    updateWordInCollection(activeCollection.id, editingIndex, editDraft);
    commitWordInCollection(activeCollection.id, editingIndex);
    setEditingIndex(null);
  };

  const confirmPending = () => {
    if (pending === "delete-list") {
      removeCollection(activeCollection.id);
      resetTransientState();
    } else if (pending === "reset") {
      resetDictionaryToDefaults();
      resetTransientState();
    }
    setPending(null);
  };

  return (
    <div className="settings-page">
      <header className="flex items-center gap-2">
        <button
          type="button"
          onClick={onBack}
          aria-label={t("settings.voice.dictionary.backToVoice")}
          className="settings-back-button inline-flex size-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <HugeiconsIcon
            icon={ChevronLeftStandardIcon}
            className="size-4 rtl:rotate-180"
          />
        </button>
        <h1 className="font-heading text-xl font-semibold">
          {t("settings.voice.title")}
        </h1>
      </header>

      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="settings-heading text-sm font-semibold">
            {t("settings.voice.dictionary.title")}
          </h2>
          <p className="mt-0.5 max-w-lg text-xs text-muted-foreground">
            {t("settings.voice.dictionary.subtitle")}
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          {!isDefault ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild={true}>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="size-8 shrink-0"
                  aria-label={t("settings.voice.dictionary.listLabel")}
                >
                  <HugeiconsIcon
                    icon={MoreVerticalIcon}
                    className="size-3.5"
                  />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem
                  onClick={() => {
                    setListNameDraft(activeCollection.name);
                    setRenamingList(true);
                    setCreatingList(false);
                  }}
                >
                  {t("settings.voice.dictionary.renameList")}
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="text-destructive"
                  onClick={() => setPending("delete-list")}
                >
                  {t("settings.voice.dictionary.deleteList")}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null}
          <Button
            variant="outline"
            size="sm"
            className="shrink-0"
            onClick={() => {
              setCreatingList(true);
              setRenamingList(false);
            }}
          >
            <HugeiconsIcon icon={PlusSignIcon} className="mr-1.5 size-3.5" />
            {t("settings.voice.dictionary.newList")}
          </Button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Select value={activeCollection.id} onValueChange={handleSelectList}>
          <SelectTrigger
            size="sm"
            className="w-56"
            aria-label={t("settings.voice.dictionary.listLabel")}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {collections.map((collection) => (
              <SelectItem key={collection.id} value={collection.id}>
                {collection.name} ({collection.words.length})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="text-xs text-muted-foreground">
          {t("settings.voice.dictionary.wordCount", {
            count: activeCollection.words.length,
          })}
        </span>
        {isDefault && userWords.length > 0 ? (
          <span className="text-xs text-muted-foreground">
            {t("settings.voice.dictionary.fromProfile", {
              count: userWords.length,
            })}
          </span>
        ) : null}
        {isDefault ? (
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto h-8 text-xs"
            data-testid="dictionary-reset-defaults"
            onClick={() => setPending("reset")}
          >
            {t("settings.voice.dictionary.resetDefaults")}
          </Button>
        ) : null}
      </div>

      {creatingList ? (
        <div className="flex items-center gap-2 rounded-xl border border-border/60 bg-muted/20 p-2">
          <Input
            value={newListName}
            onChange={(event) => setNewListName(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                handleCreateList();
              } else if (event.key === "Escape") {
                setCreatingList(false);
                setNewListName("");
              }
            }}
            autoFocus={true}
            placeholder={t("settings.voice.dictionary.newListPlaceholder")}
            aria-label={t("settings.voice.dictionary.newListPlaceholder")}
            className="h-8 flex-1 text-sm"
          />
          <Button
            size="sm"
            onClick={handleCreateList}
            disabled={!newListName.trim()}
          >
            {t("settings.voice.dictionary.create")}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setCreatingList(false);
              setNewListName("");
            }}
          >
            {t("settings.voice.dictionary.cancel")}
          </Button>
        </div>
      ) : null}

      {renamingList ? (
        <div className="flex items-center gap-2 rounded-xl border border-border/60 bg-muted/20 p-2">
          <Input
            value={listNameDraft}
            onChange={(event) => setListNameDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                handleRenameList();
              } else if (event.key === "Escape") {
                setRenamingList(false);
              }
            }}
            autoFocus={true}
            aria-label={t("settings.voice.dictionary.renameList")}
            className="h-8 flex-1 text-sm"
          />
          <Button size="sm" onClick={handleRenameList}>
            {t("settings.voice.dictionary.save")}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setRenamingList(false)}
          >
            {t("settings.voice.dictionary.cancel")}
          </Button>
        </div>
      ) : null}

      {!isDefault ? (
        <div className="flex items-center gap-2">
          <Input
            value={newEntry}
            onChange={(event) => setNewEntry(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                handleAddEntry();
              }
            }}
            placeholder={t("settings.voice.dictionary.newEntryPlaceholder")}
            className="h-9 flex-1 text-sm"
            aria-label={t("settings.voice.dictionary.newEntryAria")}
          />
          <Button
            size="sm"
            className="h-9 shrink-0"
            onClick={handleAddEntry}
            disabled={!newEntry.trim()}
          >
            <HugeiconsIcon icon={PlusSignIcon} className="mr-1.5 size-3.5" />
            {t("settings.voice.dictionary.addEntry")}
          </Button>
        </div>
      ) : null}

      <div className="relative">
        <HugeiconsIcon
          icon={Search01Icon}
          strokeWidth={2}
          className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-3.5 -translate-y-1/2"
        />
        <Input
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder={t("settings.voice.dictionary.searchPlaceholder")}
          aria-label={t("settings.voice.dictionary.searchPlaceholder")}
          className="h-8 pl-8 text-sm"
          disabled={activeCollection.words.length === 0}
        />
      </div>

      <div className="overflow-hidden rounded-xl border border-border/60">
        {visibleWords.length === 0 ? (
          <p className="px-4 py-8 text-center text-xs text-muted-foreground">
            {activeCollection.words.length === 0
              ? t("settings.voice.dictionary.empty")
              : t("settings.voice.dictionary.noMatches")}
          </p>
        ) : (
          <div className="divide-y divide-border/40">
            {visibleWords.slice(0, visibleLimit).map(({ word, index }) => {
              const snippet = SNIPPET_PATTERN.exec(word);
              const editing = editingIndex === index;
              return (
                <div
                  key={`${activeCollection.id}-${index}`}
                  className="group flex items-center gap-2 px-3 py-1.5 transition-colors hover:bg-muted/40"
                >
                  {editing ? (
                    <>
                      <Input
                        value={editDraft}
                        onChange={(event) => setEditDraft(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") {
                            event.preventDefault();
                            commitEdit();
                          } else if (event.key === "Escape") {
                            setEditingIndex(null);
                          }
                        }}
                        autoFocus={true}
                        aria-label={t("settings.voice.dictionary.entryAria", {
                          index: index + 1,
                        })}
                        className="h-8 flex-1 text-sm"
                      />
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        className="size-7 shrink-0 text-primary"
                        onClick={commitEdit}
                        aria-label={t("settings.voice.dictionary.save")}
                      >
                        <HugeiconsIcon icon={Tick02Icon} className="size-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        className="size-7 shrink-0 text-muted-foreground"
                        onClick={() => setEditingIndex(null)}
                        aria-label={t("settings.voice.dictionary.cancel")}
                      >
                        <HugeiconsIcon
                          icon={Cancel01Icon}
                          className="size-3.5"
                        />
                      </Button>
                    </>
                  ) : (
                    <>
                      <div className="min-w-0 flex-1">
                        {snippet ? (
                          <span className="flex items-center gap-1.5 text-sm">
                            <span
                              className="truncate font-mono text-xs text-muted-foreground"
                              title={snippet[1]}
                            >
                              {snippet[1]}
                            </span>
                            <HugeiconsIcon
                              icon={ArrowRight01Icon}
                              className="size-3 shrink-0 text-muted-foreground"
                            />
                            <span className="truncate" title={snippet[2]}>
                              {snippet[2]}
                            </span>
                          </span>
                        ) : (
                          <span className="block truncate text-sm" title={word}>
                            {word}
                          </span>
                        )}
                      </div>
                      <div className="flex shrink-0 items-center gap-0.5 opacity-100 transition-opacity sm:opacity-0 sm:group-focus-within:opacity-100 sm:group-hover:opacity-100">
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          className="size-7 text-muted-foreground"
                          onClick={() => startEditing(index, word)}
                          aria-label={t("settings.voice.dictionary.editEntry")}
                          title={t("settings.voice.dictionary.editEntry")}
                        >
                          <HugeiconsIcon
                            icon={PencilEdit02Icon}
                            className="size-3.5"
                          />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          className="size-7 text-muted-foreground hover:text-destructive"
                          onClick={() => {
                            setEditingIndex(null);
                            removeWordFromCollection(
                              activeCollection.id,
                              index,
                            );
                          }}
                          aria-label={t(
                            "settings.voice.dictionary.removeEntryAria",
                            { index: index + 1 },
                          )}
                          title={t("settings.voice.dictionary.removeEntryAria", {
                            index: index + 1,
                          })}
                        >
                          <HugeiconsIcon
                            icon={Delete02Icon}
                            className="size-3.5"
                          />
                        </Button>
                      </div>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {visibleWords.length > visibleLimit ? (
        <div className="flex justify-center">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setVisibleLimit((limit) => limit + 100)}
          >
            {t("settings.voice.dictionary.showMore", {
              count: visibleWords.length - visibleLimit,
            })}
          </Button>
        </div>
      ) : null}

      <AlertDialog
        open={pending !== null}
        onOpenChange={(open) => {
          if (!open) setPending(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {pending === "reset"
                ? t("settings.voice.dictionary.resetTitle")
                : t("settings.voice.dictionary.deleteListTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {pending === "reset"
                ? t("settings.voice.dictionary.resetDescription")
                : t("settings.voice.dictionary.deleteListDescription", {
                    name: activeCollection.name,
                    count: activeCollection.words.length,
                  })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>
              {t("settings.voice.dictionary.cancel")}
            </AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              className="text-destructive hover:text-destructive"
              onClick={(event) => {
                event.preventDefault();
                confirmPending();
              }}
            >
              {pending === "reset"
                ? t("settings.voice.dictionary.resetDefaults")
                : t("settings.voice.dictionary.deleteList")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
