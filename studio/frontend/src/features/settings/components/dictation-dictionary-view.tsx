// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronLeftStandardIcon } from "@/lib/chevron-icons";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { useUserProfileStore } from "@/features/profile/stores/user-profile-store";
import { useT } from "@/i18n";
import {
  Delete02Icon,
  Folder01Icon,
  MoreVerticalIcon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useState } from "react";
import {
  extractUserNameWords,
  useVoiceSettingsStore,
} from "../stores/voice-settings-store";

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
  const addWordToCollection = useVoiceSettingsStore((s) => s.addWordToCollection);
  const updateWordInCollection = useVoiceSettingsStore(
    (s) => s.updateWordInCollection,
  );
  const commitWordInCollection = useVoiceSettingsStore(
    (s) => s.commitWordInCollection,
  );
  const removeWordFromCollection = useVoiceSettingsStore(
    (s) => s.removeWordFromCollection,
  );
  const batchRemoveWordsFromCollection = useVoiceSettingsStore(
    (s) => s.batchRemoveWordsFromCollection,
  );
  const batchMoveWordsToCollection = useVoiceSettingsStore(
    (s) => s.batchMoveWordsToCollection,
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
  const [newCollectionName, setNewCollectionName] = useState("");
  const [isAddingCollection, setIsAddingCollection] = useState(false);
  const [editingCollectionName, setEditingCollectionName] = useState<string | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);

  useEffect(() => {
    syncUserNameToDictionary(displayName, nickname);
  }, [displayName, nickname, syncUserNameToDictionary]);

  const activeCollection = useMemo(() => {
    return (
      collections.find((c) => c.id === activeCollectionId) ??
      collections[0] ?? {
        id: "default",
        name: "Default Vocabulary",
        isDefault: true,
        words: [],
      }
    );
  }, [collections, activeCollectionId]);

  const otherCollections = useMemo(() => {
    return collections.filter((c) => c.id !== activeCollection.id);
  }, [collections, activeCollection.id]);

  const userWords = extractUserNameWords(displayName, nickname);

  // Clear selections when switching collection
  const handleSelectCollection = (id: string) => {
    setActiveCollectionId(id);
    setSelectedIndices([]);
  };

  const handleAddEntry = () => {
    const trimmed = newEntry.trim();
    if (!trimmed) return;
    addWordToCollection(activeCollection.id, trimmed);
    setNewEntry("");
  };

  const handleCreateCollection = () => {
    const trimmed = newCollectionName.trim();
    if (!trimmed) return;
    addCollection(trimmed);
    setNewCollectionName("");
    setIsAddingCollection(false);
  };

  const handleRenameCollection = () => {
    if (!editingCollectionName || !editingCollectionName.trim()) {
      setEditingCollectionName(null);
      return;
    }
    renameCollection(activeCollection.id, editingCollectionName.trim());
    setEditingCollectionName(null);
  };

  const toggleSelectIndex = (index: number) => {
    setSelectedIndices((prev) =>
      prev.includes(index) ? prev.filter((i) => i !== index) : [...prev, index],
    );
  };

  const toggleSelectAll = () => {
    if (selectedIndices.length === activeCollection.words.length) {
      setSelectedIndices([]);
    } else {
      setSelectedIndices(activeCollection.words.map((_, i) => i));
    }
  };

  const handleBatchDelete = () => {
    if (selectedIndices.length === 0) return;
    batchRemoveWordsFromCollection(activeCollection.id, selectedIndices);
    setSelectedIndices([]);
  };

  const handleBatchMove = (targetId: string) => {
    if (selectedIndices.length === 0) return;
    batchMoveWordsToCollection(activeCollection.id, targetId, selectedIndices);
    setSelectedIndices([]);
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
          <HugeiconsIcon icon={ChevronLeftStandardIcon} className="size-4 rtl:rotate-180" />
        </button>
        <h1 className="font-heading text-xl font-semibold">
          {t("settings.voice.title")}
        </h1>
      </header>

      {/* Collection tabs & management */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-1.5">
            {collections.map((col) => {
              const isActive = col.id === activeCollection.id;
              return (
                <button
                  key={col.id}
                  type="button"
                  onClick={() => handleSelectCollection(col.id)}
                  className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                    isActive
                      ? "bg-primary text-primary-foreground shadow-xs"
                      : "bg-muted text-muted-foreground hover:bg-muted/80 hover:text-foreground"
                  }`}
                >
                  <HugeiconsIcon icon={Folder01Icon} className="size-3" />
                  <span>{col.name}</span>
                  <span className="rounded-full bg-background/20 px-1.5 py-0.2 text-[10px] tabular-nums">
                    {col.words.length}
                  </span>
                </button>
              );
            })}

            {isAddingCollection ? (
              <div className="flex items-center gap-1">
                <Input
                  value={newCollectionName}
                  onChange={(e) => setNewCollectionName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleCreateCollection();
                    } else if (e.key === "Escape") {
                      setIsAddingCollection(false);
                      setNewCollectionName("");
                    }
                  }}
                  autoFocus
                  placeholder="Collection name..."
                  className="h-7 w-36 text-xs"
                />
                <Button
                  size="sm"
                  className="h-7 px-2 text-xs"
                  onClick={handleCreateCollection}
                  disabled={!newCollectionName.trim()}
                >
                  Save
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-xs"
                  onClick={() => {
                    setIsAddingCollection(false);
                    setNewCollectionName("");
                  }}
                >
                  Cancel
                </Button>
              </div>
            ) : (
              <Button
                variant="outline"
                size="sm"
                className="h-7 rounded-full px-2.5 text-xs text-muted-foreground"
                onClick={() => setIsAddingCollection(true)}
              >
                <HugeiconsIcon icon={PlusSignIcon} className="mr-1 size-3" />
                New Collection
              </Button>
            )}
          </div>

          <div className="flex items-center gap-1">
            {activeCollection.isDefault ? (
              <Button
                variant="ghost"
                size="sm"
                onClick={resetDictionaryToDefaults}
                data-testid="dictionary-reset-defaults"
              >
                {t("settings.voice.dictionary.resetDefaults")}
              </Button>
            ) : (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon-sm" className="size-7">
                    <HugeiconsIcon icon={MoreVerticalIcon} className="size-3.5" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem
                    onClick={() => setEditingCollectionName(activeCollection.name)}
                  >
                    Rename Collection
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    className="text-destructive"
                    onClick={() => removeCollection(activeCollection.id)}
                  >
                    Delete Collection
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
          </div>
        </div>

        {editingCollectionName !== null ? (
          <div className="flex items-center gap-2 py-1">
            <Input
              value={editingCollectionName}
              onChange={(e) => setEditingCollectionName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleRenameCollection();
                } else if (e.key === "Escape") {
                  setEditingCollectionName(null);
                }
              }}
              autoFocus
              className="h-8 max-w-sm text-sm"
            />
            <Button size="sm" onClick={handleRenameCollection}>
              Save
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setEditingCollectionName(null)}
            >
              Cancel
            </Button>
          </div>
        ) : (
          <div className="flex flex-col gap-0.5">
            <h2 className="settings-heading text-sm font-semibold">
              {activeCollection.name} ({activeCollection.words.length})
            </h2>
            <p className="text-xs text-muted-foreground">
              {activeCollection.isDefault
                ? `Standard quality vocabulary including technical terms and words from your profile (${userWords.length} from your profile name).`
                : "Custom vocabulary and snippet expansion shortcuts applied to your dictations."}
            </p>
          </div>
        )}
      </div>

      {/* Add word row */}
      <div className="flex items-center gap-2 py-1.5">
        <Input
          value={newEntry}
          onChange={(e) => setNewEntry(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              handleAddEntry();
            }
          }}
          placeholder="Add word or phrase (or shortcut -> replacement)..."
          className="h-8 flex-1 text-sm"
          aria-label={t("settings.voice.dictionary.newEntryAria")}
        />
        <Button
          variant="outline"
          size="sm"
          className="shrink-0"
          onClick={handleAddEntry}
          disabled={!newEntry.trim()}
        >
          <HugeiconsIcon icon={PlusSignIcon} className="mr-1.5 size-3.5" />
          {t("settings.voice.dictionary.addEntry")}
        </Button>
      </div>

      {/* Batch actions toolbar */}
      {selectedIndices.length > 0 ? (
        <div className="flex items-center justify-between rounded-lg border border-border/60 bg-muted/50 px-3 py-1.5 text-xs">
          <span className="font-medium">
            {selectedIndices.length} {selectedIndices.length === 1 ? "word" : "words"} selected
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs text-muted-foreground"
              onClick={toggleSelectAll}
            >
              {selectedIndices.length === activeCollection.words.length
                ? "Deselect All"
                : "Select All"}
            </Button>

            {otherCollections.length > 0 ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm" className="h-7 text-xs">
                    Move to...
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  {otherCollections.map((target) => (
                    <DropdownMenuItem
                      key={target.id}
                      onClick={() => handleBatchMove(target.id)}
                    >
                      {target.name}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            ) : null}

            <Button
              variant="destructive"
              size="sm"
              className="h-7 text-xs"
              onClick={handleBatchDelete}
            >
              <HugeiconsIcon icon={Delete02Icon} className="mr-1 size-3" />
              Delete Selected
            </Button>
          </div>
        </div>
      ) : activeCollection.words.length > 0 ? (
        <div className="flex items-center justify-end px-1">
          <button
            type="button"
            onClick={toggleSelectAll}
            className="text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            Select All
          </button>
        </div>
      ) : null}

      {/* Word list */}
      <div className="flex flex-col">
        {activeCollection.words.length === 0 ? (
          <div className="py-8 text-center text-xs text-muted-foreground italic">
            This collection is empty. Type a word above and press Enter.
          </div>
        ) : (
          activeCollection.words.map((entry, index) => (
            <div
              // biome-ignore lint/suspicious/noArrayIndexKey: entries are editable in place
              key={index}
              className="flex items-center gap-2 py-1.5"
            >
              <Checkbox
                checked={selectedIndices.includes(index)}
                onCheckedChange={() => toggleSelectIndex(index)}
                aria-label={`Select ${entry}`}
                className="size-4 shrink-0"
              />
              <Input
                value={entry}
                onChange={(e) =>
                  updateWordInCollection(activeCollection.id, index, e.target.value)
                }
                onBlur={(e) => {
                  if (
                    e.relatedTarget instanceof HTMLElement &&
                    e.relatedTarget.dataset.removeIndex === String(index)
                  ) {
                    return;
                  }
                  commitWordInCollection(activeCollection.id, index);
                }}
                className="h-8 flex-1 text-sm font-mono text-xs"
                aria-label={t("settings.voice.dictionary.entryAria", {
                  index: index + 1,
                })}
              />
              <Button
                variant="ghost"
                size="icon"
                className="size-8 shrink-0 text-muted-foreground hover:text-destructive"
                data-remove-index={index}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => removeWordFromCollection(activeCollection.id, index)}
                aria-label={t("settings.voice.dictionary.removeEntryAria", {
                  index: index + 1,
                })}
              >
                <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
              </Button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
