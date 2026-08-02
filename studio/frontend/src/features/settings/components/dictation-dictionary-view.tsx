// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useT } from "@/i18n";
import {
  ArrowLeft01Icon,
  Delete02Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { useVoiceSettingsStore } from "../stores/voice-settings-store";

// Full-page editor for the dictation dictionary. Kept on its own subpage so a
// long list of entries does not crowd the main Voice settings.
export function DictationDictionaryView({ onBack }: { onBack: () => void }) {
  const t = useT();
  const dictionary = useVoiceSettingsStore((s) => s.dictionary);
  const addDictionaryEntry = useVoiceSettingsStore((s) => s.addDictionaryEntry);
  const updateDictionaryEntry = useVoiceSettingsStore(
    (s) => s.updateDictionaryEntry,
  );
  const commitDictionaryEntry = useVoiceSettingsStore(
    (s) => s.commitDictionaryEntry,
  );
  const removeDictionaryEntry = useVoiceSettingsStore(
    (s) => s.removeDictionaryEntry,
  );
  const [newEntry, setNewEntry] = useState("");

  const handleAddEntry = () => {
    const trimmed = newEntry.trim();
    if (!trimmed) return;
    addDictionaryEntry(trimmed);
    setNewEntry("");
  };

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-center gap-2">
        <button
          type="button"
          onClick={onBack}
          aria-label={t("settings.voice.dictionary.backToVoice")}
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
          {t("settings.voice.dictionary.sectionTitle")}
        </h2>
        <p className="text-xs text-muted-foreground">
          {t("settings.voice.dictionary.sectionDescription")}
        </p>
      </div>

      <div className="flex flex-col">
        {dictionary.map((entry, index) => (
          <div
            // biome-ignore lint/suspicious/noArrayIndexKey: entries are editable in place
            key={index}
            className="flex items-center gap-2 py-1.5"
          >
            <Input
              value={entry}
              onChange={(e) => updateDictionaryEntry(index, e.target.value)}
              // Tabbing to this row's remove button must not commit-splice the
              // row first, which shifts indices and deletes the wrong entry.
              onBlur={(e) => {
                if (
                  e.relatedTarget instanceof HTMLElement &&
                  e.relatedTarget.dataset.removeIndex === String(index)
                ) {
                  return;
                }
                commitDictionaryEntry(index);
              }}
              className="h-8 flex-1 text-sm"
              aria-label={`Dictionary entry ${index + 1}`}
            />
            <Button
              variant="ghost"
              size="icon"
              className="size-8 shrink-0 text-muted-foreground hover:text-destructive"
              data-remove-index={index}
              // Keep the click from blurring an empty input first, which would
              // commit-splice this row and make onClick delete the next one.
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => removeDictionaryEntry(index)}
              aria-label={`Remove dictionary entry ${index + 1}`}
            >
              <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
            </Button>
          </div>
        ))}
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
            placeholder="Jane Doe"
            className="h-8 flex-1 text-sm"
            aria-label="New dictionary entry"
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
      </div>
    </div>
  );
}
