// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useT } from "@/i18n";
import { Add01Icon, Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useId, useState } from "react";
import {
  type ModelAlias,
  loadModelAliases,
  saveModelAliases,
} from "../api/model-aliases";
import { listOpenAIModels } from "../api/openai-models";
import { SettingsSection } from "./settings-section";

function sameAliases(a: ModelAlias[], b: ModelAlias[]): boolean {
  return (
    a.length === b.length &&
    a.every(
      (alias, i) => alias.name === b[i].name && alias.model === b[i].model,
    )
  );
}

export function ModelAliasesSection() {
  const t = useT();
  const listId = useId();
  const [saved, setSaved] = useState<ModelAlias[] | null>(null);
  const [draft, setDraft] = useState<ModelAlias[]>([]);
  const [modelIds, setModelIds] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void loadModelAliases()
      .then((loaded) => {
        if (cancelled) {
          return;
        }
        setSaved(loaded);
        setDraft(loaded);
      })
      .catch((loadError) => {
        if (cancelled) {
          return;
        }
        setError(
          loadError instanceof Error
            ? loadError.message
            : t("settings.general.modelAliases.loadError"),
        );
      });
    // Suggestions only: a failed listing leaves the model field free text.
    void listOpenAIModels()
      .then((models) => {
        if (!cancelled) {
          setModelIds(models.map((m) => m.id));
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [t]);

  const aliasNames = new Set(draft.map((a) => a.name.trim().toLowerCase()));
  const suggestions = modelIds.filter(
    (id) => !aliasNames.has(id.toLowerCase()),
  );
  const dirty = saved !== null && !sameAliases(saved, draft);

  const update = (index: number, patch: Partial<ModelAlias>) => {
    setDraft((rows) =>
      rows.map((row, i) => (i === index ? { ...row, ...patch } : row)),
    );
  };

  const handleSave = async () => {
    setIsSaving(true);
    setError(null);
    try {
      const stored = await saveModelAliases(
        draft.filter((row) => row.name.trim() || row.model.trim()),
      );
      setSaved(stored);
      setDraft(stored);
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : t("settings.general.modelAliases.saveError"),
      );
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <SettingsSection
      title={t("settings.general.modelAliases.sectionTitle")}
      description={t("settings.general.modelAliases.description")}
    >
      <datalist id={listId}>
        {suggestions.map((id) => (
          <option key={id} value={id} />
        ))}
      </datalist>
      <div className="flex flex-col gap-2 py-3">
        {draft.map((row, index) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: rows have no id until saved
          <div key={index} className="flex items-center gap-2">
            <Input
              value={row.name}
              placeholder={t("settings.general.modelAliases.namePlaceholder")}
              aria-label={t("settings.general.modelAliases.nameAriaLabel")}
              disabled={saved === null || isSaving}
              onChange={(event) => update(index, { name: event.target.value })}
              className="h-8 w-40"
            />
            <Input
              value={row.model}
              list={listId}
              placeholder={t("settings.general.modelAliases.modelPlaceholder")}
              aria-label={t("settings.general.modelAliases.modelAriaLabel")}
              disabled={saved === null || isSaving}
              onChange={(event) => update(index, { model: event.target.value })}
              className="h-8 min-w-0 flex-1"
            />
            <Button
              variant="ghost"
              size="sm"
              aria-label={t("settings.general.modelAliases.removeAriaLabel")}
              disabled={saved === null || isSaving}
              onClick={() =>
                setDraft((rows) => rows.filter((_, i) => i !== index))
              }
            >
              <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
            </Button>
          </div>
        ))}
        <div className="flex items-center justify-between gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={saved === null || isSaving}
            onClick={() =>
              setDraft((rows) => [...rows, { name: "", model: "" }])
            }
          >
            <HugeiconsIcon icon={Add01Icon} className="mr-1.5 size-3.5" />
            {t("settings.general.modelAliases.add")}
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={!dirty || isSaving}
            onClick={() => void handleSave()}
          >
            {isSaving ? t("common.saving") : t("common.save")}
          </Button>
        </div>
        {error ? (
          <span className="text-xs text-destructive">{error}</span>
        ) : null}
      </div>
    </SettingsSection>
  );
}
