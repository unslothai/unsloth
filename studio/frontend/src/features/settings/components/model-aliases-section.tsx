// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { kvCacheDtypeOptions } from "@/features/model-picker/model-config/per-model-config";
import { useLlamaCppBackend } from "@/hooks/use-llama-backend";
import { useT } from "@/i18n";
import { TrashIcon } from "@/lib/hugeicons-derived";
import {
  SPECULATIVE_TYPE_LABELS,
  SPECULATIVE_TYPES,
} from "@/lib/speculative-modes";
import {
  PlusSignIcon,
  SlidersHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useId, useState } from "react";
import {
  type AliasLoadSettings,
  type ModelAlias,
  loadModelAliases,
  saveModelAliases,
} from "../api/model-aliases";
import { listOpenAIModels } from "../api/openai-models";
import { SettingsSection } from "./settings-section";

const SAVED_SETTING = "saved";

function sameAliases(a: ModelAlias[], b: ModelAlias[]): boolean {
  return (
    a.length === b.length &&
    a.every(
      (alias, i) =>
        alias.name === b[i].name &&
        alias.model === b[i].model &&
        alias.load_settings.max_seq_length ===
          b[i].load_settings.max_seq_length &&
        alias.load_settings.kv_cache_dtype ===
          b[i].load_settings.kv_cache_dtype &&
        alias.load_settings.speculative_type ===
          b[i].load_settings.speculative_type,
    )
  );
}

function loadSettingsSummary(settings: AliasLoadSettings): string {
  const speculative = settings.speculative_type;
  return [
    settings.max_seq_length,
    settings.kv_cache_dtype,
    speculative &&
      ((SPECULATIVE_TYPE_LABELS as Record<string, string>)[speculative] ??
        speculative),
  ]
    .filter(Boolean)
    .join(" · ");
}

function LoadSettingSelect({
  label,
  value,
  options,
  disabled,
  onChange,
}: {
  label: string;
  value: string | undefined;
  options: string[][];
  disabled: boolean;
  onChange: (value: string | undefined) => void;
}) {
  const t = useT();
  return (
    <div className="flex min-w-0 flex-col gap-1">
      {label}
      <Select
        value={value ?? SAVED_SETTING}
        disabled={disabled}
        onValueChange={(next) =>
          onChange(next === SAVED_SETTING ? undefined : next)
        }
      >
        <SelectTrigger aria-label={label} size="sm" className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={SAVED_SETTING}>
            {t("settings.general.modelAliases.savedSetting")}
          </SelectItem>
          {options.map(([option, optionLabel]) => (
            <SelectItem key={option} value={option}>
              {optionLabel}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

export function ModelAliasesSection() {
  const t = useT();
  const listId = useId();
  const llamaBackend = useLlamaCppBackend();
  const [saved, setSaved] = useState<ModelAlias[] | null>(null);
  const [draft, setDraft] = useState<ModelAlias[]>([]);
  const [modelIds, setModelIds] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [openRow, setOpenRow] = useState<number | null>(null);

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

  const updateLoadSettings = (index: number, patch: AliasLoadSettings) => {
    setDraft((rows) =>
      rows.map((row, i) =>
        i === index
          ? { ...row, load_settings: { ...row.load_settings, ...patch } }
          : row,
      ),
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
          <div key={index} className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <Input
                value={row.name}
                placeholder={t("settings.general.modelAliases.namePlaceholder")}
                aria-label={t("settings.general.modelAliases.nameAriaLabel")}
                disabled={saved === null || isSaving}
                onChange={(event) => update(index, { name: event.target.value })}
                className="h-8 w-28 sm:w-40"
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
                aria-label={t("settings.general.modelAliases.loadSettings")}
                title={t("settings.general.modelAliases.loadSettings")}
                aria-expanded={openRow === index}
                disabled={saved === null}
                onClick={() => setOpenRow(openRow === index ? null : index)}
              >
                <HugeiconsIcon
                  icon={SlidersHorizontalIcon}
                  className="size-3.5"
                />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                aria-label={t("settings.general.modelAliases.removeAriaLabel")}
                disabled={saved === null || isSaving}
                onClick={() => {
                  setOpenRow(null);
                  setDraft((rows) => rows.filter((_, i) => i !== index));
                }}
              >
                <HugeiconsIcon icon={TrashIcon} className="size-3.5" />
              </Button>
            </div>
            {openRow === index ? null : (
              <span className="px-3 text-xs text-muted-foreground tabular-nums empty:hidden">
                {loadSettingsSummary(row.load_settings)}
              </span>
            )}
            {openRow === index ? (
              <div className="grid grid-cols-1 gap-x-3 gap-y-2 rounded-xl bg-muted/50 p-3 text-xs text-muted-foreground sm:grid-cols-3">
                <label className="flex min-w-0 flex-col gap-1">
                  {t("settings.general.modelAliases.contextLength")}
                  <Input
                    type="number"
                    min={1}
                    value={row.load_settings.max_seq_length ?? ""}
                    placeholder={t("settings.general.modelAliases.savedSetting")}
                    disabled={saved === null || isSaving}
                    onChange={(event) =>
                      updateLoadSettings(index, {
                        max_seq_length: event.target.value
                          ? Number(event.target.value)
                          : undefined,
                      })
                    }
                    className="h-8"
                  />
                </label>
                <LoadSettingSelect
                  label={t("settings.general.modelAliases.kvCacheType")}
                  value={row.load_settings.kv_cache_dtype}
                  options={[
                    "f16",
                    ...kvCacheDtypeOptions(
                      llamaBackend,
                      row.load_settings.kv_cache_dtype,
                    ),
                  ].map((dtype) => [dtype, dtype])}
                  disabled={saved === null || isSaving}
                  onChange={(kv_cache_dtype) =>
                    updateLoadSettings(index, { kv_cache_dtype })
                  }
                />
                <LoadSettingSelect
                  label={t("settings.general.modelAliases.speculativeDecoding")}
                  value={row.load_settings.speculative_type}
                  options={SPECULATIVE_TYPES.map((type) => [
                    type,
                    SPECULATIVE_TYPE_LABELS[type],
                  ])}
                  disabled={saved === null || isSaving}
                  onChange={(speculative_type) =>
                    updateLoadSettings(index, { speculative_type })
                  }
                />
                <span className="sm:col-span-3">
                  {t("settings.general.modelAliases.loadSettingsHint")}
                </span>
              </div>
            ) : null}
          </div>
        ))}
        <div className="flex items-center justify-between gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={saved === null || isSaving}
            onClick={() =>
              setDraft((rows) => [
                ...rows,
                { name: "", model: "", load_settings: {} },
              ])
            }
          >
            <HugeiconsIcon icon={PlusSignIcon} className="mr-1.5 size-3.5" />
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
