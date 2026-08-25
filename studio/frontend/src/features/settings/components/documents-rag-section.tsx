// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useChatRuntimeStore } from "@/features/chat";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { type ReactElement, useEffect, useState } from "react";
import {
  EmbeddingModelBlockedError,
  EmbeddingModelVerificationError,
  resetEmbeddingModelSettings,
  updateEmbeddingModelSettings,
} from "../api/embedding-model";
import { useEmbeddingModelStore } from "../stores/embedding-model-store";
import { EmbeddingModelCombobox } from "./embedding-model-combobox";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

/**
 * Which model indexes uploaded documents. Rendered in both General and Data,
 * off one shared store, so a save on one is what the other reads.
 */
export function DocumentsRagSection(): ReactElement {
  const t = useT();
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const embeddingModel = useEmbeddingModelStore((s) => s.settings);
  const loadError = useEmbeddingModelStore((s) => s.loadError);
  const save = useEmbeddingModelStore((s) => s.save);
  // Null until the user edits, so the field follows a save made on the other
  // surface instead of pinning whatever this mount first read.
  const [draftOverride, setDraftOverride] = useState<string | null>(null);
  const draftEmbeddingModel =
    draftOverride ?? embeddingModel?.embeddingModel ?? "";
  const [saveError, setSaveError] = useState<string | null>(null);
  // The store carries the backend's reason; an unreadable failure has none.
  const loadFailure =
    loadError === null
      ? null
      : loadError || t("settings.general.rag.loadError");
  const embeddingModelError = saveError ?? loadFailure;
  // Set after a 409 (unverifiable model); offers "Save anyway".
  const [embeddingModelNeedsForce, setEmbeddingModelNeedsForce] =
    useState(false);
  const [isSavingEmbeddingModel, setIsSavingEmbeddingModel] = useState(false);

  useEffect(() => {
    void useEmbeddingModelStore.getState().load();
  }, []);

  const saveEmbeddingModel = async (force: boolean) => {
    const trimmed = draftEmbeddingModel.trim();
    if (!trimmed) {
      setSaveError(t("settings.general.rag.emptyError"));
      return;
    }
    setIsSavingEmbeddingModel(true);
    setSaveError(null);
    try {
      const stood = await save(() =>
        updateEmbeddingModelSettings(trimmed, {
          hfToken: hfToken || undefined,
          force,
        }),
      );
      // A later save owns the field now, so this answer says nothing current.
      if (!stood) return;
      setDraftOverride(null);
      setEmbeddingModelNeedsForce(false);
      toast.success(t("settings.general.rag.saved"), {
        description: t("settings.general.rag.reindexWarning"),
      });
    } catch (error) {
      // A hard security block cannot be forced; keep the "save anyway" action hidden.
      if (error instanceof EmbeddingModelBlockedError) {
        setEmbeddingModelNeedsForce(false);
      } else if (error instanceof EmbeddingModelVerificationError) {
        setEmbeddingModelNeedsForce(true);
      }
      setSaveError(
        error instanceof Error
          ? error.message
          : t("settings.general.rag.saveError"),
      );
    } finally {
      setIsSavingEmbeddingModel(false);
    }
  };

  const resetEmbeddingModel = async () => {
    setIsSavingEmbeddingModel(true);
    setSaveError(null);
    setEmbeddingModelNeedsForce(false);
    try {
      if (!(await save(resetEmbeddingModelSettings))) return;
      setDraftOverride(null);
    } catch (error) {
      setSaveError(
        error instanceof Error
          ? error.message
          : t("settings.general.rag.saveError"),
      );
    } finally {
      setIsSavingEmbeddingModel(false);
    }
  };

  return (
    <SettingsSection title={t("settings.general.rag.sectionTitle")}>
      <SettingsRow
        label={t("settings.general.rag.embeddingModel")}
        description={t("settings.general.rag.embeddingModelDescription", {
          defaultModel: embeddingModel?.defaultEmbeddingModel ?? "",
        })}
        className="max-[360px]:flex-col max-[360px]:items-stretch max-[360px]:gap-3"
      >
        <div className="flex flex-col items-end gap-1 max-[360px]:w-full">
          <div className="flex items-center gap-2 max-[360px]:w-full">
            <EmbeddingModelCombobox
              value={draftEmbeddingModel}
              onChange={(next) => {
                setDraftOverride(next);
                setEmbeddingModelNeedsForce(false);
                setSaveError(null);
              }}
              accessToken={hfToken || undefined}
              disabled={!embeddingModel}
              placeholder={t("settings.general.rag.searchPlaceholder")}
              ariaLabel={t("settings.general.rag.embeddingModel")}
              className="w-[220px] max-[360px]:min-w-0 max-[360px]:flex-1"
            />
            <Button
              variant="outline"
              size="sm"
              disabled={
                !embeddingModel ||
                isSavingEmbeddingModel ||
                draftEmbeddingModel.trim() === embeddingModel.embeddingModel
              }
              onClick={() => void saveEmbeddingModel(false)}
            >
              {isSavingEmbeddingModel ? t("common.saving") : t("common.save")}
            </Button>
          </div>
          {embeddingModelError ? (
            <span className="max-w-[300px] text-right text-xs text-destructive">
              {embeddingModelError}
            </span>
          ) : null}
          <div className="flex items-center gap-2">
            {embeddingModelNeedsForce ? (
              <Button
                variant="outline"
                size="sm"
                disabled={isSavingEmbeddingModel}
                onClick={() => void saveEmbeddingModel(true)}
              >
                {t("settings.general.rag.saveAnyway")}
              </Button>
            ) : null}
            {embeddingModel?.isCustom ? (
              <Button
                variant="ghost"
                size="sm"
                disabled={isSavingEmbeddingModel}
                onClick={() => void resetEmbeddingModel()}
              >
                {t("settings.general.rag.resetAction")}
              </Button>
            ) : null}
          </div>
          <span className="max-w-[300px] text-right text-xs text-muted-foreground">
            {t("settings.general.rag.reindexWarning")}
          </span>
        </div>
      </SettingsRow>
    </SettingsSection>
  );
}
