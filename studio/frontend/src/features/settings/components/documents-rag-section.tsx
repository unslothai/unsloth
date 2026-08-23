// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useChatRuntimeStore } from "@/features/chat";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { type ReactElement, useEffect, useState } from "react";
import {
  EmbeddingModelBlockedError,
  type EmbeddingModelSettings,
  EmbeddingModelVerificationError,
  loadEmbeddingModelSettings,
  resetEmbeddingModelSettings,
  updateEmbeddingModelSettings,
} from "../api/embedding-model";
import { EmbeddingModelCombobox } from "./embedding-model-combobox";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

/**
 * Which model indexes uploaded documents. Rendered in both General and Data:
 * each mount loads the setting, so the two never disagree.
 */
export function DocumentsRagSection(): ReactElement {
  const t = useT();
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const [embeddingModel, setEmbeddingModel] =
    useState<EmbeddingModelSettings | null>(null);
  const [draftEmbeddingModel, setDraftEmbeddingModel] = useState("");
  const [embeddingModelError, setEmbeddingModelError] = useState<string | null>(
    null,
  );
  // Set after a 409 (unverifiable model); offers "Save anyway".
  const [embeddingModelNeedsForce, setEmbeddingModelNeedsForce] =
    useState(false);
  const [isSavingEmbeddingModel, setIsSavingEmbeddingModel] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void loadEmbeddingModelSettings()
      .then((settings) => {
        if (cancelled) return;
        setEmbeddingModel(settings);
        setDraftEmbeddingModel(settings.embeddingModel);
      })
      .catch((error) => {
        if (cancelled) return;
        setEmbeddingModelError(
          error instanceof Error
            ? error.message
            : t("settings.general.rag.loadError"),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const saveEmbeddingModel = async (force: boolean) => {
    const trimmed = draftEmbeddingModel.trim();
    if (!trimmed) {
      setEmbeddingModelError(t("settings.general.rag.emptyError"));
      return;
    }
    setIsSavingEmbeddingModel(true);
    setEmbeddingModelError(null);
    try {
      const settings = await updateEmbeddingModelSettings(trimmed, {
        hfToken: hfToken || undefined,
        force,
      });
      setEmbeddingModel(settings);
      setDraftEmbeddingModel(settings.embeddingModel);
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
      setEmbeddingModelError(
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
    setEmbeddingModelError(null);
    setEmbeddingModelNeedsForce(false);
    try {
      const settings = await resetEmbeddingModelSettings();
      setEmbeddingModel(settings);
      setDraftEmbeddingModel(settings.embeddingModel);
    } catch (error) {
      setEmbeddingModelError(
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
                setDraftEmbeddingModel(next);
                setEmbeddingModelNeedsForce(false);
                setEmbeddingModelError(null);
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
