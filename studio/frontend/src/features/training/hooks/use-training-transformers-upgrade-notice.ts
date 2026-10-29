// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useHfTokenStore } from "@/features/hub";
import {
  checkTransformersUpgrade,
  useTransformersUpgradeDialogStore,
} from "@/features/transformers-upgrade";
import { useEffect, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { trainingLoadsIn4Bit } from "../api/mappers";
import {
  type TrainingTransformersUpgradeNotice,
  trainingTransformersUpgradeNotice,
} from "../lib/training-transformers-upgrade";
import {
  hasUpgradeNoticeCache,
  readUpgradeNoticeCache,
  upgradeNoticeCacheKey,
  writeUpgradeNoticeCache,
} from "../lib/training-upgrade-notice-cache";
import { useTrainingConfigStore } from "../stores/training-config-store";

export type { TrainingTransformersUpgradeNotice };

const EMPTY: TrainingTransformersUpgradeNotice = {
  installVersion: null,
  fourBitUnavailable: false,
  installSwitchesTo16Bit: false,
};

/** Discloses a model no installed transformers ships, and the sidecar's 16-bit rule that makes
 * a "QLoRA · 4-bit" preview understate VRAM. */
export function useTrainingTransformersUpgradeNotice(): TrainingTransformersUpgradeNotice {
  const { selectedModel, trainingMethod, modelKnownCached, modelLocalPath } =
    useTrainingConfigStore(
      useShallow((s) => ({
        selectedModel: s.selectedModel,
        trainingMethod: s.trainingMethod,
        modelKnownCached: s.modelKnownCached,
        modelLocalPath: s.modelLocalPath,
      })),
    );
  const hfToken = useHfTokenStore((s) => s.token);
  // An install changes every cached answer, so key on the sidecar generation.
  const sidecarGeneration = useTransformersUpgradeDialogStore(
    (s) => s.sidecarGeneration,
  );
  // Resolved like freshModelCachePin; a known-cached row can have a null path, so the flag travels alone.
  const preferLocalCache = Boolean(modelKnownCached);
  const localPath = (preferLocalCache && modelLocalPath) || null;
  // The cache is the state; this counter only forces a re-render once an answer lands.
  const [, markAnswered] = useState(0);
  const key = selectedModel
    ? upgradeNoticeCacheKey(
        sidecarGeneration,
        selectedModel,
        preferLocalCache,
        localPath,
        hfToken,
      )
    : null;
  const check = key ? readUpgradeNoticeCache(sidecarGeneration, key) : null;

  useEffect(() => {
    if (
      !(key && selectedModel) ||
      hasUpgradeNoticeCache(sidecarGeneration, key)
    ) {
      return;
    }
    let active = true;
    checkTransformersUpgrade(selectedModel, hfToken || null, {
      preferLocalCache,
      modelLocalPath: localPath,
    })
      .then((result) => {
        writeUpgradeNoticeCache(sidecarGeneration, key, result);
        if (active) {
          markAnswered((n) => n + 1);
        }
      })
      // An unreachable check leaves the preview as it was.
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [
    key,
    selectedModel,
    preferLocalCache,
    localPath,
    hfToken,
    sidecarGeneration,
  ]);

  if (!(selectedModel && check)) {
    return EMPTY;
  }
  return trainingTransformersUpgradeNotice(
    check,
    trainingLoadsIn4Bit({ trainingMethod, selectedModel }),
  );
}
