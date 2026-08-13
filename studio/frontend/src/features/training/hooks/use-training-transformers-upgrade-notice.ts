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

/** What the Configure preview must disclose about the transformers this model needs.
 *
 * Two things the preview otherwise gets wrong: a model no installed transformers ships
 * (the run stops on a consent dialog first), and the latest sidecar's 16-bit rule, which
 * makes a "QLoRA · 4-bit" preview understate the run's VRAM by roughly threefold. */
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
  // Every cached answer is about the sidecar that was installed when it was taken: an
  // install lands the release the preview offered AND flips the run to 16-bit. Reading
  // through the generation re-asks after one, so returning to Configure in the same
  // session does not keep offering an install that already ran, or keep promising 4-bit
  // for a run the new overlay loads in 16-bit.
  const sidecarGeneration = useTransformersUpgradeDialogStore(
    (s) => s.sidecarGeneration,
  );
  // The copy the run would load, resolved exactly as freshModelCachePin resolves it for
  // the start: a cached model loads from its pinned snapshot, and the preview must
  // describe that snapshot's architecture rather than whatever the repo publishes today.
  // A known-cached row can carry a null path, and the backend still resolves the pin from
  // the cache roots, so the flag travels on its own rather than being read off the path.
  const preferLocalCache = Boolean(modelKnownCached);
  const localPath = (preferLocalCache && modelLocalPath) || null;
  // The cache is the state. This counter exists only to re-render once an answer
  // lands in it, which keeps every setState out of the effect body: reading the
  // map during render already gives the right answer for a model that has been
  // asked about before, and switching models needs no reset because the key
  // changes with it.
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
      // A preview notice is never worth an error surface: an unreachable check just
      // leaves the preview reading as it did before this hook existed.
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
