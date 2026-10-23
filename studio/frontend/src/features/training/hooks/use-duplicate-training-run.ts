// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { getTrainingRun } from "../api/history-api";
import { useTrainingConfigStore } from "../stores/training-config-store";
import {
  isTrainingStartPending,
  useTrainingRuntimeStore,
} from "../stores/training-runtime-store";

let duplicating = false;

/** Both entry points load the same fresh draft; duplication never starts training. */
export function useDuplicateTrainingRun() {
  const navigate = useNavigate();
  const [pending, setPending] = useState(false);
  const active = useTrainingRuntimeStore(isTrainingStartPending);
  const duplicate = async (runId: string) => {
    if (
      duplicating ||
      isTrainingStartPending(useTrainingRuntimeStore.getState())
    ) {
      return;
    }
    duplicating = true;
    setPending(true);
    const revision = useTrainingConfigStore.getState().userEditRevision;
    try {
      const detail = await getTrainingRun(runId);
      if (isTrainingStartPending(useTrainingRuntimeStore.getState())) {
        return;
      }
      if (useTrainingConfigStore.getState().userEditRevision !== revision) {
        throw new Error(translate("studio.training.duplicateDraftChanged"));
      }
      useTrainingConfigStore.getState().restoreRunConfig(detail.config);
      useTrainingRuntimeStore.setState((state) => ({
        selectedHistoryRunId: null,
        configureRequest: state.configureRequest + 1,
      }));
      useTrainingConfigStore.getState().ensureDatasetChecked();
      await navigate({ to: "/studio" });
    } catch (error) {
      toast.error(translate("studio.training.duplicateFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      duplicating = false;
      setPending(false);
    }
  };
  return { duplicate, disabled: active || pending };
}
