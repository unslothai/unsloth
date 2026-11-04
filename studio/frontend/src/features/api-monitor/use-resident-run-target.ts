// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat";
import { publicModelId } from "@/features/hub";
import {
  splitQuantSuffix,
  useActiveModelConfig,
} from "@/features/model-picker";
import { useMemo } from "react";
import {
  type ResidentRunTarget,
  residentMatchesMonitor,
} from "./slots-settings-request";

// The monitor reports a llama.cpp model as "<id>:<quant>"; Ollama ids carry their own ":<tag>".
export function monitorSelection(activeModel: string | null | undefined): {
  id: string;
  ggufVariant: string | null;
} {
  if (!activeModel) {
    return { id: "", ggufVariant: null };
  }
  const split = activeModel.startsWith("ollama/")
    ? null
    : splitQuantSuffix(activeModel);
  return split
    ? { id: split[0], ggufVariant: split[1] }
    : { id: activeModel, ggufVariant: null };
}

/** The resident's file-picker grant, expired or not, while it is the model the monitor reports. */
export function useResidentFileGrant(
  activeModel: string | null | undefined,
): ResidentRunTarget | null {
  const { checkpoint, isGguf } = useActiveModelConfig();
  const ggufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const isDiffusion = useChatRuntimeStore((s) => s.loadedIsDiffusion);
  const loadId = useChatRuntimeStore((s) => s.activeLoadId);
  const nativePathToken = useChatRuntimeStore((s) => s.activeNativePathToken);
  const nativePathExpiresAtMs = useChatRuntimeStore(
    (s) => s.activeNativePathExpiresAtMs,
  );
  return useMemo(() => {
    const { id } = monitorSelection(activeModel);
    if (
      !nativePathToken ||
      !id ||
      checkpoint == null ||
      publicModelId(checkpoint) !== id
    ) {
      return null;
    }
    return {
      checkpoint,
      ggufVariant,
      isGguf,
      isLora: false,
      isDiffusion,
      loadId,
      nativePathToken,
      nativePathExpiresAtMs,
    };
  }, [
    activeModel,
    checkpoint,
    isGguf,
    ggufVariant,
    isDiffusion,
    loadId,
    nativePathToken,
    nativePathExpiresAtMs,
  ]);
}

/** The resident as Chat's runtime holds it, or null when that is not the model the monitor reports. */
export function useResidentRunTarget(
  activeModel: string | null | undefined,
): ResidentRunTarget | null {
  const { checkpoint, isGguf } = useActiveModelConfig();
  const ggufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const isDiffusion = useChatRuntimeStore((s) => s.loadedIsDiffusion);
  const loadId = useChatRuntimeStore((s) => s.activeLoadId);
  const nativePathToken = useChatRuntimeStore((s) => s.activeNativePathToken);
  const nativePathExpiresAtMs = useChatRuntimeStore(
    (s) => s.activeNativePathExpiresAtMs,
  );
  const models = useChatRuntimeStore((s) => s.models);
  const loras = useChatRuntimeStore((s) => s.loras);
  return useMemo(() => {
    const { id, ggufVariant: monitorVariant } = monitorSelection(activeModel);
    // The picker's own test for handing Run settings the live config (runtimeIsResident).
    if (!id || checkpoint == null || publicModelId(checkpoint) !== id) {
      return null;
    }
    const model = models.find((entry) => entry.id === checkpoint);
    const isLora = model
      ? model.isLora
      : loras.find((entry) => entry.id === checkpoint)?.exportType === "lora";
    const resident = {
      checkpoint,
      ggufVariant,
      isGguf,
      isLora,
      isDiffusion,
      loadId,
      nativePathToken,
      nativePathExpiresAtMs,
    };
    return residentMatchesMonitor(resident, { id, ggufVariant: monitorVariant })
      ? resident
      : null;
  }, [
    activeModel,
    checkpoint,
    isGguf,
    ggufVariant,
    isDiffusion,
    loadId,
    nativePathToken,
    nativePathExpiresAtMs,
    models,
    loras,
  ]);
}
