// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo } from "react";
import { modelConfigInstanceKey } from "../model-config/config-signature";
import { residentModelConfigTarget } from "../model-config/model-config-handoff";
import type { PerModelConfig } from "../model-config/per-model-config";
import { ModelConfigPage } from "./model-config-page";

interface SidebarModelConfigProps {
  modelId: string;
  ggufVariant: string | null;
  isGguf: boolean;
  isLora: boolean;
  isDiffusion: boolean;
  nativeContextLength: number | null;
  loadedContextLength: number | null;
  loadedConfig: PerModelConfig;
  onReload: (config: PerModelConfig) => void;
}

export function SidebarModelConfig({
  modelId,
  ggufVariant,
  isGguf,
  isLora,
  isDiffusion,
  nativeContextLength,
  loadedContextLength,
  loadedConfig,
  onReload,
}: SidebarModelConfigProps) {
  const target = useMemo(
    () =>
      residentModelConfigTarget({
        modelId,
        ggufVariant,
        isGguf,
        isLora,
        contextLength: nativeContextLength,
      }),
    [modelId, ggufVariant, isGguf, isLora, nativeContextLength],
  );

  return (
    <ModelConfigPage
      key={modelConfigInstanceKey(modelId, target.ggufVariant, loadedConfig)}
      target={target}
      onRun={onReload}
      loadedConfig={loadedConfig}
      loadedContextLength={loadedContextLength}
      variant="sidebar"
      isDiffusion={isDiffusion}
    />
  );
}
