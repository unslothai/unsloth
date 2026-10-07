// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Undo02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { PerModelConfig } from "../model-config/per-model-config";
import {
  type ResettableSetting,
  settingIsDefault,
  settingResetPatch,
} from "../model-config/setting-reset";

/** Shown beside a setting's label only while it differs from the default; doubles as the "changed" mark. */
export function SettingResetButton({
  label,
  setting,
  config,
  update,
}: {
  label: string;
  setting: ResettableSetting;
  config: PerModelConfig;
  update: (patch: Partial<PerModelConfig>) => void;
}) {
  if (settingIsDefault(config, setting)) {
    return null;
  }
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          onClick={() => update(settingResetPatch(setting))}
          aria-label={`Reset ${label} to default`}
          className="flex size-4 shrink-0 items-center justify-center rounded-full text-primary transition-colors hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
        >
          <HugeiconsIcon icon={Undo02Icon} strokeWidth={2} className="size-3" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        Reset to default
      </TooltipContent>
    </Tooltip>
  );
}
