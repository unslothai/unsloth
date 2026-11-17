// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { AudioOptionFields } from "@/features/audio/audio-options-fields";
import { useChatRuntimeStore } from "@/features/chat";
import type { VoiceModelSpec } from "@/features/chat/voice/use-voice-model-spec";
import {
  changedOptionCount,
  voiceModeOptionGroups,
} from "@/features/chat/voice/voice-tts-options";
import { useVoiceSettingsStore, voiceTtsOptionsKey } from "@/features/settings";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type FC, useId } from "react";

const EMPTY = {};

/**
 * The speech model's own options for voice mode (speed, language, then the rest under More options),
 * editable before the model loads. Shared by the voice panel and Settings so both show the same values.
 */
export const VoiceSoundControls: FC<{
  modelId: string;
  spec: VoiceModelSpec;
  className?: string;
}> = ({ modelId, spec, className }) => {
  const t = useT();
  const idPrefix = useId().replace(/:/g, "");
  const key = voiceTtsOptionsKey(modelId);
  const values = useVoiceSettingsStore((s) => s.voiceTtsOptions[key] ?? EMPTY);
  const setOption = useVoiceSettingsStore((s) => s.setVoiceTtsOption);
  const reset = useVoiceSettingsStore((s) => s.resetVoiceTtsOptions);
  const active = useChatRuntimeStore((s) => s.voiceMode === "active");

  if (spec.state === "loading") {
    return (
      <p
        role="status"
        className={cn("flex items-center gap-2 px-2 py-1.5 text-ui-12 text-muted-foreground", className)}
      >
        <Spinner className="size-3.5" />
        {t("voicePanel.optionsLoading")}
      </p>
    );
  }
  if (spec.state === "unavailable") {
    return (
      <p className={cn("px-2 py-1.5 text-ui-12 leading-snug text-muted-foreground", className)}>
        {t("voicePanel.optionsUnavailable")}
      </p>
    );
  }
  const { primary, more } = voiceModeOptionGroups(spec.specs);
  if (primary.length === 0 && more.length === 0) return null;
  const onChange = (name: string, value: Parameters<typeof setOption>[2]) =>
    setOption(modelId, name, value);
  const changedMore = changedOptionCount(more, values);
  const anyChanged = changedOptionCount([...primary, ...more], values) > 0;

  return (
    <div className={cn("flex min-w-0 flex-col gap-3 px-2", className)}>
      {primary.length > 0 && (
        <div className="grid gap-3">
          <AudioOptionFields
            specs={primary}
            values={values}
            onChange={onChange}
            family={spec.family}
            idPrefix={`${idPrefix}-voice-option`}
          />
        </div>
      )}
      {more.length > 0 && (
        <details className="group">
          <summary className="flex cursor-pointer list-none items-center gap-1.5 rounded-md py-1 text-ui-13 font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring [&::-webkit-details-marker]:hidden">
            <HugeiconsIcon
              icon={ArrowDown01Icon}
              strokeWidth={1.75}
              className="size-3.5 -rotate-90 transition-transform group-open:rotate-0 motion-reduce:transition-none"
              aria-hidden="true"
            />
            {t("voicePanel.moreOptions")}
            {changedMore > 0 && (
              <span className="text-ui-12 font-normal">
                {t("voicePanel.moreOptionsChanged", { n: changedMore })}
              </span>
            )}
          </summary>
          <div className="grid gap-3 pt-2">
            <AudioOptionFields
              specs={more}
              values={values}
              onChange={onChange}
              family={spec.family}
              idPrefix={`${idPrefix}-voice-more`}
            />
          </div>
        </details>
      )}
      {(anyChanged || active) && (
        <div className="flex items-center justify-between gap-2">
          <span className="text-ui-12 leading-snug text-muted-foreground">
            {active ? t("voicePanel.optionsApplyNext") : null}
          </span>
          {anyChanged && (
            <button
              type="button"
              onClick={() => reset(modelId)}
              className="shrink-0 rounded-md px-1.5 py-1 text-ui-12 font-medium text-muted-foreground underline-offset-2 transition-colors hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              {t("voicePanel.resetOptions")}
            </button>
          )}
        </div>
      )}
    </div>
  );
};
