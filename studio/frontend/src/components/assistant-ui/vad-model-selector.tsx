// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { isAudioCppFolderId } from "@/features/audio/audio-cpp-catalog";
import { audioModelsForTask } from "@/features/audio/catalog";
import { audioRowMatchesWorkflow } from "@/features/audio/picker-filter";
import { ModelSelector } from "@/features/model-picker";
import { AUDIO_CATALOG } from "@/features/model-picker/components/model-selector/model-catalog";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { type FC, useMemo, useState } from "react";

interface VadModelSelectorProps {
  value: string;
  onValueChange: (id: string) => void;
  disabled?: boolean;
  loading?: boolean;
  loaded?: boolean;
  className?: string;
}

const detectRow = (row: Parameters<typeof audioRowMatchesWorkflow>[0]) =>
  audioRowMatchesWorkflow(row, "detect");

/** Voice mode's "Detect speech with" picker: the audio.cpp voice activity detectors. */
export const VadModelSelector: FC<VadModelSelectorProps> = ({
  value,
  onValueChange,
  disabled = false,
  loading = false,
  loaded = false,
  className,
}) => {
  const [open, setOpen] = useState(false);
  const models = useMemo(() => audioModelsForTask("vad"), []);

  return (
    <div
      className={cn(
        "flex min-w-0 items-center gap-1",
        disabled && "pointer-events-none opacity-50",
      )}
      aria-disabled={disabled || undefined}
      title={disabled ? "Select a chat model first" : undefined}
    >
      {loading && <Spinner className="size-3.5 shrink-0 text-amber-500" />}
      <ModelSelector
        models={models}
        rowFilter={detectRow}
        value={value}
        loaded={loaded}
        loadedModelIdOverride={loaded ? value : undefined}
        onValueChange={(id) => {
          if (!isAudioCppFolderId(id)) {
            toast.info(
              "Voice mode detects speech with the GGUF audio detectors only.",
            );
            return;
          }
          onValueChange(id);
        }}
        variant="ghost"
        className={cn("!h-8.5", className)}
        task={["voice-activity-detection"]}
        catalog={AUDIO_CATALOG}
        placeholder="Select speech detector"
        open={!disabled && open}
        onOpenChange={(next) => setOpen(!disabled && next)}
      />
    </div>
  );
};
