// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { InfoHint } from "@/components/ui/info-hint";
import { Switch } from "@/components/ui/switch";
import { toast } from "@/lib/toast";
import { useEffect, useId, useState } from "react";
import {
  readGgufPreference,
  writeGgufPreference,
} from "../api/gguf-preference";

/** Opt the selected GGUF quant into API loads. Mount per model/quant to isolate in-flight saves. */
export function ApiQuantDefault({
  modelId,
  variant,
}: {
  modelId: string;
  variant: string;
}) {
  const labelId = useId();
  const [preferred, setPreferred] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    readGgufPreference(modelId, controller.signal).then(
      (value) => {
        if (controller.signal.aborted) return;
        setPreferred(value);
        setReady(true);
      },
      (error: unknown) => {
        if (!controller.signal.aborted) {
          toast.error(
            error instanceof Error
              ? error.message
              : "Failed to read API default quant",
          );
        }
      },
    );
    return () => controller.abort();
  }, [modelId]);

  /** Persist the switch's new value, reflecting it only after the server confirms the write. */
  async function changeDefault(checked: boolean): Promise<void> {
    setSaving(true);
    try {
      setPreferred(
        await writeGgufPreference(modelId, checked ? variant : null),
      );
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to save API default quant",
      );
    } finally {
      setSaving(false);
    }
  }

  return (
    <div
      className="flex items-center justify-between gap-3"
      onChange={(event) => {
        // API preferences are independent of the imported run settings editor.
        event.stopPropagation();
      }}
    >
      <div className="flex min-w-0 items-center gap-1.5">
        <label
          id={labelId}
          className="text-ui-12"
          htmlFor={`${labelId}-switch`}
        >
          Use {variant} by default for API requests
        </label>
        <InfoHint>
          When API model switching is enabled, requests naming this model
          without a quant use {variant}. Explicit quant choices take priority.
          Downloads follow your API settings. Saving this preference does not
          load the model.
        </InfoHint>
      </div>
      <Switch
        id={`${labelId}-switch`}
        className="panel-switch shrink-0"
        aria-labelledby={labelId}
        checked={preferred?.toLowerCase() === variant.toLowerCase()}
        disabled={!ready || saving}
        onCheckedChange={(checked) => {
          void changeDefault(checked);
        }}
      />
    </div>
  );
}
