// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import type { Checkbox } from "@/components/ui/checkbox";
import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { toast } from "@/lib/toast";
import { useEffect, useId, useState } from "react";
import {
  type SavedSampling,
  fromApiSampling,
  recommendedSampling,
  sameSampling,
  samplingPayload,
  samplingSaveStatus,
  toApiSampling,
} from "../lib/sampling-memory";
import type { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { InferenceStatusResponse } from "../types/api";
import type { InferenceParams } from "../types/runtime";
import type { resolveQwenThinkingParams } from "../utils/qwen-sampling-table";

// Passed in: importing them here splits shared modules out of the startup bundle.
type Deps = {
  store: typeof useChatRuntimeStore;
  notify: typeof toast;
  Checkbox: typeof Checkbox;
  defaults: InferenceParams;
  qwenThinking: typeof resolveQwenThinkingParams;
};

const DEFAULTS_URL = "/api/settings/model-sampling-defaults";

async function readSavedSampling(
  modelId: string,
  signal: AbortSignal,
): Promise<SavedSampling | null> {
  const query = new URLSearchParams();
  query.set("model_id", modelId);
  const res = await authFetch(`${DEFAULTS_URL}?${query}`, { signal });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to read saved sampling"),
    );
  }
  const body = (await res.json()) as {
    resolved?: Record<string, unknown> | null;
  };
  return fromApiSampling(body.resolved);
}

async function writeSavedSampling(
  modelId: string,
  /** null forgets the model. */
  sampling: SavedSampling | null,
): Promise<void> {
  const res = await authFetch(DEFAULTS_URL, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      model_id: modelId,
      sampling: toApiSampling(sampling),
    }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to save sampling for this model"),
    );
  }
}

export default function SamplingFooter({
  store,
  notify,
  Checkbox,
  defaults,
  qwenThinking,
}: Deps) {
  const params = store((state) => state.params);
  const modelId = params.checkpoint;
  const current = samplingPayload(params);
  const [saved, setSaved] = useState<SavedSampling | null>(null);
  const [remember, setRemember] = useState(false);
  const [busy, setBusy] = useState(false);
  // Until the first read lands, an edit or save would be overwritten by its older result.
  const [loaded, setLoaded] = useState(false);
  const rememberId = useId();
  const statusId = useId();

  useEffect(() => {
    const controller = new AbortController();
    setSaved(null);
    setRemember(false);
    setLoaded(false);
    readSavedSampling(modelId, controller.signal)
      .then((value) => {
        setSaved(value);
        setRemember(value !== null);
      })
      .catch(() => undefined)
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoaded(true);
        }
      });
    return () => controller.abort();
  }, [modelId]);

  const run = (action: () => Promise<void>) => {
    setBusy(true);
    action()
      .catch((error: unknown) => {
        notify.error(error instanceof Error ? error.message : String(error));
      })
      .finally(() => setBusy(false));
  };

  // A reply that lands after a model switch belongs to the previous model.
  const stillOn = () => store.getState().params.checkpoint === modelId;

  const save = () =>
    run(async () => {
      const snapshot = store.getState().params;
      const sampling = samplingPayload(snapshot);
      await writeSavedSampling(modelId, sampling);
      if (stillOn()) {
        setSaved(sampling);
      }
      const remembered = {
        temperature: snapshot.temperature,
        topP: snapshot.topP,
        topK: snapshot.topK,
        minP: snapshot.minP,
        minPMode: snapshot.minPMode,
        repetitionPenalty: snapshot.repetitionPenalty,
        presencePenalty: snapshot.presencePenalty,
      };
      // Writing the memory also sets the live sliders: skip it if one moved during the save.
      const live = store.getState().params;
      if (
        live.checkpoint === modelId &&
        sameSampling(samplingPayload(live), sampling)
      ) {
        store.getState().setRememberedParamsForModel(modelId, remembered);
      }
      notify.success("Sampling saved for this model.");
    });

  const forget = () =>
    run(async () => {
      await writeSavedSampling(modelId, null);
      if (stillOn()) {
        setSaved(null);
      }
      notify.success("Saved sampling forgotten for this model.");
    });

  const reset = () =>
    run(async () => {
      const res = await authFetch(
        `/api/inference/status?${new URLSearchParams({ model: modelId })}`,
      );
      if (!res.ok) {
        throw new Error(
          await readFastApiError(
            res,
            "Failed to read the model's recommended sampling",
          ),
        );
      }
      const status = (await res.json()) as InferenceStatusResponse;
      if (!stillOn()) {
        return;
      }
      const state = store.getState();
      // Qwen3's recommendation depends on the thinking mode, layered on as the load paths do.
      const qwen = status.supports_reasoning
        ? qwenThinking(
            modelId,
            status.reasoning_always_on === true || state.reasoningEnabled,
          )
        : null;
      state.setParams({
        ...state.params,
        ...recommendedSampling(
          status.inference as Record<string, unknown> | undefined,
          defaults,
        ),
        ...qwen,
      });
    });

  const status = samplingSaveStatus(remember, saved, current);
  const unchanged = saved !== null && sameSampling(saved, current);

  return (
    <div className="flex flex-col gap-3.5 border-t border-[rgb(0_0_0_/_calc(0.13*var(--contrast-edge-gain,1)))] pt-4.75 dark:border-[rgb(255_255_255_/_calc(0.09*var(--contrast-edge-gain,1)))]">
      {/* Not inline: the sheet is too narrow. */}
      <div>
        <div className="flex min-w-0 items-center gap-2">
          <Checkbox
            id={rememberId}
            aria-describedby={statusId}
            checked={remember}
            disabled={!loaded}
            onCheckedChange={(checked) => setRemember(checked === true)}
          />
          <label
            htmlFor={rememberId}
            className="cursor-pointer select-none truncate text-ui-13 text-foreground"
          >
            Remember for this model
          </label>
        </div>
        <p
          id={statusId}
          aria-live="polite"
          data-state={status === "Unsaved changes" ? "dirty" : undefined}
          className="pt-1 pl-6 text-ui-11 leading-snug text-muted-foreground empty:pt-0 data-[state=dirty]:text-amber-600 dark:data-[state=dirty]:text-amber-400"
        >
          {status}
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {(remember ? !unchanged : saved !== null) && (
          <Button
            type="button"
            variant={remember ? "dark" : "outline"}
            size="sm"
            disabled={!loaded || busy}
            onClick={remember ? save : forget}
          >
            {remember ? "Save settings" : "Forget settings"}
          </Button>
        )}
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="text-muted-foreground"
          disabled={busy}
          onClick={reset}
        >
          Reset
        </Button>
      </div>
    </div>
  );
}
