// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { InfoHint } from "@/components/ui/info-hint";
import { toast } from "@/lib/toast";
import { Suspense, lazy, useEffect, useState } from "react";
import {
  type SamplingDefaults,
  fetchModelSamplingDefaults,
} from "../api/model-sampling-defaults";

const SamplingDefaultsEditor = lazy(() => import("./sampling-defaults-dialog"));

export function SamplingDefaultsRow({
  modelId,
  chatModelId,
  displayName,
  rowClassName,
  labelClassName,
}: {
  /** What the server row is keyed by: the id the run settings are stored under. */
  modelId: string;
  /** The id Chat remembers this model's settings under. */
  chatModelId: string;
  displayName: string;
  rowClassName: string;
  labelClassName: string;
}) {
  const [stored, setStored] = useState<{
    modelId: string;
    saved: SamplingDefaults;
  } | null>(null);
  const [openCount, setOpenCount] = useState(0);
  const [open, setOpen] = useState(false);
  const [opening, setOpening] = useState(false);
  const saved = stored?.modelId === modelId ? stored.saved : null;

  useEffect(() => {
    let cancelled = false;
    fetchModelSamplingDefaults(modelId)
      .then((resolved) => {
        if (!cancelled) {
          setStored({ modelId, saved: resolved });
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [modelId]);

  const openEditor = async () => {
    setOpening(true);
    try {
      const resolved = await fetchModelSamplingDefaults(modelId);
      setStored({ modelId, saved: resolved });
      setOpenCount((count) => count + 1);
      setOpen(true);
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : String(error));
    } finally {
      setOpening(false);
    }
  };

  const savedCount = saved ? Object.keys(saved).length : 0;
  return (
    <div className={rowClassName}>
      <div className="flex min-w-0 items-center gap-1.5">
        <span className={labelClassName}>Sampling Defaults</span>
        <InfoHint>
          Temperature, Top P and the other sampling values this model starts
          with in Chat and uses for API requests that leave them out.
        </InfoHint>
      </div>
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="shrink-0"
        aria-label="Edit sampling defaults"
        disabled={opening}
        onClick={() => {
          openEditor().catch(() => undefined);
        }}
      >
        {savedCount > 0 ? `Custom (${savedCount})` : "Model default"}
      </Button>
      {open ? (
        <Suspense fallback={null}>
          <SamplingDefaultsEditor
            key={openCount}
            open={open}
            onOpenChange={setOpen}
            modelId={modelId}
            chatModelId={chatModelId}
            displayName={displayName}
            stored={saved ?? {}}
            onWritten={(written) => setStored({ modelId, saved: written })}
          />
        </Suspense>
      ) : null}
    </div>
  );
}
