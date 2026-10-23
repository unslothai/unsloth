// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { type GgufVariantsResponse, listGgufVariants } from "../inventory/api";
import type { HubModelRunSelection } from "../lib/model-run-selection";
import { formatBytes } from "../lib/format";
import { useHfTokenStore } from "../stores/hf-token-store";
import { DotTag } from "./dot-tag";
import { CardDivider, ModelRunActionButton } from "./download-card";

export function MlxPrecisionCard({
  repoId,
  onRun,
  runPending = false,
}: {
  repoId: string;
  onRun?: (selection: HubModelRunSelection) => void;
  runPending?: boolean;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const [listing, setListing] = useState<GgufVariantsResponse | null>(null);
  const [picked, setPicked] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setListing(null);
    setPicked(null);
    listGgufVariants(repoId, hfToken || undefined, {
      signal: controller.signal,
    })
      .then((res) => {
        if (controller.signal.aborted) return;
        if (res.variant_format === "mlx" && res.variants.length > 0) {
          setListing(res);
          setPicked(res.default_variant ?? res.variants[0].quant);
        }
      })
      .catch(() => {
      });
    return () => controller.abort();
  }, [repoId, hfToken]);

  if (!listing) return null;
  const chosen =
    listing.variants.find((v) => v.quant === picked) ?? listing.variants[0];

  return (
    <div className="hub-download-card" data-mlx-precision-card={true}>
      <div className="px-3 pt-2 text-ui-12 text-muted-foreground">
        Precision
      </div>
      <div
        role="radiogroup"
        aria-label="Precision"
        className="flex flex-col px-1.5 pb-1.5 pt-1"
      >
        {listing.variants.map((v) => (
          <button
            key={v.quant}
            type="button"
            role="radio"
            aria-checked={v.quant === chosen.quant}
            onClick={() => setPicked(v.quant)}
            className={cn(
              "flex h-8 items-center gap-2 rounded-md px-1.5 text-left text-ui-12",
              v.quant === chosen.quant
                ? "bg-accent text-foreground"
                : "text-muted-foreground hover:bg-accent/60",
            )}
          >
            <span className="font-medium text-foreground">{v.quant}</span>
            {listing.default_variant === v.quant && (
              <span>same as the repo root</span>
            )}
            <span className="ml-auto flex items-center gap-2">
              {v.downloaded && <DotTag tone="success" label="On device" />}
              <span className="tabular-nums">{formatBytes(v.size_bytes)}</span>
            </span>
          </button>
        ))}
      </div>
      {onRun && (
        <>
          <CardDivider />
          <div className="flex items-center justify-end">
            <ModelRunActionButton
              label={`Configure and run ${repoId} ${chosen.quant}`}
              loading={runPending}
              onClick={() =>
                onRun({
                  ggufVariant: chosen.quant,
                  ggufFilename: chosen.filename,
                  expectedBytes: chosen.size_bytes,
                  precisionFolder: true,
                  downloaded: chosen.downloaded === true,
                })
              }
            />
          </div>
        </>
      )}
    </div>
  );
}
