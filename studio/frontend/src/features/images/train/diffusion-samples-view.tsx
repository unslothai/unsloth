// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactElement, useEffect, useMemo, useState } from "react";

import {
  type DiffusionSampleImage,
  diffusionSampleUrl,
  fetchGalleryObjectUrl,
} from "../api";
import { groupSamplesByStep } from "./diffusion-samples";

// One auth-fetched preview (object URL, revoked on unmount or path change).
function SampleImage({
  jobId,
  image,
}: {
  jobId: string;
  image: DiffusionSampleImage;
}): ReactElement {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    let url: string | null = null;
    let cancelled = false;
    fetchGalleryObjectUrl(diffusionSampleUrl(jobId, image.path))
      .then(({ url: u }) => {
        if (cancelled) {
          URL.revokeObjectURL(u);
          return;
        }
        url = u;
        setSrc(u);
      })
      .catch(() => {
        /* a missing image keeps the placeholder */
      });
    return () => {
      cancelled = true;
      if (url) {
        URL.revokeObjectURL(url);
      }
    };
  }, [jobId, image.path]);
  return (
    <figure className="flex min-w-0 flex-col gap-1">
      <div className="aspect-square w-full overflow-hidden rounded-md bg-muted">
        {src ? (
          <img
            src={src}
            alt={image.prompt || "Sample"}
            className="size-full object-cover"
          />
        ) : null}
      </div>
      {image.prompt ? (
        <figcaption
          className="truncate text-ui-11 text-muted-foreground"
          title={image.prompt}
        >
          {image.prompt}
        </figcaption>
      ) : null}
    </figure>
  );
}

// Newest round by default (follows a live run until the user picks a step).
export function DiffusionSamples({
  jobId,
  samples,
}: {
  jobId: string;
  samples: DiffusionSampleImage[] | null | undefined;
}): ReactElement | null {
  const rounds = useMemo(() => groupSamplesByStep(samples), [samples]);
  // Keyed by job so another run's pick never carries over.
  const [picked, setPicked] = useState<{ jobId: string; step: number } | null>(
    null,
  );
  if (rounds.length === 0) {
    return null;
  }
  const pickedStep = picked?.jobId === jobId ? picked.step : null;
  const current =
    rounds.find((r) => r.step === pickedStep) ?? rounds[rounds.length - 1];
  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-semibold">
          Samples at step {current.step}
        </span>
        <span className="text-ui-11 text-muted-foreground">
          Same prompts and seeds every round
        </span>
      </div>
      <div className="flex flex-wrap gap-1">
        {rounds.map((r) => (
          <button
            key={r.step}
            type="button"
            onClick={() => setPicked({ jobId, step: r.step })}
            className={`rounded px-2 py-0.5 text-ui-11 tabular-nums ${
              r.step === current.step
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground hover:text-foreground"
            }`}
          >
            {r.step}
          </button>
        ))}
      </div>
      <div className="grid max-w-3xl grid-cols-2 gap-3 @min-[720px]:grid-cols-4">
        {current.images.map((img) => (
          <SampleImage key={img.path} jobId={jobId} image={img} />
        ))}
      </div>
    </section>
  );
}
