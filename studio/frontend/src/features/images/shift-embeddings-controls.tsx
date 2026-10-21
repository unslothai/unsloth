// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { InfoHint } from "@/components/ui/info-hint";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { listDiffusionEmbeddings, type DiffusionEmbeddingInfo } from "./embeddings-api";
import { shiftFieldInvalid, type ShiftControl, type ShiftValue } from "./lib/shift-choice";

// The resolved defaults can be e.g. ln(e^0.69); show them the way a user would type them.
function roundShift(value: number): number {
  return Number(value.toFixed(3));
}

function Label({ label, hint }: { label: string; hint: ReactNode }) {
  return (
    <div className="flex items-center gap-1">
      <label className="text-xs font-medium text-muted-foreground">{label}</label>
      <InfoHint>{hint}</InfoHint>
    </div>
  );
}

function NumberBox({
  ariaLabel,
  value,
  placeholder,
  exclusiveMin,
  onChange,
}: {
  ariaLabel: string;
  value: string;
  placeholder: string;
  exclusiveMin: boolean;
  onChange: (v: string) => void;
}) {
  return (
    <Input
      aria-label={ariaLabel}
      inputMode="decimal"
      className={cn("h-8 text-xs", shiftFieldInvalid(value, exclusiveMin) && "border-destructive")}
      placeholder={placeholder}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

/** The Shift override for flow-matching models; renders nothing when the loaded model offers none. */
export function ShiftField({
  control,
  value,
  onChange,
}: {
  control: ShiftControl | null | undefined;
  value: ShiftValue;
  onChange: (next: ShiftValue) => void;
}) {
  if (!control) return null;
  if (control.node === "sd3") {
    return (
      <div className="flex flex-col gap-1.5 pt-2">
        <Label
          label="Shift"
          hint="Moves denoising effort toward the early, composition-forming steps (higher) or the late, detail steps (lower). Same as ComfyUI's ModelSamplingSD3 / AuraFlow shift. Leave empty for the model's default."
        />
        <NumberBox
          ariaLabel="Shift"
          value={value.shift}
          placeholder={control.shift != null ? `Model default (${roundShift(control.shift)})` : "Model default"}
          exclusiveMin
          onChange={(shift) => onChange({ ...value, shift })}
        />
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-1.5 pt-2">
      <Label
        label="Shift (base / max)"
        hint="Resolution-dependent shift, as ComfyUI's ModelSamplingFlux: base shift applies at 256 latent tokens, max shift at 4096 (1024x1024), interpolated for your size. Higher spends more steps on composition. Empty boxes keep the value Studio already uses for this model (shown in grey)."
      />
      <div className="flex items-center gap-2">
        <NumberBox
          ariaLabel="Base shift"
          value={value.baseShift}
          placeholder={`Base (${roundShift(control.base_shift ?? 0.5)})`}
          exclusiveMin={false}
          onChange={(baseShift) => onChange({ ...value, baseShift })}
        />
        <NumberBox
          ariaLabel="Max shift"
          value={value.maxShift}
          placeholder={`Max (${roundShift(control.max_shift ?? 1.15)})`}
          exclusiveMin={false}
          onChange={(maxShift) => onChange({ ...value, maxShift })}
        />
      </div>
    </div>
  );
}

/** Textual-inversion embeddings in the Studio embeddings folder; clicking one adds its trigger word. */
export function EmbeddingsField({
  family,
  enabled,
  onInsert,
}: {
  family: string | null | undefined;
  enabled: boolean;
  onInsert: (trigger: string, target: "prompt" | "negative") => void;
}) {
  // Keyed by family, so a list fetched for the previous model is never shown for this one.
  const [listing, setListing] = useState<{
    family: string;
    items: DiffusionEmbeddingInfo[];
    dir: string | null;
  } | null>(null);
  useEffect(() => {
    if (!enabled || !family) return;
    let cancelled = false;
    listDiffusionEmbeddings(family)
      .then((res) => {
        if (!cancelled) {
          setListing({ family, items: res.embeddings, dir: res.embeddings_dir ?? null });
        }
      })
      .catch(() => {
        if (!cancelled) setListing({ family, items: [], dir: null });
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, family]);
  if (!enabled || !family) return null;
  const items = listing?.family === family ? listing.items : [];
  const dir = listing?.family === family ? listing.dir : null;
  return (
    <div className="flex flex-col gap-1.5 pt-2">
      <Label
        label="Embeddings"
        hint={
          <>
            Textual inversion embeddings (.safetensors) such as EasyNegative. Put files in{" "}
            {dir ? <code className="break-all">{dir}</code> : "the Studio embeddings folder"}; the file name is the
            trigger word, so writing it in the prompt or negative prompt uses the embedding.
          </>
        }
      />
      {items.length === 0 ? (
        <p className="text-xs text-muted-foreground">No embeddings for this model yet.</p>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {items.map((e) => (
            <div key={e.id} className="flex items-center rounded-md border text-xs">
              <span className="px-2 py-1" title={`${e.vectors} vector${e.vectors === 1 ? "" : "s"}`}>
                {e.trigger}
              </span>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-6 px-1.5 text-xs"
                aria-label={`Add ${e.trigger} to the prompt`}
                onClick={() => onInsert(e.trigger, "prompt")}
              >
                + Prompt
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-6 px-1.5 text-xs"
                aria-label={`Add ${e.trigger} to the negative prompt`}
                onClick={() => onInsert(e.trigger, "negative")}
              >
                + Negative
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
