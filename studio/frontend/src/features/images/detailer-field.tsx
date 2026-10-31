// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

import { Input } from "@/components/ui/input";
import { InfoHint } from "@/components/ui/info-hint";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { ParamSlider } from "@/features/chat";

import {
  DETAILER_DEFAULTS,
  DETAILER_HINT,
  DETAILER_LABEL,
  DETAILER_PROMPT_HINT,
  DETAILER_TARGETS_HINT,
  type DetailerOptions,
} from "./lib/detailer";
import { SEGMENTER_STATUS_EVENT, type SegmenterStatus } from "./lib/mask-from-text";
import { getSegmenterStatus } from "./mask-segment-api";

/** SAM 3 status when the model can inpaint, else null; polled during a download so the control appears. */
export function useDetailerSegmenter(modelCapable: boolean, modelKey: string): SegmenterStatus | null {
  const [status, setStatus] = useState<SegmenterStatus | null>(null);
  const seq = useRef(0);
  const read = useCallback(() => {
    const run = ++seq.current;
    getSegmenterStatus()
      .then((next) => {
        if (run === seq.current) setStatus(next);
      })
      .catch(() => {
        if (run === seq.current) setStatus(null);
      });
  }, []);
  useEffect(() => {
    if (!modelCapable) {
      seq.current += 1;
      setStatus(null);
      return;
    }
    read();
    const onStatus = (event: Event) => {
      const next = (event as CustomEvent<SegmenterStatus>).detail;
      if (next) {
        seq.current += 1;
        setStatus(next);
      }
    };
    window.addEventListener(SEGMENTER_STATUS_EVENT, onStatus);
    return () => window.removeEventListener(SEGMENTER_STATUS_EVENT, onStatus);
  }, [modelCapable, modelKey, read]);
  const downloading = modelCapable && status?.download.downloading === true;
  useEffect(() => {
    if (!downloading) return;
    const id = window.setInterval(read, 2000);
    return () => window.clearInterval(id);
  }, [downloading, read]);
  return status;
}

/** Empty means "the image's own"; keeps partial text like "4." while editing. */
function OptionalNumberInput({
  ariaLabel,
  value,
  onChange,
}: {
  ariaLabel: string;
  value: number | null;
  onChange: (next: number | null) => void;
}) {
  const [text, setText] = useState(value === null ? "" : String(value));
  useEffect(() => {
    setText((current) => {
      const n = current.trim() === "" ? null : Number(current);
      return n === value ? current : value === null ? "" : String(value);
    });
  }, [value]);
  return (
    <Input
      aria-label={ariaLabel}
      inputMode="decimal"
      className="h-8 text-xs"
      placeholder="Same as the image"
      value={text}
      onChange={(e) => {
        setText(e.target.value);
        const raw = e.target.value.trim();
        const n = Number(raw);
        onChange(raw === "" || !Number.isFinite(n) || n < 0 ? null : n);
      }}
    />
  );
}

function Row({ label, hint, children }: { label: string; hint: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1">
        <label className="text-xs font-medium text-muted-foreground">{label}</label>
        <InfoHint>{hint}</InfoHint>
      </div>
      {children}
    </div>
  );
}

export function DetailerField({
  available,
  enabled,
  onEnabledChange,
  options,
  onOptionsChange,
}: {
  available: boolean;
  enabled: boolean;
  onEnabledChange: (on: boolean) => void;
  options: DetailerOptions;
  onOptionsChange: (next: DetailerOptions) => void;
}) {
  if (!available) return null;
  const set = <K extends keyof DetailerOptions>(key: K, value: DetailerOptions[K]) =>
    onOptionsChange({ ...options, [key]: value });
  return (
    <div className="flex flex-col gap-2 pt-2" data-testid="images-detailer">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
          {DETAILER_LABEL}
          <InfoHint>{DETAILER_HINT}</InfoHint>
        </span>
        <Switch checked={enabled} onCheckedChange={onEnabledChange} aria-label={DETAILER_LABEL} />
      </div>
      {enabled && (
        <div className="flex flex-col gap-2 rounded-md border border-border/60 px-2.5 py-2">
          <Row label="Fix" hint={DETAILER_TARGETS_HINT}>
            <Input
              aria-label="Detailer targets"
              className="h-8 text-xs"
              placeholder={DETAILER_DEFAULTS.targets}
              value={options.targets}
              onChange={(e) => set("targets", e.target.value)}
            />
          </Row>
          <Row label="Detailer prompt" hint={DETAILER_PROMPT_HINT}>
            <Textarea
              aria-label="Detailer prompt"
              className="min-h-14 text-xs"
              placeholder="Same as the image's prompt"
              value={options.prompt}
              onChange={(e) => set("prompt", e.target.value)}
            />
          </Row>
          <Row label="Detailer negative prompt" hint="Negative prompt for the redraw. Empty uses the image's negative prompt; [SEP] works as in the prompt.">
            <Textarea
              aria-label="Detailer negative prompt"
              className="min-h-10 text-xs"
              placeholder="Same as the image's negative prompt"
              value={options.negativePrompt}
              onChange={(e) => set("negativePrompt", e.target.value)}
            />
          </Row>
          <ParamSlider
            inline={true}
            label="Denoise"
            info="How much each face is redrawn. Lower keeps it closer to the original; FaceDetailer's default is 0.5."
            value={options.denoise}
            min={0.05}
            max={1}
            step={0.05}
            onChange={(v) => set("denoise", v)}
          />
          <ParamSlider
            inline={true}
            label="Crop factor"
            info="How much of the surroundings each redraw sees, times the face's box. More context blends better; less gives the face more pixels."
            value={options.cropFactor}
            min={1}
            max={6}
            step={0.1}
            onChange={(v) => set("cropFactor", v)}
          />
          <ParamSlider
            inline={true}
            label="Guide size"
            info="The face's shorter side is enlarged to this many pixels before the redraw (never past Max size)."
            value={options.guideSize}
            min={256}
            max={1536}
            step={8}
            onChange={(v) => set("guideSize", v)}
          />
          <ParamSlider
            inline={true}
            label="Max size"
            info="Caps the enlargement: an enlarged face's crop stays within this many pixels. A crop already larger is redrawn at its own size, never shrunk, as in ComfyUI's FaceDetailer."
            value={options.maxSize}
            min={256}
            max={2048}
            step={8}
            onChange={(v) => set("maxSize", v)}
          />
          <ParamSlider
            inline={true}
            label="Feather"
            info="Blur, in pixels, where the redraw meets the original."
            value={options.feather}
            min={0}
            max={32}
            step={1}
            onChange={(v) => set("feather", v)}
          />
          <ParamSlider
            inline={true}
            label="Mask grow"
            info="Grow each found mask by this many pixels (negative shrinks it)."
            value={options.dilation}
            min={-32}
            max={64}
            step={1}
            onChange={(v) => set("dilation", v)}
          />
          <ParamSlider
            inline={true}
            label="Drop size"
            info="Faces whose box is this many pixels or fewer on a side are left alone."
            value={options.dropSize}
            min={1}
            max={128}
            step={1}
            onChange={(v) => set("dropSize", v)}
          />
          <ParamSlider
            inline={true}
            label="Confidence"
            info="SAM 3 score a face needs to be fixed."
            value={options.confidence}
            min={0.05}
            max={0.95}
            step={0.05}
            onChange={(v) => set("confidence", v)}
          />
          <Row label="Detailer CFG" hint="Guidance for each redraw. Empty uses the image's guidance.">
            <OptionalNumberInput
              ariaLabel="Detailer CFG"
              value={options.guidance}
              onChange={(v) => set("guidance", v)}
            />
          </Row>
          <Row label="Detailer steps" hint="Steps for each redraw. Empty uses the image's steps.">
            <Input
              aria-label="Detailer steps"
              inputMode="numeric"
              className="h-8 text-xs"
              placeholder="Same as the image"
              value={options.steps === null ? "" : String(options.steps)}
              onChange={(e) => {
                const raw = e.target.value.trim();
                const n = Number(raw);
                set("steps", raw === "" || !Number.isFinite(n) || n < 1 ? null : Math.round(n));
              }}
            />
          </Row>
        </div>
      )}
    </div>
  );
}
