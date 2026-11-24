// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Refresh01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { InfoHint } from "@/components/ui/info-hint";
import { Input } from "@/components/ui/input";
import { ParamSlider } from "@/features/chat";
import type { DiffusionLoraInfo, LoraSpecInput } from "@/features/images/api";
import { LoraAddMenu } from "@/features/images/lora-add-menu";
import { TrashIcon } from "@/lib/hugeicons-derived";

import { MAX_VIDEO_LORAS } from "./video-lora";

/** The Video page's load-time LoRA picker. Adapters are merged into the denoiser when the model loads, so
 * the rows are a pending selection: `reloadNeeded` flags one that differs from what the resident build baked. */
export function VideoLoraField({
  catalog,
  selection,
  onSelectionChange,
  baked,
  reloadNeeded,
  reloadDisabled,
  onReload,
}: {
  catalog: readonly DiffusionLoraInfo[];
  selection: LoraSpecInput[];
  onSelectionChange: (next: LoraSpecInput[]) => void;
  baked: readonly string[];
  reloadNeeded: boolean;
  reloadDisabled: boolean;
  onReload: () => void;
}) {
  const [focusRow, setFocusRow] = useState<number | null>(null);
  const sorted = useMemo(
    () => [...catalog].sort((a, b) => Number(Boolean(b.fine_tuned)) - Number(Boolean(a.fine_tuned))),
    [catalog],
  );
  const taken = useMemo(() => new Set(selection.map((l) => l.id.trim())), [selection]);
  const update = (i: number, patch: Partial<LoraSpecInput>) =>
    onSelectionChange(selection.map((p, j) => (j === i ? { ...p, ...patch } : p)));

  return (
    <div className="flex flex-col gap-1.5" data-testid="video-loras">
      <div className="flex items-center gap-1">
        <span className="text-xs font-medium text-muted-foreground">LoRAs</span>
        <InfoHint>
          Adapters trained for this model, merged into its weights when it loads. Changing the
          set or a weight reloads the model; 1.0 is the full effect, 0 leaves it out.
        </InfoHint>
      </div>
      <div className="space-y-2">
        {sorted.length > 0 && (
          <datalist id="video-lora-suggestions">
            {sorted.map((a) => (
              <option key={a.id} value={a.id}>
                {a.fine_tuned ? `${a.display_name} (fine-tuned)` : a.display_name}
              </option>
            ))}
          </datalist>
        )}
        {selection.map((sel, i) => (
          <div
            // Index key: sel.id is the editable value, so keying on it drops focus.
            // biome-ignore lint/suspicious/noArrayIndexKey: the row position is its identity while editing
            key={i}
            className="space-y-1.5 rounded-lg border border-border bg-muted/30 p-2"
          >
            <div className="flex items-center gap-2">
              <Input
                value={sel.id}
                autoFocus={i === focusRow}
                onFocus={i === focusRow ? () => setFocusRow(null) : undefined}
                list={sorted.length > 0 ? "video-lora-suggestions" : undefined}
                placeholder="LoRA id"
                aria-label={`LoRA ${i + 1}`}
                spellCheck={false}
                autoCapitalize="none"
                autoCorrect="off"
                className="h-8 flex-1 text-xs"
                onChange={(e) => update(i, { id: e.target.value })}
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="size-8 shrink-0"
                aria-label={`Remove LoRA ${i + 1}`}
                onClick={() => onSelectionChange(selection.filter((_, j) => j !== i))}
              >
                <HugeiconsIcon icon={TrashIcon} className="size-3.5" />
              </Button>
            </div>
            <ParamSlider
              inline={true}
              label="Weight"
              value={sel.weight}
              min={0}
              max={2}
              step={0.05}
              onChange={(v) => update(i, { weight: v })}
            />
          </div>
        ))}
        {selection.length < MAX_VIDEO_LORAS && (
          <div className="flex gap-2">
            <LoraAddMenu
              catalog={sorted}
              takenIds={taken}
              onAdd={(spec, focus) => {
                setFocusRow(focus ? selection.length : null);
                onSelectionChange([...selection, spec]);
              }}
            />
          </div>
        )}
        {baked.length > 0 && (
          <p className="text-ui-11 leading-snug text-muted-foreground" data-testid="video-baked-loras">
            Baked into the loaded model: {baked.join(", ")}
          </p>
        )}
        {reloadNeeded && (
          <div className="flex items-center justify-between gap-2 rounded-lg border border-dashed px-3 py-2 text-xs text-muted-foreground">
            <span>LoRAs are merged at load. Reload to apply this selection.</span>
            <Button
              type="button"
              size="sm"
              variant="secondary"
              className="shrink-0 text-xs"
              disabled={reloadDisabled}
              onClick={onReload}
            >
              <HugeiconsIcon icon={Refresh01Icon} className="size-3.5" />
              Reload to apply
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
