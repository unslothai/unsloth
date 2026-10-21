// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";

const EXAMPLE =
  "a sloth hiking, <lora:sloth-style:0.8>\nNegative prompt: blurry\nSteps: 28, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 6, Seed: 12345, Size: 832x1216";

/** Paste generation parameters: AUTOMATIC1111 / Forge / Civitai text (or an Unsloth recipe). */
export function PasteParametersDialog({
  open,
  onOpenChange,
  text,
  onTextChange,
  onApply,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  text: string;
  onTextChange: (text: string) => void;
  onApply: (text: string) => boolean;
}) {
  const apply = () => {
    if (text.trim() && onApply(text)) {
      onTextChange("");
      onOpenChange(false);
    }
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="corner-squircle dialog-soft-surface gap-5 sm:max-w-lg">
        <form
          className="grid gap-5"
          onSubmit={(event) => {
            event.preventDefault();
            apply();
          }}
        >
          <DialogHeader>
            <DialogTitle className="text-ui-21">
              Paste generation parameters
            </DialogTitle>
            <DialogDescription>
              The text AUTOMATIC1111, Forge or Civitai copy: prompt, Negative
              prompt and the Steps line. LoRA tags pick installed LoRAs;
              settings this model cannot use are listed as not applied.
            </DialogDescription>
          </DialogHeader>
          <Textarea
            aria-label="Generation parameters"
            autoFocus
            rows={8}
            spellCheck={false}
            placeholder={EXAMPLE}
            value={text}
            onChange={(e) => onTextChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                apply();
              }
            }}
            className="max-h-[50dvh] min-h-40 resize-y font-mono text-xs"
          />
          <DialogFooter>
            <Button
              type="button"
              variant="ghost"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={!text.trim()}>
              Apply
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
