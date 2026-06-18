// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { Settings02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { InferenceLoadSettingsDialog } from "./inference-load-settings-dialog";

/** Gear button on a quant row that opens the pre-load inference settings dialog.
 * onLoad runs the same load path as clicking the row. */
export function ModelLoadSettingsAction({
  ariaLabel,
  repoId,
  quant,
  maxContext,
  gpuGb,
  systemRamGb,
  onLoad,
}: {
  ariaLabel: string;
  repoId: string;
  quant: string;
  maxContext?: number | null;
  gpuGb?: number;
  systemRamGb?: number;
  onLoad: () => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen(true);
        }}
        aria-label={ariaLabel}
        className={cn(
          "shrink-0 rounded-md p-1 text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground",
        )}
      >
        <HugeiconsIcon
          icon={Settings02Icon}
          strokeWidth={1.75}
          className="size-3"
        />
      </button>

      {open && (
        <InferenceLoadSettingsDialog
          open={open}
          onOpenChange={setOpen}
          repoId={repoId}
          quant={quant}
          maxContext={maxContext}
          gpuGb={gpuGb}
          systemRamGb={systemRamGb}
          onLoad={onLoad}
        />
      )}
    </>
  );
}
