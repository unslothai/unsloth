// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CookBookIcon, TestTube01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";
import { Button } from "@/components/ui/button";
import type { RecipeExecutionKind } from "../../execution-types";

type RunValidateFloatingControlsProps = {
  runBusy: boolean;
  runDialogKind: RecipeExecutionKind;
  validateLoading: boolean;
  executionLocked: boolean;
  onOpenRunDialog: (kind: RecipeExecutionKind) => void;
  onValidate: () => void;
};

export function RunValidateFloatingControls({
  runBusy,
  runDialogKind,
  validateLoading,
  executionLocked,
  onOpenRunDialog,
  onValidate,
}: RunValidateFloatingControlsProps): ReactElement {
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-3 z-20 flex justify-center">
      <div className="pointer-events-auto flex items-center gap-2">
        <Button
          type="button"
          className="corner-squircle h-11 px-5"
          onClick={() => onOpenRunDialog(runDialogKind)}
          disabled={runBusy}
        >
          <HugeiconsIcon icon={CookBookIcon} className="size-4" />
          {runBusy ? "Running..." : "Run"}
        </Button>
        <Button
          type="button"
          variant="outline"
          className="corner-squircle h-11 px-5"
          onClick={onValidate}
          disabled={validateLoading || executionLocked}
        >
          <HugeiconsIcon icon={TestTube01Icon} className="size-4" />
          {validateLoading ? "Checking..." : "Check"}
        </Button>
      </div>
    </div>
  );
}
