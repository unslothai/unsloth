// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { type OverlapValue, roformerOverlapLogic } from "../separate-policy";
import type { AudioToolPanel } from "./types";

export const roformerOverlapPanel: AudioToolPanel<OverlapValue> = {
  ...roformerOverlapLogic,
  Component: ({ value, onChange, disabled }) => (
    <div className="grid gap-1.5">
      <label
        htmlFor="separate-overlap"
        className="flex items-center justify-between gap-3 text-ui-13 font-medium text-foreground"
      >
        Overlap (slower, cleaner)
        <Switch
          id="separate-overlap"
          checked={value.overlap}
          disabled={disabled}
          onCheckedChange={(overlap) => onChange({ overlap })}
        />
      </label>
      <p className="text-ui-11p5 leading-snug text-muted-foreground">
        Runs the model over overlapping windows and blends them. Off is about
        three times faster. Changing it reloads the model, which takes a few
        seconds.
      </p>
    </div>
  ),
};

export const SEPARATE_TOOL_PANELS = [roformerOverlapPanel] as const;
