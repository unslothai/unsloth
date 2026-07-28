// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PICKER_FOCUS_VISIBLE_CLASS } from "@/components/resource-picker/picker-focus";
import { cn } from "@/lib/utils";

export const TRAIN_PICKER_TRIGGER_CLASS = cn(
  "hub-menu-trigger field-soft inline-flex h-9 cursor-pointer select-none items-center gap-1.5 rounded-[12px] px-3 text-ui-12p5 text-muted-foreground transition-colors",
  PICKER_FOCUS_VISIBLE_CLASS,
);
