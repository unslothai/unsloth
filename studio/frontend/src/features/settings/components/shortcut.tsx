// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ShortcutId } from "../lib/keyboard-shortcuts";
import { useShortcut } from "../hooks/use-shortcut";

/** One element per slot avoids calling hooks in a loop. */
export function Shortcut({
  id,
  onTrigger,
  enabled,
  skipInTextFields,
}: {
  id: ShortcutId;
  onTrigger: (event: KeyboardEvent) => void;
  enabled?: boolean;
  skipInTextFields?: boolean;
}): null {
  useShortcut(id, onTrigger, { enabled, skipInTextFields });
  return null;
}
