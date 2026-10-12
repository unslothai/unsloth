// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { useChatRuntimeStore } from "@/features/chat";
import { Tick02Icon } from "@/lib/tick-icon";
import { CursorInWindowIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";
import { useBrowserAgentStore } from "./agent-store";

/** Like MCP: the model needs Studio's tool loop, so only a loaded model without tools rules it out. */
function useBrowserAgentDisabled(): boolean {
  return useChatRuntimeStore((s) => !!s.params.checkpoint && !s.modelLoading && !s.supportsTools);
}

/** The composer's + menu row that turns browser use on or off. */
export function BrowserAgentMenuItem() {
  const disabled = useBrowserAgentDisabled();
  const enabled = useBrowserAgentStore((state) => state.enabled);
  const on = enabled && !disabled;
  return (
    <DropdownMenuItem
      disabled={disabled}
      className={on ? "text-primary font-medium" : undefined}
      onSelect={() => useBrowserAgentStore.getState().setEnabled(!enabled)}
    >
      <HugeiconsIcon icon={CursorInWindowIcon} strokeWidth={2} />
      Browser use
      {on ? <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="ml-auto" /> : null}
    </DropdownMenuItem>
  );
}

/** Shown while browser use is on; a click turns it off. `glyph` wraps the icon like the other pills. */
export function BrowserAgentPill({ glyph }: { glyph: (icon: ReactNode) => ReactNode }) {
  const disabled = useBrowserAgentDisabled();
  const enabled = useBrowserAgentStore((state) => state.enabled);
  if (!enabled) return null;
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => useBrowserAgentStore.getState().setEnabled(false)}
      className="composer-pill-btn"
      data-pill-label="Browser"
      data-active={disabled ? "false" : "true"}
      aria-label="Disable browser use"
    >
      {glyph(
        <HugeiconsIcon
          icon={CursorInWindowIcon}
          className="size-[calc(17px*var(--ui-space-scale,1))]"
          strokeWidth={2}
        />,
      )}
      <span>Browser</span>
    </button>
  );
}
