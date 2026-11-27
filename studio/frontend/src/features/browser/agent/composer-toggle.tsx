// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { useChatRuntimeStore } from "@/features/chat";
import { Tick02Icon } from "@/lib/tick-icon";
import { CursorInWindowIcon } from "@/lib/browser-use-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";
import { useBrowserPanelAvailable } from "../panel-availability";
import { useBrowserAgentStore } from "./agent-store";

/** Off when the model lacks tools, as MCP is, or the chat has no panel (mobile, compare). */
export function useBrowserAgentDisabled(): boolean {
  const noTools = useChatRuntimeStore((s) => !!s.params.checkpoint && !s.modelLoading && !s.supportsTools);
  // Always call both hooks: short-circuit skips one, and React crashes on changed hook order.
  const panelAvailable = useBrowserPanelAvailable();
  return noTools || !panelAvailable;
}

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
