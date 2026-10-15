// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuShortcut,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useShortcut, useShortcutLabel } from "@/features/settings";
import { useT } from "@/i18n";
import { Add01Icon, InternetIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { EnterFullViewIcon } from "./icons";
import { useBrowserStore } from "./store";

// Long enough that passing over the button, or crossing the gap to the menu, doesn't toggle it.
const HOVER_OPEN_MS = 150;
const HOVER_CLOSE_MS = 200;

function openNewTab(fullView = false) {
  const state = useBrowserStore.getState();
  state.newTab();
  if (fullView) state.setFullView(true);
}

export function BrowserToggleButton({ active = true }: { active?: boolean }) {
  const t = useT();
  const open = useBrowserStore((state) => state.open);
  const newTabShortcut = useShortcutLabel("newBrowserTab");
  const fullViewShortcut = useShortcutLabel("toggleBrowserFullView");
  const [menuOpen, setMenuOpen] = useState(false);
  const timer = useRef<number | null>(null);
  // Focus given back when the hover menu closes without opening a tab.
  const restore = useRef<HTMLElement | null>(null);
  const openTab = (fullView = false) => {
    clearTimer();
    restore.current = null;
    setMenuOpen(false);
    openNewTab(fullView);
  };
  // The chat stays mounted on other pages: its shortcuts only work while it is shown.
  useShortcut(
    "newBrowserTab",
    (event) => {
      event.preventDefault();
      openTab();
    },
    { enabled: active },
  );
  useShortcut(
    "toggleBrowserFullView",
    (event) => {
      event.preventDefault();
      openTab(true);
    },
    { enabled: active && !open },
  );
  const clearTimer = () => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
  };
  const hoverTo = (next: boolean) => {
    clearTimer();
    timer.current = window.setTimeout(() => {
      if (next && !menuOpen && document.activeElement instanceof HTMLElement) {
        restore.current = document.activeElement;
      }
      setMenuOpen(next);
    }, next ? HOVER_OPEN_MS : HOVER_CLOSE_MS);
  };
  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );
  useEffect(() => {
    if (!open) return;
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
    setMenuOpen(false);
  }, [open]);
  if (open) return null;
  const label = t("browser.newTab");
  const onMouse = (handler: () => void) => (event: { pointerType: string }) => {
    if (event.pointerType === "mouse") handler();
  };
  return (
    <DropdownMenu open={menuOpen} onOpenChange={setMenuOpen} modal={false}>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          aria-label={label}
          onPointerEnter={onMouse(() => hoverTo(true))}
          onPointerLeave={onMouse(() => hoverTo(false))}
          // A press opens a tab rather than the menu; hovering, or the arrow keys, open the menu.
          onPointerDown={(event) => event.preventDefault()}
          onKeyDown={(event) => {
            if (event.key !== "Enter" && event.key !== " ") return;
            event.preventDefault();
            openTab();
          }}
          onClick={() => openTab()}
          className="flex size-[calc(30px*var(--ui-space-scale,1))] cursor-pointer items-center justify-center rounded-[10px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring aria-expanded:bg-nav-surface-hover aria-expanded:text-black dark:hover:text-white dark:aria-expanded:text-white"
        >
          <HugeiconsIcon icon={InternetIcon} strokeWidth={1.75} className="size-[calc(var(--icon-size)*0.95)]" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        sideOffset={6}
        onPointerEnter={onMouse(clearTimer)}
        onPointerLeave={onMouse(() => hoverTo(false))}
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          restore.current?.focus({ preventScroll: true });
          restore.current = null;
        }}
        className="browser-menu library-actions-menu w-56"
      >
        <DropdownMenuItem onSelect={() => openTab()}>
          <HugeiconsIcon icon={Add01Icon} strokeWidth={1.75} className="size-icon" />
          {label}
          {newTabShortcut ? <DropdownMenuShortcut>{newTabShortcut}</DropdownMenuShortcut> : null}
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => openTab(true)}>
          <HugeiconsIcon icon={EnterFullViewIcon} strokeWidth={1.75} className="size-icon" />
          {t("browser.newTabFullView")}
          {fullViewShortcut ? (
            <DropdownMenuShortcut>{fullViewShortcut}</DropdownMenuShortcut>
          ) : null}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
