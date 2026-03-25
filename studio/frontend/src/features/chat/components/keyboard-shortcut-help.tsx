// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { FC } from "react";

const isMac = typeof navigator !== "undefined" && /Mac|iPod|iPhone|iPad/.test(navigator.userAgent);
const mod = isMac ? "\u2318" : "Ctrl";

const shortcuts = [
  { keys: `${mod}+K`, description: "Open command palette" },
  { keys: `${mod}+Shift+N`, description: "New chat" },
  { keys: `${mod}+Shift+C`, description: "Toggle compare mode" },
  { keys: `${mod}+Shift+S`, description: "Toggle settings" },
  { keys: "Escape", description: "Close dialogs / Cancel" },
  { keys: "?", description: "Show this help" },
];

export const KeyboardShortcutHelp: FC<{
  open: boolean;
  onOpenChange: (open: boolean) => void;
}> = ({ open, onOpenChange }) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>Keyboard Shortcuts</DialogTitle>
          <DialogDescription>Quick actions for power users</DialogDescription>
        </DialogHeader>
        <div className="space-y-1">
          {shortcuts.map(({ keys, description }) => (
            <div
              key={keys}
              className="flex items-center justify-between py-1.5 text-sm"
            >
              <span className="text-muted-foreground">{description}</span>
              <kbd className="rounded border bg-muted px-2 py-0.5 font-mono text-xs">
                {keys}
              </kbd>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
};
