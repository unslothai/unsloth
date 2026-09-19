// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { FULL_ACCESS_WARNING } from "./permission-mode-select";

// The danger-confirmation dialog. Mounted once at the chat-page root, not inside a Composer or the
// menu, and driven by global store state, so it works for both the main and shared composers,
// never duplicates in Compare mode, and never leaves the composer popovers frozen open.
export function BypassPermissionsConfirmDialog() {
  const open = useChatRuntimeStore((s) => s.bypassConfirmOpen);
  const setOpen = useChatRuntimeStore((s) => s.setBypassConfirmOpen);
  const setPermissionMode = useChatRuntimeStore((s) => s.setPermissionMode);

  return (
    <AlertDialog open={open} onOpenChange={setOpen}>
      <AlertDialogContent size="sm">
        <AlertDialogHeader>
          <AlertDialogTitle>Enable Full access?</AlertDialogTitle>
          <AlertDialogDescription>
            {FULL_ACCESS_WARNING}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            className="!bg-destructive !text-destructive-foreground hover:!bg-destructive/90"
            onClick={() => {
              setPermissionMode("full");
              setOpen(false);
            }}
          >
            I understand
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
