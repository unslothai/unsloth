// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ShieldBanIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

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
import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { Tick02Icon } from "@/lib/tick-icon";

// "Bypass permissions" entry for the composer "+" -> More menu. Mirrors the
// settings toggle: enabling demands the danger warning, disabling is immediate.
// The menu closes normally on select (no preventDefault) -- the warning dialog
// lives outside the menu (BypassPermissionsConfirmDialog, mounted once at the
// chat-page root and driven by the store), so it survives the menu unmounting
// and the "+"/More popovers don't stay frozen.
export function BypassPermissionsMenuItem() {
  const bypassPermissions = useChatRuntimeStore((s) => s.bypassPermissions);
  const setBypassPermissions = useChatRuntimeStore(
    (s) => s.setBypassPermissions,
  );
  const setBypassConfirmOpen = useChatRuntimeStore(
    (s) => s.setBypassConfirmOpen,
  );

  return (
    <DropdownMenuItem
      className={bypassPermissions ? "text-bypass font-medium" : undefined}
      onSelect={() => {
        if (bypassPermissions) {
          setBypassPermissions(false);
        } else {
          // Defer past Radix's menu-close focus restoration: opening the dialog
          // synchronously here lets the dropdown grab focus back and breaks the
          // dialog's focus trap.
          setTimeout(() => setBypassConfirmOpen(true), 0);
        }
      }}
    >
      <HugeiconsIcon icon={ShieldBanIcon} strokeWidth={2} />
      Bypass permissions
      {bypassPermissions ? (
        <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="ml-auto" />
      ) : null}
    </DropdownMenuItem>
  );
}

// The danger-confirmation dialog. Mounted once at the chat-page root (not inside
// a Composer or the menu) and driven by global store state, so it works for both
// the main and shared composers, never duplicates in Compare mode, and confirming
// or cancelling never leaves the composer "+"/More popovers frozen open.
export function BypassPermissionsConfirmDialog() {
  const open = useChatRuntimeStore((s) => s.bypassConfirmOpen);
  const setOpen = useChatRuntimeStore((s) => s.setBypassConfirmOpen);
  const setBypassPermissions = useChatRuntimeStore(
    (s) => s.setBypassPermissions,
  );

  return (
    <AlertDialog open={open} onOpenChange={setOpen}>
      <AlertDialogContent size="sm">
        <AlertDialogHeader>
          <AlertDialogTitle>Enable Bypass permissions?</AlertDialogTitle>
          <AlertDialogDescription>
            Bypass permissions is dangerous since the AI model might delete,
            corrupt your machine, and or cause real world damage to you or the
            world - only accept if you are certain
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            className="!bg-destructive !text-destructive-foreground hover:!bg-destructive/90"
            onClick={() => {
              setBypassPermissions(true);
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
