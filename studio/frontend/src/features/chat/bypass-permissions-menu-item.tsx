// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";

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

// "Bypass Permissions" entry for the composer "+" -> More menu. Mirrors the
// settings toggle: enabling demands the danger warning, disabling is immediate.
// onSelect preventDefault keeps the menu mounted so the warning dialog (which
// lives in this same fragment) survives instead of unmounting with the menu.
export function BypassPermissionsMenuItem() {
  const bypassPermissions = useChatRuntimeStore((s) => s.bypassPermissions);
  const setBypassPermissions = useChatRuntimeStore(
    (s) => s.setBypassPermissions,
  );
  const [dialogOpen, setDialogOpen] = useState(false);

  return (
    <>
      <DropdownMenuItem
        className={
          bypassPermissions ? "text-bypass font-medium" : undefined
        }
        onSelect={(e) => {
          if (bypassPermissions) {
            setBypassPermissions(false);
          } else {
            e.preventDefault();
            setDialogOpen(true);
          }
        }}
      >
        <HugeiconsIcon icon={ShieldBanIcon} strokeWidth={2} />
        Bypass permissions
        {bypassPermissions ? (
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="ml-auto" />
        ) : null}
      </DropdownMenuItem>
      <AlertDialog open={dialogOpen} onOpenChange={setDialogOpen}>
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
                setDialogOpen(false);
              }}
            >
              I understand
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
