// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ShieldBanIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import { AlertDialog } from "@/components/ui/alert-dialog";
import {
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  FullAccessConfirmContent,
  PermissionMenuLabel,
  PermissionModeMenuItems,
} from "./permission-mode-select";

// Fallback while recording hides the composer pill, the normal permission-mode control.
export function BypassPermissionsMenuItem() {
  const setBypassConfirmOpen = useChatRuntimeStore(
    (s) => s.setBypassConfirmOpen,
  );

  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger>
        <HugeiconsIcon icon={ShieldBanIcon} strokeWidth={2} />
        Tool permissions
      </DropdownMenuSubTrigger>
      <DropdownMenuSubContent className="unsloth-plus-menu w-[calc(330px*var(--ui-space-scale,1))]">
        <PermissionMenuLabel sandboxControls />
        <PermissionModeMenuItems
          // Defer past Radix focus restoration, or the dropdown steals focus from the dialog trap.
          onRequestFullAccess={() =>
            setTimeout(() => setBypassConfirmOpen(true), 0)
          }
        />
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  );
}

// Mounted once at the chat-page root and store-driven, so it never duplicates in Compare.
export function BypassPermissionsConfirmDialog() {
  const open = useChatRuntimeStore((s) => s.bypassConfirmOpen);
  const setOpen = useChatRuntimeStore((s) => s.setBypassConfirmOpen);
  const setPermissionMode = useChatRuntimeStore((s) => s.setPermissionMode);

  return (
    <AlertDialog open={open} onOpenChange={setOpen}>
      <FullAccessConfirmContent
        onConfirm={() => {
          setPermissionMode("full");
          setOpen(false);
        }}
        onClose={() => setOpen(false)}
      />
    </AlertDialog>
  );
}
