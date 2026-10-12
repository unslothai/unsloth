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

// Dictation-only "+" menu fallback: the composer pill is the normal control, but it is hidden
// while recording, so this is the sole way to reach permission mode then.
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
          // Defer past Radix's menu-close focus restoration, or the dropdown grabs focus back
          // and breaks the dialog's focus trap.
          onRequestFullAccess={() =>
            setTimeout(() => setBypassConfirmOpen(true), 0)
          }
        />
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  );
}

// The danger-confirmation dialog. Mounted once at the chat-page root, not inside a Composer or the
// menu, and driven by global store state, so it works for both the main and shared composers,
// never duplicates in Compare mode, and never leaves the composer popovers frozen open.
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
