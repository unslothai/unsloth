// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The app's own microphone consent dialog. It replaces the WebView2 permission
// bubble on the desktop build and is shared by every recorder in the app.

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
import { useT } from "@/i18n";
import { useMicrophonePermissionStore } from "./permission-store";

export function MicrophonePermissionDialog() {
  const t = useT();
  const open = useMicrophonePermissionStore((state) => state.dialogOpen);
  const allow = useMicrophonePermissionStore((state) => state.allow);
  const deny = useMicrophonePermissionStore((state) => state.deny);

  return (
    <AlertDialog
      open={open}
      onOpenChange={(next) => {
        if (!next && useMicrophonePermissionStore.getState().dialogOpen) {
          void deny();
        }
      }}
    >
      <AlertDialogContent
        className="sm:max-w-[calc(380px*var(--ui-space-scale,1))]"
        data-testid="microphone-permission-dialog"
      >
        <AlertDialogHeader>
          <AlertDialogTitle>
            {t("settings.voice.voiceTyping.permissionTitle")}
          </AlertDialogTitle>
          <AlertDialogDescription>
            {t("settings.voice.voiceTyping.permissionDescription")}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel
            onClick={() => {
              void deny();
            }}
            data-testid="microphone-permission-deny"
          >
            {t("settings.voice.voiceTyping.permissionNotNow")}
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              void allow();
            }}
            data-testid="microphone-permission-allow"
          >
            {t("settings.voice.voiceTyping.permissionAllow")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
