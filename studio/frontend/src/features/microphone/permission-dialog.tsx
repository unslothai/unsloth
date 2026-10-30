// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The app's own microphone consent dialog. It replaces the WebView2 permission
// bubble on the desktop build and is shared by every recorder in the app.

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT } from "@/i18n";
import { MicIcon } from "@/lib/mic-icon";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type MicrophonePermissionReason,
  useMicrophonePermissionStore,
} from "./permission-store";

function reasonLabel(
  reason: MicrophonePermissionReason,
  t: ReturnType<typeof useT>,
): string {
  switch (reason) {
    case "voice-typing":
      return t("settings.voice.voiceTyping.permissionReasonVoiceTyping");
    case "audio":
      return t("settings.voice.voiceTyping.permissionReasonAudio");
    case "settings":
      return t("settings.voice.voiceTyping.permissionReasonSettings");
    default:
      return t("settings.voice.voiceTyping.permissionReasonDictation");
  }
}

export function MicrophonePermissionDialog() {
  const t = useT();
  const open = useMicrophonePermissionStore((state) => state.dialogOpen);
  const reason = useMicrophonePermissionStore((state) => state.reason);
  const allow = useMicrophonePermissionStore((state) => state.allow);
  const deny = useMicrophonePermissionStore((state) => state.deny);

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        // A controlled close after Allow must not turn the answer into a deny.
        if (!next && useMicrophonePermissionStore.getState().dialogOpen) {
          void deny();
        }
      }}
    >
      <DialogContent
        showCloseButton={false}
        className="sm:max-w-[calc(420px*var(--ui-space-scale,1))]"
        data-testid="microphone-permission-dialog"
      >
        <DialogHeader>
          <div className="mb-1 flex size-11 items-center justify-center rounded-2xl bg-primary/10 text-primary ring-1 ring-primary/15">
            <MicIcon className="size-5" />
          </div>
          <DialogTitle className="text-lg">
            {t("settings.voice.voiceTyping.permissionTitle")}
          </DialogTitle>
          <DialogDescription className="text-sm leading-relaxed">
            {t("settings.voice.voiceTyping.permissionDescription")}
          </DialogDescription>
          <span className="mt-1 inline-flex w-fit items-center rounded-full bg-muted px-2.5 py-1 text-ui-11 font-medium text-muted-foreground">
            {reasonLabel(reason, t)}
          </span>
        </DialogHeader>

        <ul className="flex flex-col gap-2.5 text-sm">
          {[
            t("settings.voice.voiceTyping.permissionOnlyWhileDictating"),
            t("settings.voice.voiceTyping.permissionLocalProcessing"),
            t("settings.voice.voiceTyping.permissionChangeAnytime"),
          ].map((item) => (
            <li key={item} className="flex items-start gap-2.5">
              <span className="mt-0.5 flex size-4 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
                <HugeiconsIcon
                  icon={Tick02Icon}
                  className="size-3"
                  strokeWidth={2.5}
                />
              </span>
              <span className="text-muted-foreground leading-snug">{item}</span>
            </li>
          ))}
        </ul>

        <DialogFooter>
          <Button
            variant="ghost"
            onClick={() => {
              void deny();
            }}
            data-testid="microphone-permission-deny"
          >
            {t("settings.voice.voiceTyping.permissionNotNow")}
          </Button>
          <Button
            onClick={() => {
              void allow();
            }}
            data-testid="microphone-permission-allow"
          >
            <MicIcon className="mr-2 size-4" />
            {t("settings.voice.voiceTyping.permissionAllow")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
