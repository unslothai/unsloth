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
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";

/**
 * alerts when search or code is clicked on an external openai-compat connection
 * without opting into the local tool runtime. Mounted once at chat-page root
 * and driven by store state like BypassPermissionsConfirmDialog: both composers
 * share one copy and Compare mode never duplicates it.
 */
export function LocalToolsNoticeDialog() {
  const providerName = useChatRuntimeStore((s) => s.localToolsNoticeProvider);
  const setProviderName = useChatRuntimeStore(
    (s) => s.setLocalToolsNoticeProvider,
  );
  const openSettings = useSettingsDialogStore((s) => s.openDialog);

  return (
    <AlertDialog
      open={providerName !== null}
      onOpenChange={(open) => {
        if (!open) setProviderName(null);
      }}
    >
      <AlertDialogContent size="sm">
        <AlertDialogHeader>
          <AlertDialogTitle>Local tools are off</AlertDialogTitle>
          <AlertDialogDescription>
            Web search and code execution run on this machine. Turn on Local
            tools for {providerName || "this connection"} to enable the Search
            and Code pills.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Close</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              setProviderName(null);
              openSettings("connections");
            }}
          >
            Open settings
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
