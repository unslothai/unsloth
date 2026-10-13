// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Shared by the Chat, Images and Video load toasts. The corner close button only hid the toast,
 *  so beside Cancel it read as a second cancel (#11245): both are labelled buttons instead. */
export const MODEL_LOAD_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
  cancelButton:
    "!h-auto !rounded-none !border-0 !bg-transparent !px-1 !text-ui-11 !font-normal !text-muted-foreground hover:!bg-transparent hover:!text-destructive focus-visible:!text-destructive",
  actionButton:
    "!h-auto !rounded-none !border-0 !bg-transparent !px-1 !text-ui-11 !font-normal !text-muted-foreground hover:!bg-transparent hover:!text-foreground focus-visible:!text-foreground",
} as const;

export function modelLoadToastControls(
  onCancel: (() => void) | undefined,
  onHide: () => void,
) {
  return {
    closeButton: false,
    ...(onCancel ? { cancel: { label: "Cancel loading", onClick: onCancel } } : {}),
    action: { label: "Hide", onClick: onHide },
    classNames: MODEL_LOAD_TOAST_CLASSNAMES,
  };
}
