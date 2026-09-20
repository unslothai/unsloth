// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";

const MEDIA_MODEL_LOAD_TOAST_CLASSNAMES = {
  toast: "chat-model-load-toast",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
} as const;

export function mediaModelLoadToastOptions({
  id,
  description,
  onCancel,
  onHide,
}: {
  id?: string | number;
  description: ReactNode;
  onCancel?: () => void;
  onHide: () => void;
}) {
  return {
    ...(id != null ? { id } : {}),
    description,
    duration: Infinity,
    closeButton: false,
    ...(onCancel
      ? { cancel: { label: "Cancel loading", onClick: onCancel } }
      : {}),
    action: { label: "Hide", onClick: onHide },
    classNames: MEDIA_MODEL_LOAD_TOAST_CLASSNAMES,
  };
}
