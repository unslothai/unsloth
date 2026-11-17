// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ExternalToast, toast } from "@/lib/toast";

type ToastId = string | number;

export interface MediaLoadToast<P> {
  /** Raise this load's toast, on a handed-over toast id when given, replacing any earlier one. */
  open: (progress: P, id?: ToastId) => void;
  update: (progress: P) => void;
  dismiss: () => void;
  /** What the toast's Cancel runs; set after render, since the handler is built after the toast. */
  setCancel: (onCancel: () => void) => void;
}

export function createMediaLoadToast<P>({
  toastOptions,
  onClosedChange,
}: {
  toastOptions: (progress: P, cancel: () => void) => ExternalToast;
  /** The progress for the inline status while the user has the toast closed, else null. */
  onClosedChange: (progress: P | null) => void;
}): MediaLoadToast<P> {
  let id: ToastId | null = null;
  let latest: P | null = null;
  let closed = false;
  let onCancel: (() => void) | null = null;
  const cancel = () => onCancel?.();

  // Sonner shows a closed toast again when its id is raised again, so a user close must stop the raises.
  const onDismiss = (dismissed: { id: ToastId }) => {
    // dismiss() clears `id` first, so only a user close gets past this.
    if (dismissed.id !== id) {
      return;
    }
    id = null;
    closed = true;
    onClosedChange(latest);
  };

  const raise = (progress: P, reuse?: ToastId): ToastId =>
    toast(null, { ...toastOptions(progress, cancel), ...(reuse != null ? { id: reuse } : {}), onDismiss });

  const dismiss = () => {
    if (id != null) {
      toast.dismiss(id);
    }
    id = null;
    latest = null;
    if (closed) {
      closed = false;
      onClosedChange(null);
    }
  };

  return {
    open: (progress, reuse) => {
      dismiss();
      latest = progress;
      id = raise(progress, reuse);
    },
    update: (progress) => {
      latest = progress;
      if (id != null) {
        raise(progress, id);
      } else if (closed) {
        onClosedChange(progress);
      }
    },
    dismiss,
    setCancel: (handler) => {
      onCancel = handler;
    },
  };
}
