// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "@/lib/toast";

type DeleteSuccessMessage = string | (() => string);
type DeleteErrorToast =
  | { title: string; description?: string }
  | ((error: unknown) => { title: string; description?: string });

function resolveSuccessMessage(message: DeleteSuccessMessage): string {
  return typeof message === "function" ? message() : message;
}

function resolveErrorToast(
  errorToast: DeleteErrorToast,
  error: unknown,
): { title: string; description?: string } {
  return typeof errorToast === "function" ? errorToast(error) : errorToast;
}

export function useDeleteConfirmAction({
  action,
  successMessage,
  errorToast,
  onSuccess,
  onSettled,
}: {
  action: () => Promise<void>;
  successMessage: DeleteSuccessMessage;
  errorToast: DeleteErrorToast;
  onSuccess?: () => void;
  onSettled?: () => void;
}) {
  const [deleting, setDeleting] = useState(false);
  const inFlightRef = useRef(false);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const runDelete = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    setDeleting(true);
    try {
      await action();
      toast.success(resolveSuccessMessage(successMessage));
      onSuccess?.();
    } catch (error) {
      const resolved = resolveErrorToast(errorToast, error);
      toast.error(resolved.title, { description: resolved.description });
    } finally {
      inFlightRef.current = false;
      if (mountedRef.current) setDeleting(false);
      onSettled?.();
    }
  }, [action, errorToast, onSettled, onSuccess, successMessage]);

  return { deleting, runDelete };
}
