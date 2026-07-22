// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useDeleteConfirmAction } from "./use-delete-confirm-action";

type DeleteSuccessMessage = string | (() => string);

export function useCardDelete({
  action,
  resourceName,
  successMessage,
  onSuccess,
  onSettled,
}: {
  action: () => Promise<void>;
  resourceName: string;
  successMessage: DeleteSuccessMessage;
  onSuccess?: () => void;
  onSettled?: () => void;
}) {
  return useDeleteConfirmAction({
    action,
    successMessage,
    errorToast: (err) => ({
      title: `Failed to delete ${resourceName}`,
      description: err instanceof Error ? err.message : undefined,
    }),
    onSuccess,
    onSettled,
  });
}
