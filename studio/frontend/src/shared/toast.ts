// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";

export function toastSuccess(message: string): void {
  toast.success(message);
}

export function toastError(message: string, description?: string): void {
  toast.error(message, {
    description,
  });
}
