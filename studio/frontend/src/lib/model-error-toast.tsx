// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ErrorDetails } from "@/components/ui/error-details";
import type { ExternalToast } from "sonner";
import { modelErrorParts } from "./model-error-message";
import { toast } from "./toast";

// Re-shown through Sonner on toggle so its measured height and action position stay current.
export function modelErrorToast(
  message: string,
  options?: ExternalToast,
  diagnostic?: string,
) {
  const parts = modelErrorParts(message);
  const details = diagnostic ?? parts.details;
  const description = options?.description;
  let id = options?.id;
  const show = (expanded = false): string | number => {
    id = toast.error(parts.summary, {
      ...options,
      id,
      closeButton: true,
      duration: expanded ? Number.POSITIVE_INFINITY : options?.duration,
      description: details ? (
        <>
          {typeof description === "function" ? description() : description}
          <ErrorDetails message={details} onToggle={show} />
        </>
      ) : (
        description
      ),
    });
    return id;
  };
  return show();
}
