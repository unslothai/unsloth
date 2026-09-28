// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { hasRejectedHfToken, subscribeHfTokenRejected } from "@/lib/hf-token-rejection";
import { toast } from "@/lib/toast";
import { useEffect } from "react";

/** One toast when the Hub first refuses the saved token and a read recovers without it. The
 * store notifies once per token, so a page of refused requests is one message, not many. */
export function useHfTokenRejectedToast(): void {
  const t = useT();
  useEffect(
    () =>
      subscribeHfTokenRejected(() => {
        // The store also notifies when a refusal is cleared; that is good news, not a toast.
        if (!hasRejectedHfToken()) return;
        toast.error(t("studio.modelPicker.tokenRejectedTitle"), {
          id: "hf-token-rejected",
          description: t("studio.modelPicker.tokenRejectedAnonymousBody"),
        });
      }),
    [t],
  );
}
