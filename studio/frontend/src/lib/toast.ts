// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Re-export of sonner. Swipe blocking lives on the Toaster via
// `swipeDirections={[]}`, so no per-toast dismissible override.

import { Spinner } from "@/components/ui/spinner";
import { createElement } from "react";

function createLoadingToastIcon() {
  return createElement(Spinner, {
    className: "size-4 text-muted-foreground",
  });
}

export { toast } from "sonner";
export type { ExternalToast } from "sonner";
export { createLoadingToastIcon };
