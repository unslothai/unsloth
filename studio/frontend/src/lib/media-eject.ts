// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";

/** Images / Video pill eject: the chat pill's unloading toast, one unload at a time. */
export async function ejectWithToast(
  inFlight: { current: boolean },
  unload: () => Promise<boolean>,
): Promise<void> {
  if (inFlight.current) {
    toast.info("Wait for the model to finish unloading.");
    return;
  }
  inFlight.current = true;
  const toastId = toast.loading("Unloading model");
  let unloaded = false;
  try {
    unloaded = await unload();
  } finally {
    inFlight.current = false;
    if (unloaded) toast.success("Model unloaded", { id: toastId, duration: 1200 });
    else toast.dismiss(toastId);
  }
}
