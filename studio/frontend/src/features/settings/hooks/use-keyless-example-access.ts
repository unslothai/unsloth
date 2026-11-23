// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { useEffect, useState } from "react";
import {
  type KeylessApiAccessExposure,
  type KeylessApiAccessScope,
  loadKeylessApiAccess,
} from "../api/keyless-api-access";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

export type KeylessExampleAccess = {
  scope: KeylessApiAccessScope;
  tools: boolean;
  exposure: KeylessApiAccessExposure | null;
};

/** The keyless policy the examples' placeholder key follows; owner-only route, else "off". */
export function useKeylessExampleAccess(isOwner: boolean): KeylessExampleAccess {
  const [keyless, setKeyless] = useState<KeylessExampleAccess>({
    scope: "off",
    tools: false,
    exposure: null,
  });
  // Tunnel changes exposure; Settings edits the policy while the API Monitor stays mounted behind it.
  const cloudflareUrl = usePlatformStore((state) => state.cloudflareUrl);
  const settingsOpen = useSettingsDialogStore((state) => state.open);
  // biome-ignore lint/correctness/useExhaustiveDependencies: cloudflareUrl and settingsOpen are triggers, not inputs
  useEffect(() => {
    if (!isOwner) {
      return;
    }
    let cancelled = false;
    loadKeylessApiAccess()
      .then((next) => {
        if (!cancelled) {
          setKeyless({
            scope: next.suspendedByAccounts ? "off" : next.scope,
            tools: next.tools,
            exposure: next.exposure,
          });
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [isOwner, cloudflareUrl, settingsOpen]);
  return keyless;
}
