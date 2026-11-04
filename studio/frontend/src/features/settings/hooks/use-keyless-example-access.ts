// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { useEffect, useState } from "react";
import {
  type KeylessApiAccessExposure,
  type KeylessApiAccessScope,
  loadKeylessApiAccess,
} from "../api/keyless-api-access";

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
  // A tunnel coming or going changes exposure.
  const cloudflareUrl = usePlatformStore((state) => state.cloudflareUrl);
  // biome-ignore lint/correctness/useExhaustiveDependencies: cloudflareUrl is a trigger, not an input
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
  }, [isOwner, cloudflareUrl]);
  return keyless;
}
