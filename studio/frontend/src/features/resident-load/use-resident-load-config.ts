// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ResolvedControl,
  resolvedSeedKey,
} from "@/lib/resolved-precision";
import { useEffect, useMemo, useRef } from "react";
import type { ResidentLoadStatus } from "./resident-load-config";

/**
 * Mirror the resident build's load options into the page's Advanced selects, so what the panel
 * shows is what is loaded, and so a Reapply reloads that build rather than replacing it.
 */
export function useResidentLoadConfig<Config>({
  residentKey,
  resolved,
  parse,
  apply,
  busy,
}: {
  residentKey: string | null;
  resolved?: Record<string, ResolvedControl> | null;
  parse: (status: ResidentLoadStatus) => Config | null;
  apply: (config: Config) => void;
  busy: string | null;
}): Config | null {
  const config = useMemo(() => parse({ resolved }), [parse, resolved]);
  const seedKey = residentKey ? resolvedSeedKey(resolved) : null;
  const handledSeedKey = useRef<string | null>(null);

  useEffect(() => {
    if (!residentKey) {
      handledSeedKey.current = null;
      return;
    }
    // A failed load leaves the previous resident status in place. Busy alone must not clear the
    // handled key and replay that old config over the options the user may want to retry.
    if (busy === "loading" || busy === "unloading" || !seedKey || !config) {
      return;
    }
    const key = `${residentKey}\0${seedKey}`;
    if (handledSeedKey.current === key) {
      return;
    }
    handledSeedKey.current = key;
    apply(config);
  }, [apply, busy, config, residentKey, seedKey]);

  return config;
}
