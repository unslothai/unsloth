// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ResolvedControl,
  resolvedSeedKey,
} from "@/lib/resolved-precision";
import { useEffect, useMemo, useRef } from "react";

type ResolvedStatus = {
  resolved?: Record<string, ResolvedControl> | null;
};

export function useResidentPresetLoadConfig<Config>({
  residentKey,
  resolved,
  parse,
  apply,
  hydrated,
  busy,
}: {
  residentKey: string | null;
  resolved?: Record<string, ResolvedControl> | null;
  parse: (status: ResolvedStatus) => Config | null;
  apply: (config?: Config) => void;
  hydrated: boolean;
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
    if (
      busy === "loading" ||
      busy === "unloading" ||
      !hydrated ||
      !seedKey ||
      !config
    ) {
      return;
    }
    const key = `${residentKey}\0${seedKey}`;
    if (handledSeedKey.current === key) {
      return;
    }
    handledSeedKey.current = key;
    // Always, including the first seed of a model this page did not load. Saved settings own the
    // generation recipe, never the load options: those describe the build that is actually
    // resident. Skipping this left the selects reading "auto" over a build running low_vram, and
    // Reapply submits what the selects say, so it silently reloaded away the resident choice.
    apply(config);
  }, [apply, busy, config, hydrated, residentKey, seedKey]);

  return config;
}
