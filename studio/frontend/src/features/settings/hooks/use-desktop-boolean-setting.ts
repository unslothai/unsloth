// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";

interface DesktopBooleanSettingOptions {
  enabled: boolean;
  load: () => Promise<boolean | null>;
  save: (enabled: boolean) => Promise<boolean>;
  loadError: string;
  saveError: string;
}

export function useDesktopBooleanSetting({
  enabled,
  load,
  save,
  loadError,
  saveError,
}: DesktopBooleanSettingOptions) {
  const [value, setValue] = useState<boolean | null>(null);
  const [supported, setSupported] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    void load()
      .then((loaded) => {
        if (cancelled) return;
        setValue(loaded);
        setSupported(loaded !== null);
        setError(null);
      })
      .catch((cause) => {
        if (cancelled) return;
        setError(cause instanceof Error ? cause.message : loadError);
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, load, loadError]);

  const update = useCallback(
    async (next: boolean) => {
      setSaving(true);
      setError(null);
      try {
        setValue(await save(next));
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : saveError);
      } finally {
        setSaving(false);
      }
    },
    [save, saveError],
  );

  return { value, supported, error, saving, update };
}
