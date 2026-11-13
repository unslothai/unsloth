// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";

interface StudioInstallLocation {
  path: string;
  is_default: boolean;
  can_change: boolean;
}

// Windows Desktop only: elsewhere, or on an older shell, nothing renders.
export default function InstallLocation({ disabled }: { disabled: boolean }) {
  const t = useT();
  const [location, setLocation] = useState<StudioInstallLocation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const next = await invoke<StudioInstallLocation>(
          "studio_install_location",
        );
        if (!cancelled) setLocation(next);
      } catch {
        // Older shell without the command.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (!location?.can_change) return null;

  async function run(command: string) {
    setBusy(true);
    setError(null);
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const next = await invoke<StudioInstallLocation | null>(command);
      if (next) setLocation(next);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex max-w-[min(560px,90vw)] flex-col items-center gap-1 text-center text-ui-12 text-muted-foreground">
      <p className="break-all">
        {t("desktopStartup.installLocation")}{" "}
        <span className="text-foreground">{location.path}</span>
      </p>
      <div className="flex items-center gap-3">
        <Button
          variant="link"
          size="xs"
          disabled={disabled || busy}
          onClick={() => void run("choose_studio_install_location")}
        >
          {t("desktopStartup.changeLocation")}
        </Button>
        {!location.is_default && (
          <Button
            variant="link"
            size="xs"
            disabled={disabled || busy}
            onClick={() => void run("reset_studio_install_location")}
          >
            {t("desktopStartup.useDefaultLocation")}
          </Button>
        )}
      </div>
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
