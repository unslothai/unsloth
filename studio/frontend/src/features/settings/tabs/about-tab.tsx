// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { ShutdownDialog } from "@/components/shutdown-dialog";
import { UpdateStudioInstructions } from "../components/update-studio-instructions";
import { usePlatformStore } from "@/config/env";
import { apiUrl } from "@/lib/api-base";
import { removeTrainingUnloadGuard } from "@/features/training/hooks/use-training-unload-guard";
import {
  ArrowUpRight01Icon,
  Book03Icon,
  Cancel01Icon,
  MessageNotification01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

export function AboutTab() {
  const deviceType = usePlatformStore((s) => s.deviceType);
  const defaultShell = deviceType === "windows" ? "windows" : "unix";
  const [shutdownOpen, setShutdownOpen] = useState(false);
  const [version, setVersion] = useState("dev");

  useEffect(() => {
    let canceled = false;

    (async () => {
      try {
        const res = await fetch(apiUrl("/api/health"));
        if (!res.ok) return;
        const data = (await res.json()) as { version?: string };
        if (!canceled && data.version) {
          setVersion(data.version);
        }
      } catch {
        // fall back to dev label
      }
    })();

    return () => {
      canceled = true;
    };
  }, []);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading">About</h1>
        <p className="text-xs text-muted-foreground">
          Unsloth Studio build info and support.
        </p>
      </header>

      <SettingsSection title="Studio">
        <SettingsRow label="Version">
          <code className="font-mono text-xs text-muted-foreground">{version}</code>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title="Updates">
        <div className="py-2">
          <UpdateStudioInstructions defaultShell={defaultShell} showTitle={false} />
        </div>
      </SettingsSection>

      <SettingsSection title="Help">
        <SettingsRow label="Documentation">
          <a
            href="https://unsloth.ai/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            <HugeiconsIcon icon={Book03Icon} className="size-3.5" />
            unsloth.ai/docs
            <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
          </a>
        </SettingsRow>
        <SettingsRow label="Feedback">
          <a
            href="https://github.com/unslothai/unsloth/issues"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            <HugeiconsIcon icon={MessageNotification01Icon} className="size-3.5" />
            Report an issue
            <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
          </a>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title="Danger zone">
        <SettingsRow
          destructive
          label="Shut down Unsloth Studio"
          description="Stops the Studio server process and ends your session."
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShutdownOpen(true)}
            className="text-destructive hover:text-destructive hover:border-destructive/60"
          >
            <HugeiconsIcon icon={Cancel01Icon} className="size-3.5 mr-1.5" />
            Shut down
          </Button>
        </SettingsRow>
      </SettingsSection>

      <ShutdownDialog
        open={shutdownOpen}
        onOpenChange={setShutdownOpen}
        onAfterShutdown={removeTrainingUnloadGuard}
      />
    </div>
  );
}
