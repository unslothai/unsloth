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
import { useI18n } from "@/features/i18n";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

export function AboutTab() {
  const { t } = useI18n();
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
        <h1 className="text-lg font-semibold font-heading">
          {t("settings.about.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.about.subtitle")}
        </p>
      </header>

      <SettingsSection title={t("settings.about.studio")}>
        <SettingsRow label={t("settings.about.version")}>
          <code className="font-mono text-xs text-muted-foreground">{version}</code>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.about.updates")}>
        <div className="py-2">
          <UpdateStudioInstructions defaultShell={defaultShell} showTitle={false} />
        </div>
      </SettingsSection>

      <SettingsSection title={t("settings.about.help")}>
        <SettingsRow label={t("settings.about.documentation")}>
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
        <SettingsRow label={t("settings.about.feedback")}>
          <a
            href="https://github.com/unslothai/unsloth/issues"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            <HugeiconsIcon icon={MessageNotification01Icon} className="size-3.5" />
            {t("settings.about.reportIssue")}
            <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
          </a>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.about.dangerZone")}>
        <SettingsRow
          destructive
          label={t("settings.about.shutdown.label")}
          description={t("settings.about.shutdown.description")}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShutdownOpen(true)}
            className="text-destructive hover:text-destructive hover:border-destructive/60"
          >
            <HugeiconsIcon icon={Cancel01Icon} className="size-3.5 mr-1.5" />
            {t("settings.about.shutdown.cta")}
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
