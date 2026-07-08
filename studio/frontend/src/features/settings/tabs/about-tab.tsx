// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ShutdownDialog } from "@/components/shutdown-dialog";
import { Button } from "@/components/ui/button";
import { usePlatformStore } from "@/config/env";
import { getAuthToken } from "@/features/auth";
import { removeTrainingUnloadGuard } from "@/features/training";
import { useHardwareInfo } from "@/hooks/use-hardware-info";
import { useT } from "@/i18n";
import { apiUrl, isTauri } from "@/lib/api-base";
import {
  ArrowUpRight01Icon,
  Book03Icon,
  Cancel01Icon,
  MessageNotification01Icon,
  NewReleasesIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { StudioVersionSection } from "../components/studio-version-section";
import {
  type UpdateInstallSource,
  UpdateStudioInstructions,
} from "../components/update-studio-instructions";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

type ApiObject = Record<string, unknown>;

const INSTALL_SOURCE_KEY = "install_source";

const UPDATE_INSTALL_SOURCES = new Set<UpdateInstallSource>([
  "pypi",
  "editable",
  "local_path",
  "vcs",
  "local_repo",
  "unknown",
]);

function isUpdateInstallSource(value: unknown): value is UpdateInstallSource {
  return (
    typeof value === "string" &&
    UPDATE_INSTALL_SOURCES.has(value as UpdateInstallSource)
  );
}

async function fetchInstallSource(): Promise<UpdateInstallSource> {
  if (isTauri) {
    return "unknown";
  }

  const token = getAuthToken();
  if (!token) {
    return "unknown";
  }

  try {
    const headers = new Headers();
    headers.set("Authorization", `Bearer ${token}`);
    const res = await fetch(apiUrl("/api/studio/install-source"), { headers });
    if (!res.ok) {
      return "unknown";
    }
    const data = (await res.json()) as ApiObject;
    const installSource = data[INSTALL_SOURCE_KEY];
    return isUpdateInstallSource(installSource) ? installSource : "unknown";
  } catch {
    return "unknown";
  }
}

export function AboutTab() {
  const t = useT();
  const deviceType = usePlatformStore((s) => s.deviceType);
  const defaultShell = deviceType === "windows" ? "windows" : "unix";
  const hw = useHardwareInfo();
  const updateSectionRef = useRef<HTMLDivElement | null>(null);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);
  const consumeScrollTarget = useSettingsDialogStore(
    (s) => s.consumeScrollTarget,
  );
  const [shutdownOpen, setShutdownOpen] = useState(false);
  const [installSource, setInstallSource] = useState<
    UpdateInstallSource | "loading"
  >("loading");

  useEffect(() => {
    let canceled = false;

    fetchInstallSource().then((nextInstallSource) => {
      if (!canceled) {
        setInstallSource(nextInstallSource);
      }
    });

    return () => {
      canceled = true;
    };
  }, []);

  useEffect(() => {
    if (scrollTarget !== "about-updates") {
      return;
    }
    const frame = window.requestAnimationFrame(() => {
      updateSectionRef.current?.scrollIntoView({
        block: "start",
        behavior: "smooth",
      });
      consumeScrollTarget("about-updates");
    });
    return () => window.cancelAnimationFrame(frame);
  }, [consumeScrollTarget, scrollTarget]);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold font-heading">
          {t("settings.about.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.about.description")}
        </p>
      </header>

      {/* llama.cpp row lives in the shared version section so it sits with the
          Unsloth/Package rows; the prop keeps it About-only (General passes none). */}
      <StudioVersionSection llamaCppVersion={hw.llamaCpp} />

      <div ref={updateSectionRef} className="scroll-mt-5">
        <SettingsSection title={t("settings.about.updates")}>
          <div className="py-2">
            <UpdateStudioInstructions
              defaultShell={defaultShell}
              installSource={isTauri ? null : installSource}
              showTitle={false}
            />
          </div>
        </SettingsSection>
      </div>

      {hw.gpus.length > 0 || hw.cuda || hw.rocm ? (
        <SettingsSection title={t("settings.about.hardware")}>
          {hw.gpus.map((gpu, i) => (
            <SettingsRow
              // Index key: device order from the backend is stable per request.
              // biome-ignore lint/suspicious/noArrayIndexKey: The hardware API does not expose a stable device id.
              key={i}
              label={
                hw.gpus.length > 1
                  ? `${t("settings.about.gpu")} ${i}`
                  : t("settings.about.gpu")
              }
            >
              <code className="font-mono text-xs text-muted-foreground">
                {gpu.name ?? "—"}
                {gpu.vramTotalGb != null
                  ? ` · ${Math.round(gpu.vramTotalGb)} GiB`
                  : ""}
              </code>
            </SettingsRow>
          ))}
          {hw.cuda || hw.rocm ? (
            <SettingsRow
              label={
                hw.cuda ? t("settings.about.cuda") : t("settings.about.rocm")
              }
            >
              <code className="font-mono text-xs text-muted-foreground">
                {hw.cuda ?? hw.rocm}
              </code>
            </SettingsRow>
          ) : null}
        </SettingsSection>
      ) : null}

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
        <SettingsRow label={t("settings.about.releaseNotes")}>
          <a
            href="https://unsloth.ai/docs/new/changelog"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            <HugeiconsIcon icon={NewReleasesIcon} className="size-3.5" />
            {t("settings.about.whatsNew")}
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
            <HugeiconsIcon
              icon={MessageNotification01Icon}
              className="size-3.5"
            />
            {t("settings.about.reportIssue")}
            <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
          </a>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.about.license.sectionTitle")}>
        <SettingsRow
          label={t("settings.about.license.studioLabel")}
          description={t("settings.about.license.studioDescription")}
        >
          <a
            href="https://github.com/unslothai/unsloth/blob/main/studio/LICENSE.AGPL-3.0"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 font-mono text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            {t("settings.about.license.studioLicense")}
            <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
          </a>
        </SettingsRow>
        <SettingsRow
          label={t("settings.about.license.libraryLabel")}
          description={t("settings.about.license.libraryDescription")}
        >
          <a
            href="https://github.com/unslothai/unsloth/blob/main/LICENSE"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 font-mono text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            {t("settings.about.license.libraryLicense")}
            <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
          </a>
        </SettingsRow>
      </SettingsSection>

      {!isTauri && (
        <SettingsSection title={t("settings.about.dangerZone")}>
          <SettingsRow
            destructive={true}
            label={t("settings.about.shutDownStudio")}
            description={t("settings.about.shutDownStudioDescription")}
          >
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShutdownOpen(true)}
              className="text-destructive hover:text-destructive hover:border-destructive/60"
            >
              <HugeiconsIcon icon={Cancel01Icon} className="size-3.5 mr-1.5" />
              {t("settings.about.shutDown")}
            </Button>
          </SettingsRow>
        </SettingsSection>
      )}

      {!isTauri && (
        <ShutdownDialog
          open={shutdownOpen}
          onOpenChange={setShutdownOpen}
          onAfterShutdown={removeTrainingUnloadGuard}
        />
      )}
    </div>
  );
}
