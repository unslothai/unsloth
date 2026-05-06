// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ShutdownDialog } from "@/components/shutdown-dialog";
import { Button } from "@/components/ui/button";
import { usePlatformStore } from "@/config/env";
import { getAuthToken } from "@/features/auth";
import { removeTrainingUnloadGuard } from "@/features/training";
import { apiUrl, isTauri } from "@/lib/api-base";
import {
  ArrowUpRight01Icon,
  Book03Icon,
  Cancel01Icon,
  MessageNotification01Icon,
  NewReleasesIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import {
  type UpdateInstallSource,
  UpdateStudioInstructions,
} from "../components/update-studio-instructions";

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

async function fetchStudioVersions(): Promise<{
  packageVersion: string | null;
  studioVersion: string | null;
}> {
  try {
    const res = await fetch(apiUrl("/api/health"));
    if (!res.ok) {
      return { packageVersion: null, studioVersion: null };
    }
    const data = (await res.json()) as ApiObject;
    const packageVersion = data.version;
    const studioVersion = data.studio_version;
    return {
      packageVersion:
        typeof packageVersion === "string" ? packageVersion : null,
      studioVersion: typeof studioVersion === "string" ? studioVersion : null,
    };
  } catch {
    return { packageVersion: null, studioVersion: null };
  }
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
  const deviceType = usePlatformStore((s) => s.deviceType);
  const defaultShell = deviceType === "windows" ? "windows" : "unix";
  const [shutdownOpen, setShutdownOpen] = useState(false);
  const [packageVersion, setPackageVersion] = useState("dev");
  const [studioVersion, setStudioVersion] = useState("dev");
  const [installSource, setInstallSource] = useState<
    UpdateInstallSource | "loading"
  >("loading");

  useEffect(() => {
    let canceled = false;

    fetchStudioVersions().then((nextVersions) => {
      if (canceled) {
        return;
      }
      if (nextVersions.packageVersion) {
        setPackageVersion(nextVersions.packageVersion);
      }
      if (nextVersions.studioVersion) {
        setStudioVersion(nextVersions.studioVersion);
      }
    });

    fetchInstallSource().then((nextInstallSource) => {
      if (!canceled) {
        setInstallSource(nextInstallSource);
      }
    });

    return () => {
      canceled = true;
    };
  }, []);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading">Help</h1>
        <p className="text-xs text-muted-foreground">
          Documentation, release notes, feedback, and Studio build info.
        </p>
      </header>

      <SettingsSection title="Studio">
        <SettingsRow label="Studio Version">
          <code className="font-mono text-xs text-muted-foreground">
            {studioVersion}
          </code>
        </SettingsRow>
        <SettingsRow label="Package Version">
          <code className="font-mono text-xs text-muted-foreground">
            {packageVersion}
          </code>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title="Updates">
        <div className="py-2">
          <UpdateStudioInstructions
            defaultShell={defaultShell}
            installSource={isTauri ? null : installSource}
            showTitle={false}
          />
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
        <SettingsRow label="Release notes">
          <a
            href="https://unsloth.ai/docs/new/changelog"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            <HugeiconsIcon icon={NewReleasesIcon} className="size-3.5" />
            What's new
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
            <HugeiconsIcon
              icon={MessageNotification01Icon}
              className="size-3.5"
            />
            Report an issue
            <HugeiconsIcon icon={ArrowUpRight01Icon} className="size-3" />
          </a>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title="Danger zone">
        <SettingsRow
          destructive={true}
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
