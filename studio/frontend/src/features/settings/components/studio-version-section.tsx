// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthToken, refreshSession } from "@/features/auth";
import { useT } from "@/i18n";
import { apiUrl } from "@/lib/api-base";
import { useEffect, useState } from "react";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

type ApiObject = Record<string, unknown>;
type StudioVersions = {
  packageVersion: string | null;
  studioVersion: string | null;
};

const EMPTY_VERSIONS: StudioVersions = {
  packageVersion: null,
  studioVersion: null,
};

function hasAnyVersion(versions: StudioVersions): boolean {
  return Boolean(versions.packageVersion || versions.studioVersion);
}

async function requestStudioVersions(
  token: string | null,
): Promise<StudioVersions> {
  try {
    const headers = new Headers();
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
    const res = await fetch(apiUrl("/api/health"), { headers });
    if (!res.ok) {
      return EMPTY_VERSIONS;
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
    return EMPTY_VERSIONS;
  }
}

async function fetchStudioVersions(): Promise<StudioVersions> {
  const initialToken = getAuthToken();
  const versions = await requestStudioVersions(initialToken);
  if (hasAnyVersion(versions) || !initialToken) {
    return versions;
  }

  const refreshed = await refreshSession();
  if (!refreshed) {
    return versions;
  }

  return requestStudioVersions(getAuthToken());
}

export function StudioVersionSection() {
  const t = useT();
  const [packageVersion, setPackageVersion] = useState("dev");
  const [studioVersion, setStudioVersion] = useState("dev");

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

    return () => {
      canceled = true;
    };
  }, []);

  return (
    <SettingsSection title="Studio">
      <SettingsRow label={t("settings.about.studioVersion")}>
        <code className="font-mono text-xs text-muted-foreground">
          {studioVersion}
        </code>
      </SettingsRow>
      <SettingsRow label={t("settings.about.packageVersion")}>
        <code className="font-mono text-xs text-muted-foreground">
          {packageVersion}
        </code>
      </SettingsRow>
    </SettingsSection>
  );
}
