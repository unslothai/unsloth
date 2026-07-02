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

function parseStudioVersions(data: ApiObject): StudioVersions {
  const packageVersion = data["version"];
  const studioVersion = data["studio_version"];
  return {
    packageVersion:
      typeof packageVersion === "string" ? packageVersion : null,
    studioVersion: typeof studioVersion === "string" ? studioVersion : null,
  };
}

function hasAllVersions(versions: StudioVersions): boolean {
  return Boolean(versions.packageVersion && versions.studioVersion);
}

async function requestStudioVersions(
  token: string | null,
): Promise<StudioVersions> {
  const headers = new Headers();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(apiUrl("/api/health"), { headers });
  if (!res.ok) {
    return EMPTY_VERSIONS;
  }
  const data = (await res.json()) as ApiObject;
  return parseStudioVersions(data);
}

async function fetchStudioVersions(): Promise<StudioVersions> {
  try {
    const token = getAuthToken();
    const versions = await requestStudioVersions(token);
    if (!token || hasAllVersions(versions)) {
      return versions;
    }

    if (await refreshSession()) {
      return requestStudioVersions(getAuthToken());
    }
    return versions;
  } catch {
    return EMPTY_VERSIONS;
  }
}

// Shared "Unsloth" version block, shown in both General and About. The About
// tab passes llamaCppVersion to surface the installed llama.cpp build alongside
// the version rows; General omits it, so the row only shows on About.
export function StudioVersionSection({
  llamaCppVersion,
}: {
  llamaCppVersion?: string | null;
} = {}) {
  const t = useT();
  const [packageVersion, setPackageVersion] = useState("dev");
  const [studioVersion, setStudioVersion] = useState("dev");

  useEffect(() => {
    let canceled = false;
    fetchStudioVersions().then((next) => {
      if (canceled) return;
      if (next.packageVersion) setPackageVersion(next.packageVersion);
      if (next.studioVersion) setStudioVersion(next.studioVersion);
    });
    return () => {
      canceled = true;
    };
  }, []);

  return (
    <SettingsSection title="Unsloth">
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
      {llamaCppVersion ? (
        <SettingsRow label={t("settings.about.llamaCppVersion")}>
          <code className="font-mono text-xs text-muted-foreground">
            {llamaCppVersion}
          </code>
        </SettingsRow>
      ) : null}
    </SettingsSection>
  );
}
