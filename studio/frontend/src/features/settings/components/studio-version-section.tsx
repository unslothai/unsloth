// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthToken } from "@/features/auth";
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

let versionsPromise: Promise<StudioVersions> | null = null;

async function requestStudioVersions(): Promise<StudioVersions> {
  const token = getAuthToken();
  const headers = new Headers();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(apiUrl("/api/health"), { headers });
  if (!res.ok) {
    throw new Error(`Failed to fetch Studio versions (${res.status})`);
  }
  const data = (await res.json()) as ApiObject;
  const packageVersion = data["version"];
  const studioVersion = data["studio_version"];
  return {
    packageVersion: typeof packageVersion === "string" ? packageVersion : null,
    studioVersion: typeof studioVersion === "string" ? studioVersion : null,
  };
}

async function fetchStudioVersions(): Promise<StudioVersions> {
  if (versionsPromise) {
    return versionsPromise;
  }

  versionsPromise = requestStudioVersions().catch(() => {
    versionsPromise = null;
    return EMPTY_VERSIONS;
  });

  return versionsPromise;
}

export function StudioVersionSection() {
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
  );
}
