// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  discardRemoteCodeDownload,
  getRemoteCodeScan,
} from "../api/remote-code-api";
import { useRemoteCodeConsentDialogStore } from "../stores/remote-code-consent-dialog-store";
import type { RemoteCodeScan } from "../types";

interface ConfirmArgs {
  modelName: string;
  hfToken?: string | null;
  preferLocalCache?: boolean;
  modelLocalPath?: string | null;
  modelSnapshotPath?: string | null;
  modelSnapshotRepoId?: string | null;
  requiresTrustRemoteCode?: boolean;
  onApprove: (fingerprint: string | null) => void;
}

/** Scan, ask for consent, and call onApprove with the pinning fingerprint. False if declined. */
export async function confirmRemoteCodeIfNeeded({
  modelName,
  hfToken,
  preferLocalCache,
  modelLocalPath,
  modelSnapshotPath,
  modelSnapshotRepoId,
  requiresTrustRemoteCode,
  onApprove,
}: ConfirmArgs): Promise<boolean> {
  let scan: RemoteCodeScan;
  try {
    scan = await getRemoteCodeScan(modelName, hfToken, {
      preferLocalCache,
      modelLocalPath,
      modelSnapshotPath,
      modelSnapshotRepoId,
    });
  } catch (error) {
    if (modelSnapshotPath) {
      throw error;
    }
    scan = {
      requiresTrustRemoteCode: Boolean(requiresTrustRemoteCode),
      approvable: true,
      maxSeverity: null,
      fingerprint: null,
      findings: [],
      findingsSummary: "",
      modelName,
      createdByScan: false,
      scanCreatedRepos: [],
      unsafeFiles: [],
      securityBlocked: false,
      alreadyApproved: false,
      provider: null,
    };
  }

  // Models needing remote code ship auto_map and hit the dialog, so the flag is only set via approval.
  if (!scan.requiresTrustRemoteCode && scan.unsafeFiles.length === 0) {
    return true;
  }

  if (
    scan.alreadyApproved &&
    scan.unsafeFiles.length === 0 &&
    !scan.securityBlocked
  ) {
    onApprove(scan.fingerprint);
    return true;
  }

  const confirmed = await useRemoteCodeConsentDialogStore
    .getState()
    .requestConsent(scan);
  if (!confirmed) {
    // Declined: purge every repo the scan first downloaded (a LoRA scan pulls adapter + base).
    const toPurge =
      scan.scanCreatedRepos.length > 0
        ? scan.scanCreatedRepos
        : scan.createdByScan
          ? [scan.modelName]
          : [];
    for (const repo of toPurge) void discardRemoteCodeDownload(repo);
    return false;
  }
  onApprove(scan.fingerprint);
  return true;
}
