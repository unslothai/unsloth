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
  // Sent to the scan endpoint so gated/private repos resolve the same findings
  // and fingerprint the worker will later require.
  hfToken?: string | null;
  // Coarse fallback when the scan endpoint is unreachable (gated/offline).
  requiresTrustRemoteCode?: boolean;
  // Called on approval with the fingerprint that pins this code version.
  onApprove: (fingerprint: string | null) => void;
}

/**
 * Gate a load that may need trust_remote_code. Scans the model's custom code,
 * shows the consent dialog with findings, and on approval calls `onApprove` with
 * the pinning fingerprint. Returns true if the load may proceed (no custom code,
 * or the user approved), false if the user declined.
 */
export async function confirmRemoteCodeIfNeeded({
  modelName,
  hfToken,
  requiresTrustRemoteCode,
  onApprove,
}: ConfirmArgs): Promise<boolean> {
  let scan: RemoteCodeScan;
  try {
    scan = await getRemoteCodeScan(modelName, hfToken);
  } catch {
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
    };
  }

  // Open the dialog when the model needs custom-code consent OR Hugging Face's
  // security scan flagged unsafe files (a hard block). Otherwise there is nothing to
  // review -- but a model can still require trust_remote_code via its Studio YAML
  // default with no auto_map (e.g. GLM-4.7-Flash). The scan reports no raw auto_map
  // for those, so propagate the caller's requirement (granting an empty pin) instead
  // of dropping it, which would send trust_remote_code=false and fail the load.
  if (!scan.requiresTrustRemoteCode && scan.unsafeFiles.length === 0) {
    if (requiresTrustRemoteCode) onApprove(null);
    return true;
  }

  const confirmed = await useRemoteCodeConsentDialogStore
    .getState()
    .requestConsent(scan);
  if (!confirmed) {
    // Declined: purge every repo our scan was the first to download (a LoRA scan
    // pulls both the adapter and its base) so the untrusted custom code is not left
    // on disk. The backend leaves models the user already had, weighted repos, and
    // local paths untouched. Fall back to the primary flag for an older backend.
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
