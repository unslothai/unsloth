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
      unsafeFiles: [],
      securityBlocked: false,
    };
  }

  // Open the dialog when the model needs custom-code consent OR Hugging Face's
  // security scan flagged unsafe files (a hard block). Otherwise nothing to confirm.
  if (!scan.requiresTrustRemoteCode && scan.unsafeFiles.length === 0) return true;

  const confirmed = await useRemoteCodeConsentDialogStore
    .getState()
    .requestConsent(scan);
  if (!confirmed) {
    // Declined: if our scan was the first to download this repo, purge it so the
    // untrusted custom code is not left on disk (the backend leaves models the
    // user already had, weighted repos, and local paths untouched).
    if (scan.createdByScan) void discardRemoteCodeDownload(scan.modelName);
    return false;
  }
  onApprove(scan.fingerprint);
  return true;
}
