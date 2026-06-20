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
  // Resolves the same findings + fingerprint for gated/private repos.
  hfToken?: string | null;
  // Coarse fallback when the scan endpoint is unreachable.
  requiresTrustRemoteCode?: boolean;
  // Called on approval with the pinning fingerprint.
  onApprove: (fingerprint: string | null) => void;
}

/** Gate a load that may need trust_remote_code: scan, show the consent dialog, and on
 *  approval call onApprove with the pinning fingerprint. Returns false if declined. */
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

  // Open the dialog for custom-code consent OR flagged unsafe files. Otherwise a model
  // can still need trust_remote_code via its YAML default (no auto_map, e.g. GLM-4.7-Flash):
  // propagate the caller's requirement with an empty pin instead of sending false.
  if (!scan.requiresTrustRemoteCode && scan.unsafeFiles.length === 0) {
    if (requiresTrustRemoteCode) onApprove(null);
    return true;
  }

  const confirmed = await useRemoteCodeConsentDialogStore
    .getState()
    .requestConsent(scan);
  if (!confirmed) {
    // Declined: purge every repo our scan first downloaded (a LoRA scan pulls adapter +
    // base) so untrusted code is not left on disk. Fall back to the primary flag for an older backend.
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
