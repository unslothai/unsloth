// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type HubEndpointCheckResult = {
  api: "ok" | "official" | "failed";
  download: "mirror" | "official" | "failed";
  xet: "none" | "mirror" | "official" | "failed";
};

export type HubEndpointCheckVerdict =
  | "ok"
  | "redirected"
  | "downloadsOfficial"
  | "xetOfficial"
  | "failed";

/** Empty or the official Hub: there is no mirror to check, and the backend refuses it. */
export function isOfficialHubEndpoint(raw: string): boolean {
  const value = raw.trim();
  if (value === "") return true;
  let host: string;
  try {
    host = new URL(value.includes("://") ? value : `https://${value}`).hostname;
  } catch {
    return false;
  }
  host = host.toLowerCase().replace(/\.$/, "");
  return ["huggingface.co", "hf.co"].some(
    (domain) => host === domain || host.endsWith(`.${domain}`),
  );
}

/** The one thing the owner most needs to know: a whole-site redirect outranks a failed probe,
 * which outranks downloads or Xet chunks leaving the mirror. */
export function hubEndpointCheckVerdict(
  check: HubEndpointCheckResult,
): HubEndpointCheckVerdict {
  if (check.api === "official") return "redirected";
  if (check.api === "failed" || check.download === "failed") return "failed";
  if (check.download === "official") return "downloadsOfficial";
  if (check.xet === "official") return "xetOfficial";
  if (check.xet === "failed") return "failed";
  return "ok";
}
