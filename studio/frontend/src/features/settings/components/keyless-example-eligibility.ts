// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether a base URL is one the keyless usage example may be printed for.
 *
 * Mirror of `keyless_api_access.keyless_authority_address_allowed`: loopback always,
 * private LAN under `inference`, nothing else. If the panel says yes where admission says
 * no, it renders a copy-paste `Bearer not-needed` command that answers 401.
 *
 * Its own module, free of the component's react/shiki graph, so a unit test can pin the
 * table instead of scraping the source.
 */

import type {
  KeylessApiAccessExposure,
  KeylessApiAccessScope,
} from "../api/keyless-api-access.ts";
import { isLoopbackHost, normalizeHost } from "./agent-command.ts";

// Same networks as the backend's _PRIVATE_LAN_NETWORKS.
export function isPrivateLanHost(hostname: string): boolean {
  const host = normalizeHost(hostname).toLowerCase();
  // No IPv4-mapped unwrapping: admission refuses that form precisely because the browser
  // will not unwrap it, so a helper that does answers the wrong question. That is how the
  // panel came to advertise a mapped literal.
  const ipv4 = host.split(".").map(Number);
  if (
    ipv4.length === 4 &&
    ipv4.every((part) => Number.isInteger(part) && part >= 0 && part <= 255)
  ) {
    return (
      ipv4[0] === 10 ||
      (ipv4[0] === 172 && ipv4[1] >= 16 && ipv4[1] <= 31) ||
      (ipv4[0] === 192 && ipv4[1] === 168) ||
      (ipv4[0] === 169 && ipv4[1] === 254)
    );
  }
  return /^f[cd][0-9a-f]*:/i.test(host) || /^fe[89ab][0-9a-f]*:/i.test(host);
}

function isIpLiteralHost(host: string): boolean {
  const ipv4 = host.split(".");
  if (
    ipv4.length === 4 &&
    ipv4.every((part) => /^\d{1,3}$/.test(part) && Number(part) <= 255)
  ) {
    return true;
  }
  // normalizeHost has already stripped the URL brackets from an IPv6 authority
  return host.includes(":") && /^[0-9a-f:.]+$/i.test(host);
}

// 0.0.0.0 and every all-zero IPv6 spelling.
function isUnspecifiedHost(host: string): boolean {
  return host === "0.0.0.0" || (host.includes(":") && /^[0:]+$/.test(host));
}

// The serializer emits "::ffff:7f00:1"; the expanded form is matched too.
function isIpv4MappedHost(host: string): boolean {
  return host.startsWith("::ffff:") || /^(?:0{1,4}:){5}ffff:/i.test(host);
}

/**
 * Syntax is not the question: `[::ffff:192.168.1.24]`, `[::]` and `8.8.8.8` are all well
 * formed literals admission refuses. Checking only that a base LOOKS like an address is what
 * left the panel advertising `Bearer not-needed` for each of them.
 */
export function isKeylessAllowedAuthority(hostname: string): boolean {
  const host = normalizeHost(hostname).toLowerCase();
  if (!isIpLiteralHost(host)) return false;
  if (isUnspecifiedHost(host) || isIpv4MappedHost(host)) return false;
  return isLoopbackHost(host) || isPrivateLanHost(host);
}

export function keylessBaseEligible(
  base: string,
  scope: KeylessApiAccessScope,
  exposure: KeylessApiAccessExposure | null,
): boolean {
  if (scope === "off" || exposure === "colab" || exposure === "public_url") {
    return false;
  }
  try {
    const host = normalizeHost(new URL(base).hostname);
    if (isLoopbackHost(host)) return true;
    if (!isKeylessAllowedAuthority(host)) return false;
    // `exposure` is computed from the RESOLVED address, so it must never widen a base the
    // authority rule rejected; on its own it said nothing about how the caller spelled it.
    return scope === "inference";
  } catch {
    return false;
  }
}
