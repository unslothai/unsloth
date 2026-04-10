// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Encrypted reveal flow for the Access Endpoint API key.
 *
 * Studio is often served over plain HTTP, which rules out WebCrypto
 * (`crypto.subtle` is HTTPS-only outside of localhost). We therefore use
 * node-forge for both RSA-OAEP (session key wrap) and AES-256-GCM (payload
 * decrypt).
 *
 * Flow:
 *   1. Fetch the server's RSA public key (PEM) — cached for the session.
 *   2. Generate a random 32-byte AES-256 session key in the browser.
 *   3. RSA-OAEP-SHA256 encrypt the session key with the server public key.
 *   4. POST the encrypted session key to /api/inference/access-endpoint/reveal.
 *   5. Server decrypts the session key, AES-GCM encrypts the API key and
 *      returns {iv, ciphertext}.
 *   6. Decrypt ciphertext locally using the session key we generated.
 *
 * The session key never leaves the browser in plaintext, and the API key
 * never leaves the server in plaintext.
 */

import forge from "node-forge";

import { authFetch } from "@/features/auth";

let cachedPublicKeyPem: string | null = null;
let cachedForgeKey: forge.pki.rsa.PublicKey | null = null;

export function clearAccessEndpointPublicKeyCache(): void {
  cachedPublicKeyPem = null;
  cachedForgeKey = null;
}

async function fetchAccessEndpointPublicKey(
  forceRefresh = false,
): Promise<forge.pki.rsa.PublicKey> {
  if (!forceRefresh && cachedForgeKey) {
    return cachedForgeKey;
  }
  const response = await authFetch("/api/inference/access-endpoint/public-key");
  if (!response.ok) {
    throw new Error(`Failed to fetch public key: ${response.status}`);
  }
  const body = (await response.json()) as { public_key?: string };
  const publicKeyPem = body.public_key?.trim();
  if (!publicKeyPem) {
    throw new Error("Public key missing from response");
  }
  if (!forceRefresh && cachedPublicKeyPem === publicKeyPem && cachedForgeKey) {
    return cachedForgeKey;
  }
  const forgeKey = forge.pki.publicKeyFromPem(publicKeyPem);
  cachedPublicKeyPem = publicKeyPem;
  cachedForgeKey = forgeKey;
  return forgeKey;
}

/**
 * Perform the encrypted reveal handshake and return the plaintext API key.
 * Returns `null` if the endpoint is disabled (HTTP 404 from reveal).
 */
export async function revealAccessEndpointKey(): Promise<string | null> {
  // Step 1: fetch server public key (cached per session)
  let publicKey: forge.pki.rsa.PublicKey;
  try {
    publicKey = await fetchAccessEndpointPublicKey();
  } catch {
    // Refresh cache once on failure (server may have restarted with new keypair)
    publicKey = await fetchAccessEndpointPublicKey(true);
  }

  // Step 2: generate AES-256 session key
  const sessionKeyBytes = forge.random.getBytesSync(32);

  // Step 3: RSA-OAEP-SHA256 wrap the session key
  const encryptedSessionKeyBytes = publicKey.encrypt(
    sessionKeyBytes,
    "RSA-OAEP",
    {
      md: forge.md.sha256.create(),
      mgf1: { md: forge.md.sha256.create() },
    },
  );
  const encryptedSessionKeyB64 = forge.util.encode64(encryptedSessionKeyBytes);

  // Step 4: POST to reveal endpoint
  const response = await authFetch(
    "/api/inference/access-endpoint/reveal",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ encrypted_session_key: encryptedSessionKeyB64 }),
    },
  );

  if (response.status === 404) {
    return null; // Endpoint not enabled
  }
  if (!response.ok) {
    // A restart can invalidate the cached public key — retry once with a
    // fresh key before surrendering.
    clearAccessEndpointPublicKeyCache();
    throw new Error(`Reveal failed: ${response.status}`);
  }

  const body = (await response.json()) as { iv?: string; ciphertext?: string };
  if (!body.iv || !body.ciphertext) {
    throw new Error("Reveal response missing iv/ciphertext");
  }

  // Step 5: AES-GCM decrypt
  const ivBytes = forge.util.decode64(body.iv);
  const ciphertextWithTag = forge.util.decode64(body.ciphertext);

  // forge's GCM API wants ciphertext and tag separated (tag is the last 16 bytes)
  const tagLength = 16;
  if (ciphertextWithTag.length < tagLength) {
    throw new Error("Ciphertext too short for GCM tag");
  }
  const ctBytes = ciphertextWithTag.slice(
    0,
    ciphertextWithTag.length - tagLength,
  );
  const tagBytes = ciphertextWithTag.slice(
    ciphertextWithTag.length - tagLength,
  );

  const decipher = forge.cipher.createDecipher("AES-GCM", sessionKeyBytes);
  decipher.start({
    iv: ivBytes,
    tagLength: tagLength * 8,
    tag: forge.util.createBuffer(tagBytes),
  });
  decipher.update(forge.util.createBuffer(ctBytes));
  const ok = decipher.finish();
  if (!ok) {
    throw new Error("AES-GCM authentication failed");
  }
  return decipher.output.toString();
}
