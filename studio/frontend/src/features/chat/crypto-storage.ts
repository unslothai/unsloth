// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Password-derived AES-256-GCM encryption for API keys at rest in localStorage.
 *
 * Architecture:
 *   login password → PBKDF2(password, salt) → AES-256-GCM key (in-memory only)
 *   plaintext API key → AES-GCM encrypt → base64(salt ‖ iv ‖ ciphertext) → localStorage
 *
 * The derived key is never persisted — it lives in a module-scoped variable,
 * set on login and cleared on logout.
 */

const PBKDF2_ITERATIONS = 100_000;
const SALT_BYTES = 16;
const IV_BYTES = 12;

// ── Session password (in-memory only) ────────────────────────────

let _sessionPassword: string | null = null;

/** Store the login password in memory for the duration of the session. */
export function setSessionPassword(password: string): void {
  _sessionPassword = password;
}

/** Retrieve the in-memory session password. Returns null when logged out. */
export function getSessionPassword(): string | null {
  return _sessionPassword;
}

/** Clear the in-memory session password (called on logout). */
export function clearSessionPassword(): void {
  _sessionPassword = null;
}

// ── Key derivation ───────────────────────────────────────────────

/** Derive an AES-256-GCM CryptoKey from a password and salt via PBKDF2. */
export async function deriveKey(
  password: string,
  salt: Uint8Array,
): Promise<CryptoKey> {
  const keyMaterial = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(password),
    "PBKDF2",
    false,
    ["deriveKey"],
  );
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations: PBKDF2_ITERATIONS, hash: "SHA-256" },
    keyMaterial,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
}

// ── Encrypt / Decrypt ────────────────────────────────────────────

/**
 * Encrypt a plaintext string.
 * Returns a base64 string containing: salt (16 B) ‖ iv (12 B) ‖ ciphertext.
 */
export async function encryptValue(
  plaintext: string,
  password: string,
): Promise<string> {
  const salt = crypto.getRandomValues(new Uint8Array(SALT_BYTES));
  const iv = crypto.getRandomValues(new Uint8Array(IV_BYTES));
  const key = await deriveKey(password, salt);
  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    key,
    new TextEncoder().encode(plaintext),
  );
  // Concatenate salt + iv + ciphertext into one ArrayBuffer
  const combined = new Uint8Array(
    SALT_BYTES + IV_BYTES + ciphertext.byteLength,
  );
  combined.set(salt, 0);
  combined.set(iv, SALT_BYTES);
  combined.set(new Uint8Array(ciphertext), SALT_BYTES + IV_BYTES);
  return btoa(String.fromCharCode(...combined));
}

/**
 * Decrypt a base64 string produced by {@link encryptValue}.
 * Throws if the password is wrong or the data is tampered/not encrypted.
 */
export async function decryptValue(
  encrypted: string,
  password: string,
): Promise<string> {
  const combined = Uint8Array.from(atob(encrypted), (c) => c.charCodeAt(0));
  if (combined.byteLength < SALT_BYTES + IV_BYTES + 1) {
    throw new Error("Invalid encrypted value: too short");
  }
  const salt = combined.slice(0, SALT_BYTES);
  const iv = combined.slice(SALT_BYTES, SALT_BYTES + IV_BYTES);
  const ciphertext = combined.slice(SALT_BYTES + IV_BYTES);
  const key = await deriveKey(password, salt);
  const plainBuffer = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv },
    key,
    ciphertext,
  );
  return new TextDecoder().decode(plainBuffer);
}
