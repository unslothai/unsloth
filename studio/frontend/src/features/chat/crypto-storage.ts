// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import forge from "node-forge";

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

// ── Session password ─────────────────────────────────────────────
//
// Held in a module variable for fast access, backed by sessionStorage
// so it survives page refreshes within the same tab.  Cleared on
// logout and when the tab is closed (sessionStorage semantics).

const SESSION_PW_KEY = "unsloth_chat_session_pw";

let _sessionPassword: string | null = null;

/** Store the login password for the duration of the browser session. */
export function setSessionPassword(password: string): void {
  _sessionPassword = password;
  try {
    sessionStorage.setItem(SESSION_PW_KEY, password);
  } catch {
    // Private browsing or quota — in-memory only
  }
}

/** Retrieve the session password. Restores from sessionStorage after a page refresh. */
export function getSessionPassword(): string | null {
  if (_sessionPassword) return _sessionPassword;
  try {
    const stored = sessionStorage.getItem(SESSION_PW_KEY);
    if (stored) {
      _sessionPassword = stored;
      return stored;
    }
  } catch {
    // ignore
  }
  return null;
}

/** Clear the session password (called on logout). */
export function clearSessionPassword(): void {
  _sessionPassword = null;
  try {
    sessionStorage.removeItem(SESSION_PW_KEY);
  } catch {
    // ignore
  }
}

// ── Key derivation ───────────────────────────────────────────────

/** Derive a 256-bit AES key from a password and salt via PBKDF2-SHA256. */
export function deriveKeyBytes(
  password: string,
  salt: Uint8Array,
): string {
  const saltStr = forge.util.binary.raw.encode(salt);
  return forge.pkcs5.pbkdf2(password, saltStr, PBKDF2_ITERATIONS, 32, forge.md.sha256.create());
}

// ── Encrypt / Decrypt ────────────────────────────────────────────

/**
 * Encrypt a plaintext string.
 * Returns a base64 string containing: salt (16 B) ‖ iv (12 B) ‖ ciphertext ‖ tag (16 B).
 */
export async function encryptValue(
  plaintext: string,
  password: string,
): Promise<string> {
  const salt = forge.random.getBytesSync(SALT_BYTES);
  const iv = forge.random.getBytesSync(IV_BYTES);
  const keyBytes = deriveKeyBytes(password, Uint8Array.from(salt, (c: string) => c.charCodeAt(0)));
  const cipher = forge.cipher.createCipher("AES-GCM", keyBytes);
  cipher.start({ iv, tagLength: 128 });
  cipher.update(forge.util.createBuffer(forge.util.encodeUtf8(plaintext)));
  cipher.finish();
  const ciphertext = cipher.output.getBytes();
  const tag = cipher.mode.tag.getBytes();
  // Concatenate salt + iv + ciphertext + tag
  const combined = salt + iv + ciphertext + tag;
  return forge.util.encode64(combined);
}

/**
 * Decrypt a base64 string produced by {@link encryptValue}.
 * Throws if the password is wrong or the data is tampered/not encrypted.
 */
export async function decryptValue(
  encrypted: string,
  password: string,
): Promise<string> {
  const combined = forge.util.decode64(encrypted);
  if (combined.length < SALT_BYTES + IV_BYTES + 1 + 16) {
    throw new Error("Invalid encrypted value: too short");
  }
  const salt = combined.slice(0, SALT_BYTES);
  const iv = combined.slice(SALT_BYTES, SALT_BYTES + IV_BYTES);
  const ciphertextAndTag = combined.slice(SALT_BYTES + IV_BYTES);
  const ciphertext = ciphertextAndTag.slice(0, ciphertextAndTag.length - 16);
  const tag = ciphertextAndTag.slice(ciphertextAndTag.length - 16);
  const saltBytes = Uint8Array.from(salt, (c: string) => c.charCodeAt(0));
  const keyBytes = deriveKeyBytes(password, saltBytes);
  const decipher = forge.cipher.createDecipher("AES-GCM", keyBytes);
  decipher.start({ iv, tag: forge.util.createBuffer(tag) });
  decipher.update(forge.util.createBuffer(ciphertext));
  const ok = decipher.finish();
  if (!ok) {
    throw new Error("Decryption failed: wrong password or tampered data");
  }
  return forge.util.decodeUtf8(decipher.output.getBytes());
}
