import forge from "node-forge";
import { describe, expect, it } from "vitest";

import {
  PlatformAuthConfigurationError,
  encryptPlatformPassword,
} from "../auth-crypto";

describe("platform password encryption", () => {
  it("matches the backend base64 + RSA PKCS#1 v1.5 wire contract", () => {
    const pair = forge.pki.rsa.generateKeyPair({ bits: 1024, e: 0x10001 });
    const encrypted = encryptPlatformPassword(
      "şifre-1234",
      forge.pki.publicKeyToPem(pair.publicKey),
    );

    const decryptedBase64 = pair.privateKey.decrypt(
      forge.util.decode64(encrypted),
      "RSAES-PKCS1-V1_5",
    );
    expect(forge.util.decodeUtf8(forge.util.decode64(decryptedBase64))).toBe(
      "şifre-1234",
    );
  });

  it("fails closed when the deployment public key is missing or malformed", () => {
    expect(() => encryptPlatformPassword("password", "")).toThrow(
      PlatformAuthConfigurationError,
    );
    expect(() => encryptPlatformPassword("password", "not-a-pem")).toThrow(
      PlatformAuthConfigurationError,
    );
  });
});
