import forge from "node-forge";

import { getPlatformAuthConfig } from "./config";

export class PlatformAuthConfigurationError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "PlatformAuthConfigurationError";
  }
}

export function encryptPlatformPassword(
  password: string,
  publicKeyPem = getPlatformAuthConfig().publicKeyPem,
): string {
  if (!publicKeyPem.trim()) {
    throw new PlatformAuthConfigurationError(
      "Rag Platform giriş anahtarı yapılandırılmamış.",
    );
  }
  try {
    const key = forge.pki.publicKeyFromPem(publicKeyPem);
    const utf8 = forge.util.encodeUtf8(password);
    const base64Password = forge.util.encode64(utf8);
    const encrypted = key.encrypt(base64Password, "RSAES-PKCS1-V1_5");
    return forge.util.encode64(encrypted);
  } catch (cause) {
    throw new PlatformAuthConfigurationError(
      "Rag Platform giriş anahtarı geçersiz veya backend anahtarıyla uyumsuz.",
      { cause },
    );
  }
}
