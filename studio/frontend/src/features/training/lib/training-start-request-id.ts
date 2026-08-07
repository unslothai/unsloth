


interface RandomUuidCrypto {
  getRandomValues?: <T extends ArrayBufferView>(array: T) => T;
  randomUUID?: () => string;
}

function fillPseudoRandomBytes(bytes: Uint8Array): Uint8Array {
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = Math.floor(Math.random() * 256);
  }
  return bytes;
}

function createRandomBytes(cryptoSource: RandomUuidCrypto | null): Uint8Array {
  const bytes = new Uint8Array(16);
  if (typeof cryptoSource?.getRandomValues !== "function") {
    return fillPseudoRandomBytes(bytes);
  }
  try {
    return cryptoSource.getRandomValues(bytes);
  } catch {
    return fillPseudoRandomBytes(bytes);
  }
}

function formatUuid(bytes: Uint8Array): string {
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
}

export function createTrainingStartRequestId(
  cryptoSource: RandomUuidCrypto | null = globalThis.crypto ?? null,
): string {
  if (typeof cryptoSource?.randomUUID === "function") {
    try {
      return cryptoSource.randomUUID();
    } catch {
      return formatUuid(createRandomBytes(cryptoSource));
    }
  }
  return formatUuid(createRandomBytes(cryptoSource));
}
