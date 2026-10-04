// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function collectParts(value: unknown): Record<string, unknown>[] {
  const native = record(value);
  if (!native) return [];
  if (Array.isArray(native.parts)) {
    return native.parts.filter(
      (part): part is Record<string, unknown> => record(part) !== undefined,
    );
  }
  const signature =
    typeof native.thoughtSignature === "string"
      ? native.thoughtSignature
      : native.thought_signature;
  return ["executableCode", "codeExecutionResult", "inlineData"].flatMap(
    (key) => {
      const part = record(native[key]);
      if (!part) return [];
      return [
        {
          [key]: part,
          ...(key === "executableCode" &&
          typeof signature === "string" &&
          signature
            ? { thoughtSignature: signature }
            : {}),
        },
      ];
    },
  );
}

export function mergeGoogleNativeParts(
  args: Record<string, unknown>,
  endGoogle: unknown,
): Record<string, unknown> {
  const endNative = record(record(endGoogle)?.native_part);
  if (!endNative) return args;
  const google = record(args.google) ?? {};
  return {
    ...args,
    google: {
      ...google,
      native_part: {
        parts: [
          ...collectParts(google.native_part),
          ...collectParts(endNative),
        ],
      },
    },
  };
}
