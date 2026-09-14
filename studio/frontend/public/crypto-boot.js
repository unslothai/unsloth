// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// randomUUID is unavailable on insecure HTTP origins. Install it before the
// module graph, which may call it during evaluation.
if (globalThis.crypto && typeof globalThis.crypto.randomUUID !== "function") {
  const cryptoRef = globalThis.crypto;
  const randomByte = () =>
    typeof cryptoRef.getRandomValues === "function"
      ? cryptoRef.getRandomValues(new Uint8Array(1))[0]
      : Math.floor(Math.random() * 256);

  cryptoRef.randomUUID = () =>
    "10000000-1000-4000-8000-100000000000".replace(/[018]/g, (c) =>
      (+c ^ (randomByte() & (15 >> (+c / 4)))).toString(16),
    );
}
