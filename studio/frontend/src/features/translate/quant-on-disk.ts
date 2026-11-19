// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type VariantRow = {
  quant?: string | null;
  downloaded?: boolean;
  partial?: boolean;
};

async function readQuant(
  quant: string,
  list: (localOnly: boolean) => Promise<readonly VariantRow[]>,
  localOnly: boolean,
): Promise<boolean | null> {
  try {
    const variant = (await list(localOnly)).find(
      (entry) => entry.quant?.toLowerCase() === quant,
    );
    return variant
      ? variant.downloaded === true && variant.partial !== true
      : null;
  } catch {
    return null;
  }
}

/** Quant on disk: true / false, null when unknown. Hub asked only if disk does not show it complete. */
export async function quantOnDisk(
  quant: string,
  list: (localOnly: boolean) => Promise<readonly VariantRow[]>,
  options: { remote: boolean; signal?: AbortSignal },
): Promise<boolean | null> {
  const local = await readQuant(quant, list, true);
  if (local === true || !options.remote || options.signal?.aborted)
    return local;
  return readQuant(quant, list, false);
}
