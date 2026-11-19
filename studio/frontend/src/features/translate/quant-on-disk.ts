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

/** True / false for the quant on disk, null when unknown. The local listing answers from disk alone;
 *  the remote one is a live Hub listing, asked only when the disk does not show the quant complete. */
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
