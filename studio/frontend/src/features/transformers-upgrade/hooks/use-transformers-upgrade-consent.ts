


import { useTransformersUpgradeDialogStore } from "../stores/transformers-upgrade-dialog-store";
import type { TransformersUpgradeInfo } from "../types";

interface ConfirmArgs {
  modelName: string;
  /** validate's transformers_upgrade payload; null/undefined skips the dialog. */
  upgrade: TransformersUpgradeInfo | null | undefined;
  /** When no release is installable, offer continuing into the caller's custom-code gate. */
  trustRemoteCodeFallback?: boolean;
  /** The caller already confirmed the swap's "stop N chats" prompt: carry it into
   *  the install, which otherwise 409s on those same chats with no way forward. */
  forceCancelActive?: boolean;
}

/** Pause a load needing a newer transformers on the consent dialog and run the install.
 *  Resolves true when the load can continue; false on cancel or not-installable with no fallback. */
export async function confirmTransformersUpgradeIfNeeded({
  modelName,
  upgrade,
  trustRemoteCodeFallback,
  forceCancelActive,
}: ConfirmArgs): Promise<boolean> {
  if (!upgrade) return true;
  return useTransformersUpgradeDialogStore
    .getState()
    .requestConsent(modelName, upgrade, {
      trustRemoteCodeFallback: Boolean(trustRemoteCodeFallback),
      forceCancelActive: Boolean(forceCancelActive),
    });
}
