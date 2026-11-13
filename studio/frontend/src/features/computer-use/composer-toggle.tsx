// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { useChatRuntimeStore } from "@/features/chat";
import { parseExternalModelId } from "@/features/chat/external-providers";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { useT } from "@/i18n";
import { Tick02Icon } from "@/lib/tick-icon";
import { ComputerIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect } from "react";
import { computerUseReady, refreshComputerUseStatus, useComputerUseStore } from "./store";

/** Why the loaded model can't use the tool, or null: it needs Unsloth's tool loop and eyes. A
 *  connected model's vision isn't known here, so the server decides for those. */
function useModelProblem(): "tools" | "vision" | null {
  return useChatRuntimeStore((s) => {
    if (!s.params.checkpoint || s.modelLoading) return null;
    if (!s.supportsTools) return "tools";
    if (parseExternalModelId(s.params.checkpoint)) return null;
    return s.loadedIsMultimodal ? null : "vision";
  });
}

/** Asks the server once the menu or the pill is on screen; the answer is shared. */
function useComputerUseStatus() {
  const status = useComputerUseStore((state) => state.status);
  useEffect(() => {
    void refreshComputerUseStatus();
  }, []);
  return status;
}

export function ComputerUseMenuItem() {
  const t = useT();
  const status = useComputerUseStatus();
  const enabled = useComputerUseStore((state) => state.enabled);
  const problem = useModelProblem();
  // A server that can't control this computer has nothing to offer; one where it is only
  // switched off sends the owner to the switch.
  if (!status?.available) return null;
  const ready = computerUseReady(status);
  const on = enabled && ready && !problem;
  return (
    <DropdownMenuItem
      disabled={ready && !!problem}
      className={on ? "text-primary font-medium" : undefined}
      onSelect={() => {
        if (!ready) {
          useSettingsDialogStore.getState().openDialog("computer-use");
          return;
        }
        useComputerUseStore.getState().setEnabled(!enabled);
      }}
    >
      <HugeiconsIcon icon={ComputerIcon} strokeWidth={2} />
      {ready ? t("computerUse.label") : t("computerUse.setUp")}
      {on ? <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="ml-auto" /> : null}
    </DropdownMenuItem>
  );
}

/** Shown while computer use is on; a click turns it off. `glyph` wraps the icon like the other pills. */
export function ComputerUsePill({ glyph }: { glyph: (icon: ReactNode) => ReactNode }) {
  const t = useT();
  const status = useComputerUseStatus();
  const enabled = useComputerUseStore((state) => state.enabled);
  const problem = useModelProblem();
  if (!enabled || !computerUseReady(status)) return null;
  return (
    <button
      type="button"
      onClick={() => useComputerUseStore.getState().setEnabled(false)}
      className="composer-pill-btn"
      data-pill-label={t("computerUse.pill")}
      data-active={problem ? "false" : "true"}
      aria-label={t("computerUse.turnOff")}
      title={problem ? t(`computerUse.needs.${problem}`) : undefined}
    >
      {glyph(
        <HugeiconsIcon icon={ComputerIcon} className="size-[calc(17px*var(--ui-space-scale,1))]" strokeWidth={2} />,
      )}
      <span>{t("computerUse.pill")}</span>
    </button>
  );
}
