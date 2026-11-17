// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Cursor02Icon } from "@hugeicons/core-free-icons";
import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { useChatRuntimeStore } from "@/features/chat";
import { parseExternalModelId } from "@/features/chat/external-providers";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { useT } from "@/i18n";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect } from "react";
import { computerUseReady, refreshComputerUseStatus, useComputerUseStore } from "./store";

/** Why the loaded model can't use the tool, or null. The server decides for connected models. */
function useModelProblem(): "tools" | "vision" | null {
  return useChatRuntimeStore((s) => {
    if (!s.params.checkpoint || s.modelLoading) return null;
    if (!s.supportsTools) return "tools";
    if (parseExternalModelId(s.params.checkpoint)) return null;
    return s.loadedIsMultimodal ? null : "vision";
  });
}

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
  // Unavailable offers nothing; switched off sends the owner to the switch.
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
      <HugeiconsIcon icon={Cursor02Icon} strokeWidth={2} />
      {t("computerUse.label")}
      {on ? <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="ml-auto" /> : null}
    </DropdownMenuItem>
  );
}

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
        <HugeiconsIcon icon={Cursor02Icon} className="size-[calc(17px*var(--ui-space-scale,1))]" strokeWidth={2} />,
      )}
      <span>{t("computerUse.pill")}</span>
    </button>
  );
}
