// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import {
  type UpdatePillSlot,
  displayVersion,
  usePillHost,
  useUpdateOffer,
} from "./update-offer-store";
import { useWhatsNew } from "./whats-new-store";

/** "Update available" pill for the top-right of the shell; opens a compact offer. */
export function UpdatePill({
  slot,
  className,
}: {
  slot: UpdatePillSlot;
  className?: string;
}) {
  const t = useT();
  usePillHost(slot);
  const offer = useUpdateOffer((s) => s.offer);
  // The titlebar pill is on every route, so the header one would only repeat it.
  const covered = useUpdateOffer(
    (s) => slot === "header" && s.hosts.titlebar > 0,
  );
  const [open, setOpen] = useState(false);
  if (!offer || covered) return null;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild={true}>
        <button
          type="button"
          data-testid="update-pill"
          className={cn(
            "inline-flex h-[calc(30px*var(--ui-space-scale,1))] cursor-pointer items-center gap-1.5 rounded-full pl-2.5 pr-3 text-ui-13 font-medium text-foreground",
            "bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] transition-colors",
            "hover:bg-[color-mix(in_oklab,var(--foreground)_calc(11%*var(--contrast-wash-gain,1)),transparent)] data-[state=open]:bg-[color-mix(in_oklab,var(--foreground)_calc(11%*var(--contrast-wash-gain,1)),transparent)]",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            className,
          )}
        >
          <HugeiconsIcon
            icon={Download01Icon}
            className="size-[15px] text-primary"
            strokeWidth={2}
            aria-hidden="true"
          />
          {t("desktopUpdate.banner.update")}
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        sideOffset={8}
        className="w-[min(21.5rem,calc(100vw-32px))] gap-0 rounded-xl p-4"
        data-testid="update-popover"
      >
        <div className="flex items-start gap-3">
          <span className="flex size-9 shrink-0 items-center justify-center rounded-md bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)]">
            <HugeiconsIcon
              icon={Download01Icon}
              className="size-[18px] text-foreground"
              strokeWidth={1.75}
              aria-hidden="true"
            />
          </span>
          <div className="min-w-0 flex-1 pt-px">
            <p className="font-heading text-[15px] font-semibold leading-tight">
              {t("updateNotes.updateAvailable")}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {t("updateNotes.versionLine", {
                latest: displayVersion(offer.latestVersion),
                current: displayVersion(offer.currentVersion),
              })}
            </p>
          </div>
        </div>
        <div className="mt-4 flex items-center gap-1">
          <Button
            size="sm"
            variant="primary"
            className="h-auto rounded-full px-3.5 py-2 text-ui-13"
            disabled={offer.actionDisabled}
            title={offer.note ?? undefined}
            aria-description={offer.note ?? undefined}
            onClick={() => {
              offer.runAction();
              if (offer.source === "desktop") setOpen(false);
            }}
            data-testid="update-popover-action"
          >
            {offer.actionLabel}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-auto rounded-full px-3 py-2 text-ui-13 font-medium text-foreground"
            aria-haspopup="dialog"
            onClick={() => {
              setOpen(false);
              useWhatsNew.getState().show({
                version: offer.latestVersion,
                url: offer.notesUrl,
              });
            }}
            data-testid="update-popover-whats-new"
          >
            {t("updateNotes.whatsNew")}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="ml-auto h-auto rounded-full px-3 py-2 text-ui-13 font-medium text-muted-foreground"
            onClick={() => {
              setOpen(false);
              offer.later();
            }}
          >
            {t("desktopUpdate.banner.later")}
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
