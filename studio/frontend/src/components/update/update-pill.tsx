// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  isTrainingRunActive,
  useTrainingRuntimeStore,
} from "@/features/training/stores/training-runtime-store";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Download01Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { sameVersion } from "./release-meta";
import { releases } from "./releases";
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
  const training = useTrainingRuntimeStore(isTrainingRunActive);
  if (!offer || covered) return null;
  const release = releases.find((r) =>
    sameVersion(r.version, offer.latestVersion),
  );

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
            className="size-[calc(15px*var(--ui-space-scale,1))] text-primary"
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
              className="size-[calc(18px*var(--ui-space-scale,1))] text-foreground"
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
        {release && <Highlights release={release} />}
        {training && (
          <p className="mt-3 flex items-start gap-2 rounded-md bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)] px-2.5 py-2 text-xs leading-relaxed">
            <HugeiconsIcon
              icon={Alert02Icon}
              className="mt-px size-3.5 shrink-0 text-amber-500"
              strokeWidth={2}
              aria-hidden="true"
            />
            {t("updateNotes.stopsTraining")}
          </p>
        )}
        <div className="mt-4 flex items-center gap-1">
          <Button
            size="sm"
            variant="primary"
            className="h-auto gap-1.5 rounded-full px-3 py-2 text-ui-13 font-medium"
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
            <HugeiconsIcon
              icon={SparklesIcon}
              className="size-3.5"
              strokeWidth={2}
              aria-hidden="true"
            />
            {t("updateNotes.whatsNew")}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            className="h-auto rounded-full px-3.5 py-2 text-ui-13 font-medium"
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
            className="ml-auto h-auto rounded-full px-3 py-2 text-ui-13 font-medium text-muted-foreground"
            onClick={() => {
              setOpen(false);
              offer.later();
            }}
          >
            {t("desktopUpdate.banner.later")}
          </Button>
        </div>
        {/* A disabled button takes no hover or focus, so its reason is shown as text. */}
        {offer.actionDisabled && offer.note && (
          <p
            className="mt-3 text-xs leading-relaxed text-muted-foreground"
            data-testid="update-popover-note"
          >
            {offer.note}
          </p>
        )}
      </PopoverContent>
    </Popover>
  );
}

/** The offered release's first highlights, so the notes are read before updating. */
function Highlights({ release }: { release: (typeof releases)[number] }) {
  const t = useT();
  const items = [
    ...release.features.map((f) => [f.title, f.summary] as const),
    ...(["new", "improved", "fixed"] as const).flatMap(
      (key) => release.changes[key] ?? [],
    ),
  ].slice(0, 3);
  return (
    <div className="mt-4">
      <p className="text-ui-11 font-medium uppercase tracking-wide text-muted-foreground">
        {t("updateNotes.inThisUpdate")}
      </p>
      <ul className="mt-1.5 flex list-disc flex-col gap-1 pl-4 text-xs leading-relaxed marker:text-muted-foreground">
        {items.map(([label, text]) => (
          <li key={label}>
            <strong className="font-semibold">{label}.</strong> {text}
          </li>
        ))}
      </ul>
    </div>
  );
}
