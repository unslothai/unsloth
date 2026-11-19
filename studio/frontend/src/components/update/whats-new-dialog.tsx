// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT } from "@/i18n";
import { openExternalLink } from "@/lib/open-link";
import { cn } from "@/lib/utils";
import { SparklesIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { ReleaseNotesPanel } from "./release-notes-panel";
import { releases } from "./releases";
import { displayVersion, useUpdateOffer } from "./update-offer-store";
import { useWhatsNew } from "./whats-new-store";

const WASH =
  "bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)]";
const WASH_STRONG =
  "bg-[color-mix(in_oklab,var(--foreground)_calc(9%*var(--contrast-wash-gain,1)),transparent)]";

export function WhatsNewDialog() {
  const { open, offer, close } = useWhatsNew();
  if (!open) return null;
  return (
    <Dialog
      open={true}
      onOpenChange={(next) => {
        if (!next) close();
      }}
    >
      <DialogContent
        className="flex max-h-[min(42rem,calc(100dvh-var(--studio-window-chrome-top,0px)-2rem))] flex-col gap-0 overflow-hidden p-0 focus:outline-none sm:max-w-[52rem]"
        tabIndex={-1}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          (event.currentTarget as HTMLElement).focus();
        }}
        data-testid="whats-new-dialog"
      >
        <ReleaseHistory
          key={offer?.version ?? "history"}
          version={offer?.version}
          url={offer?.url}
        />
      </DialogContent>
    </Dialog>
  );
}

const sameVersion = (a: string, b: string) =>
  displayVersion(a) === displayVersion(b);

function ReleaseHistory({
  version,
  url,
}: {
  version?: string;
  url?: string | null;
}) {
  const t = useT();
  const close = useWhatsNew((state) => state.close);
  const liveOffer = useUpdateOffer((state) => state.offer);
  const target = version ?? liveOffer?.latestVersion;
  const bundled = target
    ? releases.find((r) => sameVersion(r.version, target))
    : undefined;
  const [selected, setSelected] = useState(
    bundled?.version ?? target ?? releases[0].version,
  );
  const entry = releases.find((r) => r.version === selected);
  const versions = releases.map((r) => r.version);
  if (target && !bundled) versions.unshift(target);
  const dateFormat = new Intl.DateTimeFormat(
    document.documentElement.lang || undefined,
    { dateStyle: "medium", timeZone: "UTC" },
  );
  const href = entry
    ? `https://github.com/unslothai/unsloth/releases/tag/v${entry.version}`
    : (url ?? liveOffer?.notesUrl ?? null);

  return (
    <>
      <header className="flex shrink-0 items-center gap-3.5 px-6 pb-4 pt-6 pr-16">
        <img
          src="/circle-logo-small.png"
          alt=""
          className="size-11 shrink-0 rounded-full"
        />
        <div className="min-w-0">
          <DialogTitle className="text-xl leading-tight">
            {t("updateNotes.studioTitle")}
          </DialogTitle>
          <DialogDescription className="mt-1 text-ui-13">
            {liveOffer
              ? t("updateNotes.offerLine", {
                  latest: displayVersion(liveOffer.latestVersion),
                  current: displayVersion(liveOffer.currentVersion),
                })
              : t("updateNotes.history")}
          </DialogDescription>
        </div>
      </header>
      <div className="grid min-h-0 flex-1 gap-x-2 px-3 max-sm:grid-rows-[auto_minmax(0,1fr)] sm:grid-cols-[11.5rem_minmax(0,1fr)]">
        <nav
          aria-label={t("updateNotes.versions")}
          className="min-w-0 px-1 pb-3 pt-1 sm:overflow-y-auto"
        >
          <p className="mb-1.5 px-2.5 text-xs font-medium text-muted-foreground">
            {t("updateNotes.versions")}
          </p>
          <div className="flex gap-1 overflow-x-auto sm:flex-col">
            {versions.map((v) => {
              const release = releases.find((item) => item.version === v);
              const isOffered =
                liveOffer != null && sameVersion(v, liveOffer.latestVersion);
              const isInstalled =
                liveOffer != null && sameVersion(v, liveOffer.currentVersion);
              return (
                <button
                  key={v}
                  type="button"
                  aria-current={selected === v ? "true" : undefined}
                  onClick={() => setSelected(v)}
                  className={cn(
                    "flex shrink-0 cursor-pointer items-center gap-3 rounded-md px-2.5 py-2 text-left transition-colors",
                    "hover:bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
                    selected === v &&
                      "bg-[color-mix(in_oklab,var(--foreground)_calc(9%*var(--contrast-wash-gain,1)),transparent)] hover:bg-[color-mix(in_oklab,var(--foreground)_calc(9%*var(--contrast-wash-gain,1)),transparent)]",
                  )}
                >
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-semibold tabular-nums">
                      {displayVersion(v)}
                    </span>
                    {release && (
                      <time
                        dateTime={release.date}
                        className="block text-xs text-muted-foreground"
                      >
                        {dateFormat.format(new Date(release.date))}
                      </time>
                    )}
                  </span>
                  {isOffered ? (
                    <span className="flex items-center gap-1 text-ui-11 font-medium text-primary">
                      <span className="size-1.5 rounded-full bg-primary" />
                      {t("updateNotes.newLabel")}
                    </span>
                  ) : isInstalled ? (
                    <span className="text-ui-11 text-muted-foreground">
                      {t("updateNotes.installedLabel")}
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
        </nav>
        <article
          key={selected}
          className="hover-scrollbar min-h-0 min-w-0 overflow-y-auto px-3 pb-6 pt-1"
        >
          <h2 className="font-heading text-lg font-semibold leading-tight">
            {entry?.title ??
              t("updateNotes.notesForVersion", {
                version: displayVersion(selected),
              })}
          </h2>
          {entry ? (
            <>
              <time
                dateTime={entry.date}
                className="mt-1 block text-xs text-muted-foreground"
              >
                {dateFormat.format(new Date(entry.date))}
              </time>
              <div
                className={cn(
                  "mt-4 grid gap-3",
                  entry.features.length > 1 && "sm:grid-cols-2",
                )}
              >
                {entry.features.map((feature) => (
                  <figure
                    key={feature.title}
                    className={cn("overflow-hidden rounded-lg", WASH)}
                  >
                    <img
                      src={feature.image}
                      onError={(event) => {
                        event.currentTarget.style.display = "none";
                      }}
                      alt={feature.alt}
                      className={cn(
                        "block w-full object-cover",
                        entry.features.length > 1 ? "h-[7.5rem]" : "h-52",
                      )}
                    />
                    <figcaption className="p-3">
                      <h3 className="font-heading text-sm font-semibold">
                        {feature.title}
                      </h3>
                      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                        {feature.summary}
                      </p>
                    </figcaption>
                  </figure>
                ))}
              </div>
              <section className="mt-6">
                <h3 className="flex items-center gap-2.5 text-sm font-semibold">
                  <span
                    className={cn(
                      "flex size-7 items-center justify-center rounded-sm text-primary",
                      WASH_STRONG,
                    )}
                  >
                    <HugeiconsIcon
                      icon={SparklesIcon}
                      className="size-4"
                      strokeWidth={1.75}
                      aria-hidden="true"
                    />
                  </span>
                  {t("updateNotes.alsoInRelease")}
                </h3>
                <dl className="mt-3 grid grid-cols-[minmax(6.5rem,auto)_1fr] gap-x-5 gap-y-2.5 pl-[2.375rem] text-ui-13 leading-relaxed">
                  {entry.changes.map(([label, description]) => (
                    <div key={label} className="contents">
                      <dt className="font-medium text-muted-foreground">
                        {label}
                      </dt>
                      <dd>{description}</dd>
                    </div>
                  ))}
                </dl>
              </section>
            </>
          ) : (
            <ReleaseNotesPanel
              version={selected}
              open={true}
              releaseNotesUrl={url ?? liveOffer?.notesUrl ?? null}
            />
          )}
        </article>
      </div>
      <footer
        className={cn(
          "flex shrink-0 items-center justify-between gap-3 px-6 py-4",
          WASH,
        )}
      >
        {href ? (
          <a
            href={href}
            target="_blank"
            rel="noreferrer"
            onClick={(e) => {
              e.preventDefault();
              openExternalLink(href);
            }}
            className="text-xs font-medium text-muted-foreground hover:text-foreground hover:underline"
          >
            {t("updateNotes.openRelease")} ↗
          </a>
        ) : (
          <span />
        )}
        {liveOffer && (
          <Button
            size="sm"
            variant="primary"
            className="h-auto rounded-full px-4 py-2 text-ui-13"
            disabled={liveOffer.actionDisabled}
            onClick={() => {
              if (liveOffer.source === "desktop") close();
              liveOffer.runAction();
            }}
            data-testid="whats-new-update"
          >
            {liveOffer.actionLabel}
          </Button>
        )}
      </footer>
    </>
  );
}
