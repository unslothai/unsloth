// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { useScrollFades } from "@/hooks/use-scroll-fades";
import { useLocale, useT } from "@/i18n";
import { openExternalLink } from "@/lib/open-link";
import { cn } from "@/lib/utils";
import { ArrowUpRight01Icon, SparklesIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useState } from "react";
import {
  type Release,
  releaseAge,
  releaseUrl,
  releasesBetween,
  sameVersion,
} from "./release-meta";
import { ReleaseNotesPanel } from "./release-notes-panel";
import { releases } from "./releases";
import { displayVersion, useUpdateOffer } from "./update-offer-store";
import { rememberUpdateTarget } from "./updated-notice";
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
          since={offer?.since}
        />
      </DialogContent>
    </Dialog>
  );
}

// The rail entry that gathers every release between the installed and offered versions.
const SINCE = "since";

/** What the rail lists and whether it leads with the catch-up entry. */
function planHistory(target: string | undefined, from: string | undefined) {
  const bundled = target
    ? releases.find((r) => sameVersion(r.version, target))
    : undefined;
  const caughtUp = from && target ? releasesBetween(from, target) : [];
  const showSince =
    caughtUp.length > 1 &&
    target !== undefined &&
    sameVersion(target, caughtUp[0].version);
  const versions = releases.map((r) => r.version);
  if (target && !bundled) {
    versions.unshift(target);
  }
  return { target, bundled, from, caughtUp, showSince, versions };
}

function ReleaseHistory({
  version,
  url,
  since,
}: {
  version?: string;
  url?: string | null;
  since?: string;
}) {
  const t = useT();
  const liveOffer = useUpdateOffer((state) => state.offer);
  const railFades = useScrollFades();
  const { target, bundled, from, caughtUp, showSince, versions } = planHistory(
    version ?? liveOffer?.latestVersion,
    since ?? liveOffer?.currentVersion,
  );
  const [selected, setSelected] = useState(
    showSince ? SINCE : (bundled?.version ?? target ?? releases[0].version),
  );
  const entry = releases.find((r) => r.version === selected);
  const dateFormat = new Intl.DateTimeFormat(
    document.documentElement.lang || undefined,
    { dateStyle: "medium", timeZone: "UTC" },
  );
  let href: string | null = url ?? liveOffer?.notesUrl ?? null;
  if (entry) {
    href = releaseUrl(entry.version);
  } else if (selected === SINCE) {
    href = null;
  }
  const releaseLink = href ? (
    <ExternalLink href={href}>{t("updateNotes.openRelease")}</ExternalLink>
  ) : null;

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
        <UpdateAction />
      </header>
      <div className="grid min-h-0 flex-1 gap-x-2 px-3 max-sm:grid-rows-[auto_minmax(0,1fr)] sm:grid-cols-[11.5rem_minmax(0,1fr)]">
        <nav
          ref={railFades.attach}
          onScroll={railFades.onScroll}
          aria-label={t("updateNotes.versions")}
          className={cn(
            "hover-scrollbar panel-scroll-fade-action min-w-0 px-1 pb-3 pt-1 sm:overflow-y-auto",
            railFades.className,
          )}
        >
          <p className="mb-1.5 px-2.5 text-xs font-medium text-muted-foreground">
            {t("updateNotes.versions")}
          </p>
          <div className="flex gap-1 overflow-x-auto sm:flex-col">
            {showSince && from && (
              <RailItem
                selected={selected === SINCE}
                onSelect={() => setSelected(SINCE)}
                title={t("updateNotes.sinceNav", {
                  version: displayVersion(from),
                })}
                subtitle={t("updateNotes.releaseCount", {
                  count: caughtUp.length,
                })}
              />
            )}
            {versions.map((v) => {
              const release = releases.find((item) => item.version === v);
              const isOffered =
                liveOffer != null && sameVersion(v, liveOffer.latestVersion);
              const isInstalled =
                liveOffer != null && sameVersion(v, liveOffer.currentVersion);
              return (
                <RailItem
                  key={v}
                  selected={selected === v}
                  onSelect={() => setSelected(v)}
                  title={displayVersion(v)}
                  subtitle={
                    release ? dateFormat.format(new Date(release.date)) : null
                  }
                  badge={
                    isOffered ? (
                      <span className="flex items-center gap-1 text-ui-11 font-medium text-primary">
                        <span className="size-1.5 rounded-full bg-primary" />
                        {t("updateNotes.newLabel")}
                      </span>
                    ) : isInstalled ? (
                      <span className="text-ui-11 text-muted-foreground">
                        {t("updateNotes.installedLabel")}
                      </span>
                    ) : null
                  }
                />
              );
            })}
          </div>
        </nav>
        <article
          key={selected}
          className="hover-scrollbar min-h-0 min-w-0 overflow-y-auto px-7 pb-6 pt-1"
        >
          <ReleaseArticle
            selected={selected}
            entry={entry}
            from={selected === SINCE ? from : undefined}
            caughtUp={caughtUp}
            releaseLink={releaseLink}
            fallbackUrl={url ?? liveOffer?.notesUrl ?? null}
          />
        </article>
      </div>
    </>
  );
}

function UpdateAction() {
  const close = useWhatsNew((state) => state.close);
  const liveOffer = useUpdateOffer((state) => state.offer);
  if (!liveOffer) {
    return null;
  }
  return (
    <Button
      size="sm"
      variant="primary"
      className="ml-auto h-auto shrink-0 rounded-full px-4 py-2 text-ui-13"
      disabled={liveOffer.actionDisabled}
      onClick={() => {
        if (liveOffer.source === "desktop") {
          close();
        }
        rememberUpdateTarget(liveOffer.latestVersion);
        liveOffer.runAction();
      }}
      data-testid="whats-new-update"
    >
      {liveOffer.actionLabel}
    </Button>
  );
}

function RailItem({
  selected,
  onSelect,
  title,
  subtitle,
  badge,
}: {
  selected: boolean;
  onSelect: () => void;
  title: string;
  subtitle: string | null;
  badge?: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-current={selected ? "true" : undefined}
      onClick={onSelect}
      className={cn(
        "flex shrink-0 cursor-pointer items-center gap-3 rounded-md px-2.5 py-2 text-left transition-colors",
        "hover:bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
        selected &&
          "bg-[color-mix(in_oklab,var(--foreground)_calc(9%*var(--contrast-wash-gain,1)),transparent)] hover:bg-[color-mix(in_oklab,var(--foreground)_calc(9%*var(--contrast-wash-gain,1)),transparent)]",
      )}
    >
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-semibold tabular-nums">
          {title}
        </span>
        {subtitle && (
          <span className="block text-xs text-muted-foreground">
            {subtitle}
          </span>
        )}
      </span>
      {badge}
    </button>
  );
}

function ExternalLink({ href, children }: { href: string; children: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      onClick={(e) => {
        e.preventDefault();
        openExternalLink(href);
      }}
      className="inline-flex items-center gap-0.5 font-medium text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
    >
      {children}
      <HugeiconsIcon
        icon={ArrowUpRight01Icon}
        className="size-3.5"
        strokeWidth={2}
        aria-hidden="true"
      />
    </a>
  );
}

function MoreChanges({ entry }: { entry: Release }) {
  const t = useT();
  if (!entry.more) {
    return null;
  }
  return (
    <ExternalLink href={releaseUrl(entry.version)}>
      {entry.more === 1
        ? t("updateNotes.moreChangesOne")
        : t("updateNotes.moreChanges", { count: entry.more })}
    </ExternalLink>
  );
}

function ReleaseBody({ entry }: { entry: Release }) {
  const t = useT();
  return (
    <>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
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
              className="block aspect-[12/5] w-full object-cover"
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
              <dt className="font-medium text-muted-foreground">{label}</dt>
              <dd>{description}</dd>
            </div>
          ))}
        </dl>
        {entry.more ? (
          <p className="mt-3 pl-[2.375rem] text-ui-13">
            <MoreChanges entry={entry} />
          </p>
        ) : null}
      </section>
    </>
  );
}

/** Every release since the installed one, as one compact list: the catch-up view. */
function CatchUp({
  current,
  list,
  locale,
}: {
  current: string;
  list: Release[];
  locale: ReturnType<typeof useLocale>;
}) {
  const t = useT();
  return (
    <>
      <h2 className="font-heading text-lg font-semibold leading-tight">
        {t("updateNotes.sinceTitle", { version: displayVersion(current) })}
      </h2>
      <p className="mt-1 text-xs text-muted-foreground">
        {t("updateNotes.releaseCount", { count: list.length })}
      </p>
      <div className="mt-5 flex flex-col gap-6">
        {list.map((entry) => (
          <section key={entry.version} className="flex gap-4">
            <div className="min-w-0 flex-1">
              <p className="text-xs text-muted-foreground">
                <span className="font-semibold text-foreground tabular-nums">
                  {displayVersion(entry.version)}
                </span>{" "}
                · {releaseAge(entry.date, locale)}
              </p>
              <h3 className="mt-0.5 font-heading text-sm font-semibold">
                {entry.title}
              </h3>
              <ul className="mt-2 flex list-disc flex-col gap-1.5 pl-4 text-ui-13 leading-relaxed marker:text-muted-foreground">
                {entry.features.map((feature) => (
                  <li key={feature.title}>
                    <strong className="font-semibold">{feature.title}.</strong>{" "}
                    {feature.summary}
                  </li>
                ))}
                {entry.changes.map(([label, description]) => (
                  <li key={label}>
                    <strong className="font-semibold">{label}.</strong>{" "}
                    {description}
                  </li>
                ))}
              </ul>
              {entry.more ? (
                <p className="mt-2 text-xs">
                  <MoreChanges entry={entry} />
                </p>
              ) : null}
            </div>
            {entry.features[0] && (
              <img
                src={entry.features[0].image}
                alt=""
                onError={(event) => {
                  event.currentTarget.style.display = "none";
                }}
                className="hidden aspect-[12/5] w-36 shrink-0 self-start rounded-md object-cover sm:block"
              />
            )}
          </section>
        ))}
      </div>
    </>
  );
}

function ReleaseArticle({
  selected,
  entry,
  from,
  caughtUp,
  releaseLink,
  fallbackUrl,
}: {
  selected: string;
  entry: Release | undefined;
  from: string | undefined;
  caughtUp: Release[];
  releaseLink: ReactNode;
  fallbackUrl: string | null;
}) {
  const t = useT();
  const locale = useLocale();
  const dateFormat = new Intl.DateTimeFormat(
    document.documentElement.lang || undefined,
    { dateStyle: "medium", timeZone: "UTC" },
  );
  return (
    <>
      {from ? (
        <CatchUp current={from} list={caughtUp} locale={locale} />
      ) : (
        <>
          <h2 className="font-heading text-lg font-semibold leading-tight">
            {entry?.title ??
              t("updateNotes.notesForVersion", {
                version: displayVersion(selected),
              })}
          </h2>
          {entry ? (
            <>
              <p className="mt-1 flex flex-wrap items-center gap-x-2 text-xs text-muted-foreground">
                <time dateTime={entry.date}>
                  {dateFormat.format(new Date(entry.date))}
                </time>
                <span aria-hidden="true">·</span>
                <span>{releaseAge(entry.date, locale)}</span>
                <span aria-hidden="true">·</span>
                {releaseLink}
              </p>
              <ReleaseBody entry={entry} />
            </>
          ) : (
            <>
              {releaseLink && (
                <p className="mt-1 text-xs text-muted-foreground">
                  {releaseLink}
                </p>
              )}
              <ReleaseNotesPanel
                version={selected}
                open={true}
                releaseNotesUrl={fallbackUrl}
              />
            </>
          )}
        </>
      )}
    </>
  );
}
