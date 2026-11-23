// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  type SettingsTab,
  useSettingsDialogStore,
} from "@/features/settings/stores/settings-dialog-store";
import { useScrollFades } from "@/hooks/use-scroll-fades";
import { useLocale, useT } from "@/i18n";
import { openExternalLink } from "@/lib/open-link";
import { cn } from "@/lib/utils";
import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useRouter } from "@tanstack/react-router";
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

const RAIL_BG =
  "bg-[color-mix(in_oklab,var(--foreground)_calc(3%*var(--contrast-wash-gain,1)),transparent)]";
const HAIRLINE =
  "border-[color-mix(in_oklab,var(--foreground)_calc(9%*var(--contrast-wash-gain,1)),transparent)]";

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
        className="flex h-[min(41rem,calc(100dvh-var(--studio-window-chrome-top,0px)-2rem))] flex-col gap-0 overflow-hidden rounded-2xl p-0 focus:outline-none sm:max-w-[60rem]"
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
    <div className="flex min-h-0 flex-1 max-sm:flex-col">
      <aside className={cn("flex min-h-0 shrink-0 flex-col sm:w-64", RAIL_BG)}>
        <header className="flex items-center gap-3 px-6 pb-5 pt-7 max-sm:pr-14">
          <img
            src="/circle-logo-small.png"
            alt=""
            className="size-7 shrink-0 rounded-full"
          />
          <div className="min-w-0">
            <DialogTitle className="truncate text-sm font-semibold leading-tight">
              {t("updateNotes.studioTitle")}
            </DialogTitle>
            <DialogDescription className="mt-0.5 truncate text-xs">
              {liveOffer
                ? t("updateNotes.offerLine", {
                    latest: displayVersion(liveOffer.latestVersion),
                    current: displayVersion(liveOffer.currentVersion),
                  })
                : t("updateNotes.history")}
            </DialogDescription>
          </div>
        </header>
        <nav
          ref={railFades.attach}
          onScroll={railFades.onScroll}
          aria-label={t("updateNotes.versions")}
          className={cn(
            "hover-scrollbar panel-scroll-fade-action min-h-0 min-w-0 flex-1 px-3 pb-4 sm:overflow-y-auto",
            railFades.className,
          )}
        >
          <div className="flex gap-0.5 overflow-x-auto max-sm:pb-1 sm:flex-col">
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
                      <span className="size-1.5 shrink-0 rounded-full bg-primary">
                        <span className="sr-only">
                          {t("updateNotes.newLabel")}
                        </span>
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
        <UpdateAction />
      </aside>
      <article
        key={selected}
        className="hover-scrollbar min-h-0 min-w-0 flex-1 overflow-y-auto px-6 pb-12 pt-8 sm:px-12 sm:pt-12"
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
  );
}

function UpdateAction() {
  const close = useWhatsNew((state) => state.close);
  const liveOffer = useUpdateOffer((state) => state.offer);
  if (!liveOffer) {
    return null;
  }
  return (
    <div className="shrink-0 px-4 pb-5 pt-2 max-sm:hidden">
      <Button
        size="sm"
        variant="primary"
        className="h-auto w-full rounded-lg px-4 py-2.5 text-ui-13"
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
    </div>
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
        "flex shrink-0 cursor-pointer items-center gap-3 rounded-md px-3 py-2 text-left transition-colors",
        "hover:bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
        selected &&
          "bg-[color-mix(in_oklab,var(--foreground)_calc(9%*var(--contrast-wash-gain,1)),transparent)] hover:bg-[color-mix(in_oklab,var(--foreground)_calc(9%*var(--contrast-wash-gain,1)),transparent)]",
      )}
    >
      <span className="min-w-0 flex-1">
        <span
          className={cn(
            "block text-ui-13 tabular-nums",
            selected ? "font-semibold" : "font-medium",
          )}
        >
          {title}
        </span>
        {subtitle && (
          <span className="mt-px block text-ui-11 text-muted-foreground">
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
      className="inline-flex items-center gap-0.5 text-muted-foreground underline-offset-4 transition-colors hover:text-foreground hover:underline"
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

/** The docs page a card or change row points to, when it has one. */
function docsLink(item: object): string | undefined {
  return "link" in item && typeof item.link === "string"
    ? item.link
    : undefined;
}

/** Where "Try it" goes: a route, or a Settings tab as `settings:<tab>`. */
function openTarget(item: object): string | undefined {
  return "open" in item && typeof item.open === "string"
    ? item.open
    : undefined;
}

const SETTINGS_PREFIX = "settings:";

function CardActions({ feature }: { feature: object }) {
  const t = useT();
  const router = useRouter();
  const close = useWhatsNew((state) => state.close);
  const guide = docsLink(feature);
  const target = openTarget(feature);
  if (!(guide || target)) {
    return null;
  }
  const tryIt = () => {
    if (!target) {
      return;
    }
    close();
    if (target.startsWith(SETTINGS_PREFIX)) {
      useSettingsDialogStore
        .getState()
        .openDialog(target.slice(SETTINGS_PREFIX.length) as SettingsTab);
    } else {
      router.history.push(target);
    }
  };
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {target && (
        <Button
          size="sm"
          variant="secondary"
          className="h-8 rounded-md px-3 text-ui-13"
          onClick={tryIt}
        >
          {target.startsWith(SETTINGS_PREFIX)
            ? t("updateNotes.openSettings")
            : t("updateNotes.tryIt")}
        </Button>
      )}
      {guide && (
        <Button
          size="sm"
          variant="ghost"
          className={cn(
            "h-8 gap-1 rounded-md px-3 text-ui-13 text-muted-foreground hover:text-foreground",
            !target && "-ml-3",
          )}
          onClick={() => openExternalLink(guide)}
        >
          {t("updateNotes.readGuide")}
          <HugeiconsIcon
            icon={ArrowUpRight01Icon}
            className="size-3.5"
            strokeWidth={2}
            aria-hidden="true"
          />
        </Button>
      )}
    </div>
  );
}

function GuideLink({ href }: { href: string | undefined }) {
  const t = useT();
  if (!href) {
    return null;
  }
  return (
    <>
      {" "}
      <ExternalLink href={href}>{t("updateNotes.guide")}</ExternalLink>
    </>
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

function SectionLabel({ children }: { children: string }) {
  return (
    <h3 className="text-ui-11 font-medium uppercase tracking-[0.08em] text-muted-foreground">
      {children}
    </h3>
  );
}

function ReleaseBody({ entry }: { entry: Release }) {
  const t = useT();
  return (
    <>
      <div className="mt-9 grid gap-x-6 gap-y-8 sm:grid-cols-2">
        {entry.features.map((feature) => (
          <figure key={feature.title}>
            <img
              src={feature.image}
              onError={(event) => {
                event.currentTarget.style.display = "none";
              }}
              alt={feature.alt}
              className={cn(
                "block aspect-[2/1] w-full rounded-xl object-cover",
              )}
            />
            <figcaption className="mt-3.5">
              <h3 className="text-sm font-semibold">{feature.title}</h3>
              <p className="mt-1 text-ui-13 leading-relaxed text-muted-foreground">
                {feature.summary}
              </p>
              <CardActions feature={feature} />
            </figcaption>
          </figure>
        ))}
      </div>
      <section className="mt-12">
        <SectionLabel>{t("updateNotes.alsoInRelease")}</SectionLabel>
        <dl className={cn("mt-3 border-t", HAIRLINE)}>
          {entry.changes.map(([label, description, link]) => (
            <div
              key={label}
              className={cn(
                "grid gap-1 border-b py-3.5 text-ui-13 leading-relaxed sm:grid-cols-[9rem_1fr] sm:gap-6",
                HAIRLINE,
              )}
            >
              <dt className="text-muted-foreground">{label}</dt>
              <dd>
                {description}
                {link && (
                  <>
                    {" "}
                    <ExternalLink href={link}>
                      {t("updateNotes.guide")}
                    </ExternalLink>
                  </>
                )}
              </dd>
            </div>
          ))}
        </dl>
        {entry.more ? (
          <p className="mt-4 text-ui-13">
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
      <h2 className="font-heading text-2xl font-semibold tracking-tight">
        {t("updateNotes.sinceTitle", { version: displayVersion(current) })}
      </h2>
      <p className="mt-2 text-ui-13 text-muted-foreground">
        {t("updateNotes.releaseCount", { count: list.length })}
      </p>
      <div className="mt-8">
        {list.map((entry) => (
          <section
            key={entry.version}
            className={cn("flex gap-8 border-t py-8", HAIRLINE)}
          >
            <div className="min-w-0 flex-1">
              <p className="text-ui-12 text-muted-foreground">
                <span className="font-medium text-foreground tabular-nums">
                  {displayVersion(entry.version)}
                </span>
                <span aria-hidden="true"> · </span>
                {releaseAge(entry.date, locale)}
              </p>
              <h3 className="mt-1 font-heading text-base font-semibold">
                {entry.title}
              </h3>
              <ul className="mt-3 flex flex-col gap-2 text-ui-13 leading-relaxed">
                {entry.features.map((feature) => (
                  <li key={feature.title}>
                    <span className="font-medium">{feature.title}.</span>{" "}
                    <span className="text-muted-foreground">
                      {feature.summary}
                    </span>
                    <GuideLink href={docsLink(feature)} />
                  </li>
                ))}
                {entry.changes.map(([label, description, link]) => (
                  <li key={label}>
                    <span className="font-medium">{label}.</span>{" "}
                    <span className="text-muted-foreground">{description}</span>
                    <GuideLink href={link} />
                  </li>
                ))}
              </ul>
              {entry.more ? (
                <p className="mt-3 text-ui-13">
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
                className={cn(
                  "hidden aspect-[2/1] w-44 shrink-0 self-start rounded-lg object-cover sm:block",
                )}
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
  return (
    <>
      {from ? (
        <CatchUp current={from} list={caughtUp} locale={locale} />
      ) : (
        <>
          <h2 className="font-heading text-2xl font-semibold tracking-tight">
            {entry?.title ??
              t("updateNotes.notesForVersion", {
                version: displayVersion(selected),
              })}
          </h2>
          {entry ? (
            <>
              <p className="mt-2 flex flex-wrap items-center gap-x-2 text-ui-13 text-muted-foreground">
                <time dateTime={entry.date}>
                  {releaseAge(entry.date, locale)}
                </time>
                <span aria-hidden="true">·</span>
                {releaseLink}
              </p>
              <ReleaseBody entry={entry} />
            </>
          ) : (
            <>
              {releaseLink && (
                <p className="mt-2 text-ui-13 text-muted-foreground">
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
