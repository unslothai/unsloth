// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { LlamaUpdateChangelogPanel } from "@/components/update/llama-update-changelog-panel";
import { resyncInferenceStatusAfterServerModelChange } from "@/features/chat";
import {
  llamaUpdateOffered,
  useLlamaUpdateCheck,
} from "@/hooks/use-llama-update-check";
import {
  type NotificationChannel,
  markNotificationShown,
  useDocumentVisible,
  useNotificationDue,
  useNotificationFrequency,
} from "@/hooks/use-notification-frequency";
import {
  type UpdateComponent,
  type UpdateComponentFlags,
  heldUpdateBannerPref,
  llamaReleaseChanged,
  llamaUpdateToastMessage,
  updateBannerAllowed,
  updateBannerComponents,
  updateComponentKey,
  updateComponentsLabel,
  updateToastTag,
} from "@/lib/llama-job-lifecycle";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Download } from "lucide-react";
import {
  type CSSProperties,
  type ReactElement,
  useEffect,
  useRef,
  useState,
} from "react";
// Creep toward this cap between coarse backend progress updates.
const RUNNING_CAP = 0.95;

// The banner is not translated, so these mirror features/settings/lib/llama-backend-
// labels.ts. An unknown backend prints its own identifier rather than nothing.
const BACKEND_LABELS: Record<string, string> = {
  auto: "Automatic",
  cpu: "CPU",
  cuda: "CUDA",
  rocm: "ROCm",
  vulkan: "Vulkan",
  metal: "Metal",
};

function backendLabel(backend: string | null | undefined): string {
  if (!backend) {
    return "";
  }
  return BACKEND_LABELS[backend] ?? backend;
}

// Smooth coarse backend progress without freezing between milestones.
function useSmoothedProgress(
  active: boolean,
  progress: number | null,
  done: boolean,
): number {
  const [display, setDisplay] = useState(0);
  const displayRef = useRef(0);
  const progressRef = useRef<number | null>(progress);
  const doneRef = useRef(done);
  progressRef.current = progress;
  doneRef.current = done;

  useEffect(() => {
    if (!active) {
      displayRef.current = 0;
      setDisplay(0);
      return;
    }
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      // Guard against a first rAF timestamp before the captured start time.
      const dt = Math.max(0, Math.min((now - last) / 1000, 0.1));
      last = now;
      const current = displayRef.current;
      const real = progressRef.current ?? 0;
      let target: number;
      let speed: number; // approach rate (fraction of remaining gap per second)
      if (doneRef.current) {
        target = 1;
        speed = 5;
      } else if (real > current) {
        target = real; // catch up to a freshly observed milestone
        speed = 4;
      } else {
        target = RUNNING_CAP; // no signal: creep toward the cap, never frozen
        speed = 0.3;
      }
      const cap = doneRef.current ? 1 : RUNNING_CAP;
      const next = Math.min(
        current + (target - current) * Math.min(speed * dt, 1),
        cap,
      );
      displayRef.current = next;
      setDisplay(next);
      if (doneRef.current && next > 0.999) {
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active]);

  return display;
}

interface LlamaUpdateBannerProps {
  enabled?: boolean;
  // false fills a shared stack; true self-anchors.
  positioned?: boolean;
}

function sizeLabelOf(sizeBytes: number | null | undefined): string | null {
  return sizeBytes && sizeBytes > 0
    ? `${Math.round(sizeBytes / (1024 * 1024))} MB`
    : null;
}

interface UpdateRow {
  component: UpdateComponent;
  installedTag: string | null;
  latestTag: string | null;
  sizeBytes: number | null;
  versionChanged: boolean;
}

function VersionLine({
  row,
  bundled,
  backendPair,
}: {
  row: UpdateRow;
  bundled: boolean;
  backendPair: readonly [string, string] | null;
}): ReactElement {
  const [from, to] = backendPair ?? [
    row.installedTag ?? "unknown",
    row.latestTag ?? "",
  ];
  const rowSize = bundled ? sizeLabelOf(row.sizeBytes) : null;
  const sizeSuffix = rowSize ? ` · ${rowSize}` : "";
  return (
    <p
      // One line each, so the changelog floor can count them.
      className={cn(
        "mt-0.5 text-xs text-muted-foreground",
        bundled && "truncate",
      )}
      title={
        bundled
          ? `${row.component} ${from} \u2192 ${to}${sizeSuffix}`
          : undefined
      }
      data-testid="llama-update-version-line"
    >
      {bundled ? (
        <span className="text-foreground">{row.component} </span>
      ) : null}
      {from} &rarr; <span className="font-medium text-foreground">{to}</span>
      {sizeSuffix}
    </p>
  );
}

/** Bottom-right llama.cpp, whisper.cpp and audio.cpp update toast. */
export function LlamaUpdateBanner({
  enabled = true,
  positioned = true,
}: LlamaUpdateBannerProps): ReactElement | null {
  // The open card stays allowed after it records its components as shown.
  const [shownChannels, setShownChannels] = useState<
    readonly NotificationChannel[]
  >([]);
  const due: UpdateComponentFlags = {
    llama: useNotificationDue("llama"),
    whisper: useNotificationDue("whisper"),
    audio: useNotificationDue("audio"),
  };
  const muted: UpdateComponentFlags = {
    llama: useNotificationFrequency("llama") === "off",
    whisper: useNotificationFrequency("whisper") === "off",
    audio: useNotificationFrequency("audio") === "off",
  };
  const [changelogVersion, setChangelogVersion] = useState<string | null>(null);
  // Not gated on showBannerPref: this hook instance is the app-wide listener
  // for a cross-tab reload_required resync (the settings-sheet's own instance
  // only runs during an MTP-fallback rebuild), so muting the banner must not
  // also silence that resync -- it only suppresses the UI below.
  const { status, visible, applying, apply, dismiss, snooze } =
    useLlamaUpdateCheck({
      enabled,
      onReloadRequired: resyncInferenceStatusAfterServerModelChange,
    });

  const migrationPending = Boolean(status?.backend_migration_available);
  const listed = updateBannerComponents(
    {
      llama: Boolean(status?.llama.update_available) || migrationPending,
      whisper: Boolean(status?.whisper?.update_available),
      audio: Boolean(status?.audio?.update_available),
    },
    muted,
  );
  const listedKey = listed.join(" ");
  // Each component's own release pair and download size, not the top-level ones: those
  // are llama's whatever the card lists.
  const liveRows: UpdateRow[] = listed.map((component) => {
    const offer = status?.[updateComponentKey(component)];
    const installedTag = offer?.installed_tag ?? null;
    const latestTag = offer?.latest_tag ?? null;
    return {
      component,
      installedTag,
      latestTag,
      sizeBytes: offer?.update_size_bytes ?? null,
      versionChanged: llamaReleaseChanged(
        Boolean(offer?.update_available),
        installedTag,
        latestTag,
      ),
    };
  });
  // An error counts as in flight so a failed phase keeps its retry on screen.
  const jobState = status?.job.state;
  const inFlight = applying || jobState === "error";
  // A chained apply clears each component from the status as its phase lands, so the
  // card keeps the rows it started the job with until the job is over.
  const [jobRows, setJobRows] = useState<UpdateRow[] | null>(null);
  if (inFlight && jobRows === null && liveRows.length > 0) {
    setJobRows(liveRows);
  } else if (!inFlight && jobRows !== null) {
    setJobRows(null);
  }
  const rows = jobRows ?? liveRows;
  const label = updateComponentsLabel(rows.map((row) => row.component));
  const bundled = rows.length > 1;
  const llamaRow = rows.find((row) => row.component === "llama.cpp") ?? null;
  const primary = rows[0] ?? null;

  async function handleUpdate() {
    // Read before applying (the status refreshes mid-job). An audio.cpp update may keep
    // the old runtime, and a chained job reports every phase, so only the job's message
    // says what happened, as for a migration.
    const fromJob =
      migrationPending || bundled || primary?.component === "audio.cpp";
    const offerTag = primary?.latestTag ?? null;
    const result = await apply();
    if (result?.ok) {
      const updatedTag = bundled
        ? null
        : (updateToastTag(
            primary?.component ?? "llama.cpp",
            result.tag,
            offerTag,
          ) ?? "the latest build");
      toast.success(
        llamaUpdateToastMessage({
          component: label,
          migrating: fromJob,
          jobMessage: result.message,
          updatedTag,
          reloadRequired: result.reloadRequired,
        }),
      );
    } else if (result) {
      // The failing phase need not be the component the card names.
      toast.error(`Update failed: ${result.error ?? "unknown error"}`);
    }
  }

  const livePref = updateBannerAllowed(listed, due, shownChannels.length > 0);
  // Held across a chained apply, whose status drops each component as it lands.
  const [heldPref, setHeldPref] = useState<boolean | null>(null);
  useEffect(() => {
    setHeldPref((prev) => heldUpdateBannerPref(prev, inFlight, livePref));
  }, [inFlight, livePref]);
  const showBannerPref = heldPref ?? livePref;
  const show =
    showBannerPref &&
    visible &&
    status != null &&
    (llamaUpdateOffered(status) || applying);
  const tabVisible = useDocumentVisible();
  useEffect(() => {
    // Every listed component counts as seen, so none of them returns on a card of its
    // own within its quiet period. Muted components are never listed, so never recorded.
    if (show && tabVisible && livePref) {
      const fresh = (listedKey.split(" ") as UpdateComponent[])
        .filter(Boolean)
        .map(updateComponentKey)
        .filter((channel) => !shownChannels.includes(channel));
      if (fresh.length > 0) {
        for (const channel of fresh) {
          markNotificationShown(channel);
        }
        setShownChannels([...shownChannels, ...fresh]);
      }
    } else if (!show && shownChannels.length > 0) {
      setShownChannels([]);
    }
  }, [show, tabVisible, livePref, listedKey, shownChannels]);
  // A migration re-applies the install's own automatic choice, so it can be offered at a
  // release the machine already has, where the backend pair replaces the version line.
  const backendChange =
    llamaRow && status?.backend_migration_available && status.to_backend
      ? `${backendLabel(status.from_backend)} \u2192 ${backendLabel(status.to_backend)}`
      : null;
  const llamaVersionChanged = Boolean(llamaRow?.versionChanged);
  const backendOnly = Boolean(backendChange) && !llamaVersionChanged;
  // A migration at the installed release has no version pair, so llama.cpp's row
  // shows the backend pair instead.
  const backendPair = backendOnly
    ? ([
        backendLabel(status?.from_backend),
        backendLabel(status?.to_backend),
      ] as const)
    : null;
  // Only the migration offer, and only the pair it was measured on: a version update
  // or a hand-picked switch keeps the plain line.
  const restartNote =
    backendOnly &&
    status?.from_backend === "rocm" &&
    status?.to_backend === "vulkan"
      ? "Vulkan is >10% faster than ROCM. No restart needed after update"
      : "No restart needed after update";
  const changelogTags =
    llamaRow?.versionChanged && llamaRow.installedTag && llamaRow.latestTag
      ? { installedTag: llamaRow.installedTag, latestTag: llamaRow.latestTag }
      : null;
  const changelogKey = changelogTags
    ? `${changelogTags.installedTag}\0${changelogTags.latestTag}`
    : null;
  const changelogAvailable = Boolean(changelogKey && !status?.source_build);
  const changelogOpen =
    changelogKey !== null && changelogVersion === changelogKey;
  // Use the same predicate for the panel and its protective height floor.
  const changelogPanelOpen = Boolean(
    !applying && changelogAvailable && changelogOpen && changelogTags,
  );
  // A bundle shows each download beside its own release instead of one total, since
  // not every component reports a size.
  const sizeLabel = bundled ? null : sizeLabelOf(primary?.sizeBytes);
  // A bundle names its components on the rows below the title.
  let title: string;
  if (applying) {
    title = bundled
      ? `Installing ${rows.length} updates...`
      : `Updating ${label}...`;
  } else if (bundled) {
    title = `${rows.length} new updates`;
  } else {
    title = backendOnly ? `New ${label} backend` : `New ${label} update`;
  }
  const updateProgress = status?.job.progress ?? null;
  const jobSucceeded = status?.job.state === "success";
  // Display value animates; aria uses the real progress.
  const displayProgress = useSmoothedProgress(
    applying,
    updateProgress,
    jobSucceeded,
  );

  // Avoid opacity/transform transitions; GPU layer churn can flash.
  return show ? (
    <div
      className={cn(
        positioned
          ? "fixed bottom-4 right-4 z-[9998] w-[calc(100vw-2rem)] max-w-[calc(448px*var(--ui-space-scale,1))]"
          : cn(
              "pointer-events-auto flex w-[calc(100vw-2rem)] max-w-[calc(448px*var(--ui-space-scale,1))] flex-col",
              // Only an open changelog needs a shrinkable height floor. Each extra
              // bundled row adds one text-xs line to the header above it.
              changelogPanelOpen
                ? "min-h-[calc(117px+(93px+18px*var(--llama-update-extra-rows,0))*var(--ui-font-scale,1))] max-[383px]:min-h-[calc(24px+(224px+18px*var(--llama-update-extra-rows,0))*var(--ui-font-scale,1))]"
                : "shrink-0",
            ),
      )}
      style={
        {
          "--llama-update-extra-rows": Math.max(rows.length - 1, 0),
        } as CSSProperties
      }
      data-testid="llama-update-banner"
    >
      {/* Paint the full floor even when the changelog content is short. */}
      <div className="relative flex max-h-[calc(100dvh_-_2rem_-_var(--studio-window-chrome-top,0px))] min-h-0 grow flex-col overflow-hidden rounded-[24px] bg-white px-5 pb-4 pt-5 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-[0_8px_28px_-6px_var(--background)]">
        {applying ? null : (
          <button
            type="button"
            onClick={dismiss}
            className="absolute top-2.5 right-3 flex size-6 items-center justify-center rounded-full text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground"
            aria-label={`Dismiss ${label} update notification`}
          >
            <svg
              aria-hidden="true"
              width="12"
              height="12"
              viewBox="0 0 14 14"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M11 3L3 11M3 3l8 8"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
          </button>
        )}

        <div className="flex min-w-0 shrink-0 items-start gap-4 pr-6">
          <Download
            aria-hidden="true"
            className="mt-1 size-5 shrink-0 text-foreground"
            strokeWidth={1.75}
          />
          <div className="min-w-0">
            <p className="font-heading text-base font-medium text-foreground">
              {title}
            </p>
            {rows.map((row) => (
              <VersionLine
                key={row.component}
                row={row}
                bundled={bundled}
                backendPair={row === llamaRow ? backendPair : null}
              />
            ))}
            <p className="mt-1 text-ui-11 text-muted-foreground/70">
              {sizeLabel ? `${sizeLabel} download · ` : ""}
              {llamaVersionChanged && backendChange
                ? `${backendChange} backend · `
                : ""}
              {restartNote}
            </p>
          </div>
        </div>

        {changelogPanelOpen && changelogTags ? (
          <LlamaUpdateChangelogPanel
            installedTag={changelogTags.installedTag}
            latestTag={changelogTags.latestTag}
          />
        ) : null}

        {applying ? (
          // biome-ignore lint/a11y/useFocusableInteractive: a read-only progress indicator must not add a keyboard stop
          <div
            className="mb-1.5 mt-4 h-1 overflow-hidden rounded-full bg-muted"
            role="progressbar"
            aria-label={`Updating ${label}`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={
              updateProgress != null
                ? Math.round(updateProgress * 100)
                : Math.round(displayProgress * 100)
            }
            data-testid="llama-update-progress"
          >
            <div
              className="h-full rounded-full bg-primary"
              style={{ width: `${Math.max(displayProgress * 100, 2)}%` }}
            />
          </div>
        ) : (
          <div
            className={cn(
              "mt-4 flex shrink-0 flex-wrap items-center gap-x-1 gap-y-2",
              changelogAvailable ? "justify-between" : "justify-end",
            )}
          >
            {changelogAvailable ? (
              <Button
                size="sm"
                variant="ghost"
                className="-ml-2 h-auto whitespace-nowrap rounded-full px-2.5 py-2 text-ui-13 font-medium text-foreground"
                onClick={() =>
                  setChangelogVersion(changelogOpen ? null : changelogKey)
                }
                aria-expanded={changelogOpen}
                data-testid="llama-update-changelog-toggle"
              >
                {changelogOpen
                  ? "Hide what's new"
                  : bundled
                    ? "What's new in llama.cpp"
                    : "Show what's new"}
              </Button>
            ) : null}
            <div className="flex flex-wrap items-center justify-end gap-x-1 gap-y-2">
              <Button
                size="sm"
                variant="ghost"
                className="h-auto whitespace-nowrap rounded-full px-2.5 py-2 text-ui-13 font-medium text-foreground"
                onClick={snooze}
                data-testid="llama-update-snooze-button"
              >
                Remind me later
              </Button>
              <Button
                size="sm"
                // Align pill edge with card padding.
                className="-mr-1 h-auto whitespace-nowrap rounded-full px-3 py-2 text-ui-13"
                onClick={handleUpdate}
                data-testid="llama-update-button"
              >
                Update
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  ) : null;
}
