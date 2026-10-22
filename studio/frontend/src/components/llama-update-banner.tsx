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
  audibleComponents,
  heldJobRows,
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
import { Download, X } from "lucide-react";
import {
  type CSSProperties,
  type MouseEvent,
  type ReactElement,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
// cap for interpolation between coarse backend progress updates.
const RUNNING_CAP = 0.95;

// untranslated labels mirror llama-backend-labels.ts; unknown backends keep their identifiers.
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
  // no audio.cpp runtime yet, so the offer is a first install.
  const lead =
    row.component === "audio.cpp" && row.installedTag === null
      ? "Install the audio runtime"
      : null;
  return (
    <p
      // one row per line keeps the changelog height calculation exact.
      className={cn(
        "mt-0.5 text-xs text-muted-foreground",
        bundled && "truncate",
      )}
      title={
        bundled
          ? `${lead ?? `${row.component} ${from} \u2192`} ${to}${sizeSuffix}`
          : undefined
      }
      data-testid="llama-update-version-line"
    >
      {lead ? `${lead} ` : null}
      {bundled && !lead ? (
        <span className="text-foreground">{row.component} </span>
      ) : null}
      {lead ? null : <>{from} &rarr; </>}
      <span className="font-medium text-foreground">{to}</span>
      {sizeSuffix}
    </p>
  );
}

export function LlamaUpdateBanner({
  enabled = true,
  positioned = true,
}: LlamaUpdateBannerProps): ReactElement | null {
  // an open card remains allowed after recording its components as shown.
  const [shownChannels, setShownChannels] = useState<
    readonly NotificationChannel[]
  >([]);
  const llamaDue = useNotificationDue("llama");
  const audioDue = useNotificationDue("audio");
  const llamaMuted = useNotificationFrequency("llama") === "off";
  const audioMuted = useNotificationFrequency("audio") === "off";
  const [detailsVersion, setDetailsVersion] = useState<string | null>(null);
  const [skipped, setSkipped] = useState<{
    key: string;
    fromKeyboard: boolean;
    message: string;
  } | null>(null);
  const clearSkipped = useCallback(() => setSkipped(null), []);
  // listen regardless of banner preference so muting the UI does not silence cross-tab reload_required resync.
  const { status, visible, applying, apply, dismiss, undoDismiss, snooze } =
    useLlamaUpdateCheck({
      enabled,
      onReloadRequired: resyncInferenceStatusAfterServerModelChange,
    });
  // whisper.cpp rides along with llama.cpp and audio.cpp; it opens the card only to retry its own failed phase.
  const whisperRetry =
    status?.job.state === "error" && status.job.whisperFailed;
  const due: UpdateComponentFlags = {
    llama: llamaDue,
    whisper: whisperRetry,
    audio: audioDue,
  };
  const muted: UpdateComponentFlags = {
    llama: llamaMuted,
    whisper: !whisperRetry,
    audio: audioMuted,
  };

  const migrationPending = Boolean(status?.backend_migration_available);
  const listed = updateBannerComponents({
    llama: Boolean(status?.llama.update_available) || migrationPending,
    whisper: Boolean(status?.whisper?.update_available),
    audio: Boolean(status?.audio?.update_available),
  });
  // muted components stay listed because update installs them, but they are not recorded.
  const recordKey = audibleComponents(listed, muted)
    .map(updateComponentKey)
    .join(" ");
  // use each component's release pair and size because the top-level fields belong to llama.cpp.
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
  // treat errors as in flight so a failed phase keeps its retry visible.
  const jobState = status?.job.state;
  const inFlight = applying || jobState === "error";
  const [jobRows, setJobRows] = useState<UpdateRow[] | null>(null);
  const nextJobRows = heldJobRows(jobRows, applying, liveRows);
  if (nextJobRows !== jobRows) {
    setJobRows(nextJobRows);
  }
  const rows = jobRows ?? liveRows;
  const label = updateComponentsLabel(rows.map((row) => row.component));
  const bundled = rows.length > 1;
  const llamaRow = rows.find((row) => row.component === "llama.cpp") ?? null;
  const primary = rows[0] ?? null;

  async function handleUpdate() {
    // prefer the job message for bundles, migrations, and unchanged audio runtimes after refresh.
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
      // the failing phase may differ from the component named by the card.
      toast.error(`Update failed: ${result.error ?? "unknown error"}`);
    }
  }

  const livePref = updateBannerAllowed(
    listed,
    due,
    muted,
    shownChannels.length > 0,
  );
  // keep the preference while a chained apply removes completed components from status.
  const [heldPref, setHeldPref] = useState<boolean | null>(null);
  useEffect(() => {
    setHeldPref((prev) => heldUpdateBannerPref(prev, inFlight, livePref));
  }, [inFlight, livePref]);
  const showBannerPref = heldPref ?? livePref;
  const show =
    showBannerPref &&
    visible &&
    status != null &&
    rows.length > 0 &&
    (llamaUpdateOffered(status) || applying);
  const tabVisible = useDocumentVisible();
  useEffect(() => {
    // record every audible listed component to suppress separate cards during the quiet period.
    if (show && tabVisible && livePref) {
      const fresh = (recordKey.split(" ") as NotificationChannel[])
        .filter(Boolean)
        .filter((channel) => !shownChannels.includes(channel));
      if (fresh.length > 0) {
        for (const channel of fresh) {
          markNotificationShown(channel);
        }
        setShownChannels([...shownChannels, ...fresh]);
      }
    } else if (!(show || skipped) && shownChannels.length > 0) {
      // the skip notice still stands for the card, so Undo can reopen it past the frequency gate.
      setShownChannels([]);
    }
  }, [show, skipped, tabVisible, livePref, recordKey, shownChannels]);
  // a migration can reapply the automatic backend at the installed release.
  const backendChange =
    llamaRow && status?.backend_migration_available && status.to_backend
      ? `${backendLabel(status.from_backend)} \u2192 ${backendLabel(status.to_backend)}`
      : null;
  const llamaVersionChanged = Boolean(llamaRow?.versionChanged);
  const backendOnly = Boolean(backendChange) && !llamaVersionChanged;
  // a skip dismisses the whole offer, so a bundle reads plural even when llama.cpp only migrates.
  const skipLabel = bundled
    ? "Skip these versions"
    : backendOnly
      ? "Skip this change"
      : "Skip this version";

  function handleSkip(event: MouseEvent<HTMLButtonElement>) {
    const skippedTag = bundled || backendOnly ? null : primary?.latestTag;
    const key = dismiss();
    if (!key) {
      return;
    }
    setSkipped({
      key,
      // detail is 0 for Enter or Space, so only keyboard users get focus moved to Undo.
      fromKeyboard: event.detail === 0,
      message: skippedTag
        ? `Skipped ${label} ${skippedTag} for now`
        : `Skipped the ${label} ${bundled ? "updates" : "update"} for now`,
    });
  }
  // show the backend pair when a migration has no version change.
  const backendPair = backendOnly
    ? ([
        backendLabel(status?.from_backend),
        backendLabel(status?.to_backend),
      ] as const)
    : null;
  // only the measured migration pair gets the speed note; other updates do not.
  const restartNote =
    backendOnly &&
    status?.from_backend === "rocm" &&
    status?.to_backend === "vulkan"
      ? "Vulkan is >10% faster than ROCM. No restart needed"
      : "No restart needed";
  const changelogTags =
    llamaRow?.versionChanged && llamaRow.installedTag && llamaRow.latestTag
      ? { installedTag: llamaRow.installedTag, latestTag: llamaRow.latestTag }
      : null;
  const changelogAvailable = Boolean(changelogTags && !status?.source_build);
  // a different offer starts collapsed.
  const detailsKey = [
    ...rows.map(
      (row) => `${row.component}\0${row.installedTag}\0${row.latestTag}`,
    ),
    backendChange ?? "",
  ].join("\n");
  const detailsOpen = !applying && detailsVersion === detailsKey;
  // the panel and its protective height floor must use the same predicate.
  const changelogPanelOpen = Boolean(
    detailsOpen && changelogAvailable && changelogTags,
  );
  const sizeLabel = sizeLabelOf(
    rows.every((row) => row.sizeBytes && row.sizeBytes > 0)
      ? rows.reduce((total, row) => total + (row.sizeBytes ?? 0), 0)
      : null,
  );
  const summary = [
    bundled ? label : null,
    sizeLabel ? `${sizeLabel} download` : null,
    backendChange ? `${backendChange} backend` : null,
    restartNote,
  ]
    .filter(Boolean)
    .join(" · ");
  // use the phase record because migration and source-build offers disappear while jobs run.
  const runningPhase = status?.job.runningPhase ?? null;
  const phaseIndex = rows.findIndex(
    (row) => updateComponentKey(row.component) === runningPhase,
  );
  // no audio.cpp runtime yet, so this offer installs it.
  const firstInstall =
    !bundled &&
    primary?.component === "audio.cpp" &&
    primary.installedTag === null;
  let title: string;
  if (applying && !bundled) {
    title = firstInstall
      ? "Installing the audio runtime..."
      : `Updating ${label}...`;
  } else if (applying) {
    title =
      phaseIndex >= 0
        ? `Installing ${rows[phaseIndex].component} (${phaseIndex + 1} of ${rows.length})...`
        : `Installing ${rows.length} updates...`;
  } else if (bundled) {
    title = `${rows.length} runtime updates`;
  } else {
    title = firstInstall
      ? "Install the audio runtime"
      : backendOnly
        ? `New ${label} backend`
        : `New ${label} update`;
  }
  const updateProgress = status?.job.progress ?? null;
  const jobSucceeded = status?.job.state === "success";
  // aria reports real progress while the visible value is smoothed.
  const displayProgress = useSmoothedProgress(
    applying,
    updateProgress,
    jobSucceeded,
  );
  // a returning card or a hidden banner retires the notice.
  useEffect(() => {
    if (show || !enabled) {
      setSkipped(null);
    }
  }, [show, enabled]);

  if (!show) {
    return skipped && enabled ? (
      <SkippedNotice
        message={skipped.message}
        focusUndo={skipped.fromKeyboard}
        positioned={positioned}
        onUndo={() => {
          undoDismiss(skipped.key);
          setSkipped(null);
        }}
        onExpire={clearSkipped}
      />
    ) : null;
  }

  // opacity and transform transitions can flash during gpu layer churn.
  return (
    <div
      className={cn(
        positioned
          ? "fixed bottom-4 right-4 z-[9998] w-[calc(100vw-2rem)] max-w-[calc(448px*var(--ui-space-scale,1))]"
          : cn(
              "pointer-events-auto flex w-[calc(100vw-2rem)] max-w-[calc(448px*var(--ui-space-scale,1))] flex-col",
              // the desktop floor plus 22px for the details toggle; each bundled row adds one text-xs line.
              changelogPanelOpen
                ? "min-h-[calc(117px+(115px+18px*var(--llama-update-extra-rows,0))*var(--ui-font-scale,1))] max-[383px]:min-h-[calc(24px+(246px+18px*var(--llama-update-extra-rows,0))*var(--ui-font-scale,1))]"
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
      {/* fill the height floor even when the changelog content is short. */}
      <div className="relative flex max-h-[calc(100dvh_-_2rem_-_var(--studio-window-chrome-top,0px))] min-h-0 grow flex-col overflow-hidden rounded-[24px] bg-white px-5 pb-4 pt-5 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-[0_8px_28px_-6px_var(--background)]">
        {/* Hides the card until the app is reopened, the same skip What's new offers. */}
        {applying ? null : (
          <button
            type="button"
            onClick={handleSkip}
            aria-label={`Hide the ${label} ${bundled ? "updates" : "update"} until Unsloth is reopened`}
            className="absolute right-3 top-3 flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            data-testid="llama-update-close-button"
          >
            <X aria-hidden="true" className="size-4" strokeWidth={2} />
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
            <p
              className="mt-0.5 text-xs text-muted-foreground"
              data-testid="llama-update-summary"
            >
              {summary}
            </p>
            {applying ? null : (
              <button
                type="button"
                className="-mx-1 mt-1 block rounded px-1 text-xs font-medium text-foreground underline decoration-foreground/30 underline-offset-2 hover:decoration-foreground/70"
                onClick={() =>
                  setDetailsVersion(detailsOpen ? null : detailsKey)
                }
                aria-expanded={detailsOpen}
                data-testid="llama-update-changelog-toggle"
              >
                {detailsOpen ? "Hide what's new" : "What's new"}
              </button>
            )}
            {detailsOpen ? (
              <div className="mt-2" data-testid="llama-update-details">
                {rows.map((row) => (
                  <VersionLine
                    key={row.component}
                    row={row}
                    bundled={bundled}
                    backendPair={row === llamaRow ? backendPair : null}
                  />
                ))}
              </div>
            ) : null}
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
          <div className="mt-4 flex shrink-0 flex-wrap items-center justify-end gap-x-1 gap-y-2">
            {detailsOpen ? (
              <Button
                size="sm"
                variant="ghost"
                className="-ml-2 mr-auto h-auto whitespace-nowrap rounded-full px-2.5 py-2 text-ui-13 font-medium text-muted-foreground hover:text-foreground"
                onClick={handleSkip}
                data-testid="llama-update-skip-button"
              >
                {skipLabel}
              </Button>
            ) : null}
            <Button
              size="sm"
              variant="ghost"
              className="h-auto whitespace-nowrap rounded-full px-2.5 py-2 text-ui-13 font-medium text-foreground"
              onClick={snooze}
              data-testid="llama-update-snooze-button"
            >
              Later
            </Button>
            <Button
              size="sm"
              // align the pill edge with the card padding.
              className="-mr-1 h-auto whitespace-nowrap rounded-full px-3 py-2 text-ui-13"
              onClick={handleUpdate}
              data-testid="llama-update-button"
            >
              Update
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

// long enough to read and reach Undo; hover or focus holds it open.
const SKIPPED_NOTICE_MS = 6000;

function SkippedNotice({
  message,
  focusUndo,
  positioned,
  onUndo,
  onExpire,
}: {
  message: string;
  focusUndo: boolean;
  positioned: boolean;
  onUndo: () => void;
  onExpire: () => void;
}): ReactElement {
  const undoRef = useRef<HTMLButtonElement>(null);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  // the skip button is gone, so keyboard focus moves to Undo instead of the page.
  useEffect(() => {
    if (focusUndo) {
      undoRef.current?.focus({ preventScroll: true });
    }
  }, [focusUndo]);
  // expiring under focus would drop it on the page body.
  useEffect(() => {
    if (hovered || focused) {
      return;
    }
    const timer = setTimeout(onExpire, SKIPPED_NOTICE_MS);
    return () => clearTimeout(timer);
  }, [hovered, focused, onExpire]);
  return (
    <div
      className={cn(
        "w-[calc(100vw-2rem)] max-w-[calc(448px*var(--ui-space-scale,1))]",
        positioned
          ? "fixed bottom-4 right-4 z-[9998]"
          : "pointer-events-auto shrink-0",
      )}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onFocus={() => setFocused(true)}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setFocused(false);
        }
      }}
      data-testid="llama-update-skipped"
    >
      <div className="flex items-center gap-3 rounded-[24px] bg-white py-2.5 pl-5 pr-3 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-[0_8px_28px_-6px_var(--background)]">
        {/* <output> carries the implicit status role, so screen readers announce the skip. */}
        <output
          className="block min-w-0 grow truncate text-ui-13 text-foreground"
          title={message}
        >
          {message}
        </output>
        <Button
          ref={undoRef}
          size="sm"
          variant="ghost"
          className="h-auto shrink-0 whitespace-nowrap rounded-full px-2.5 py-2 text-ui-13 font-medium text-foreground"
          onClick={onUndo}
          data-testid="llama-update-undo-skip"
        >
          Undo
        </Button>
      </div>
    </div>
  );
}
