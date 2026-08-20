


import { Button } from "@/components/ui/button";
import { ReleaseNotesPanel } from "@/components/update/release-notes-panel";
import { PRODUCT_NAME } from "@/config/branding";
import type {
  DesktopUpdatePolicyMode,
  RetainedUpdateFailure,
  UpdateInfo,
  UpdateStatus,
} from "@/hooks/use-tauri-update";
import type { CopySupportDiagnosticsResult } from "@/lib/tauri-diagnostics";
import { cn } from "@/lib/utils";
import { CircleAlert, Download } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";

interface UpdateBannerProps {
  status: UpdateStatus;
  info: UpdateInfo | null;
  dismissed: boolean;
  lastFailure: RetainedUpdateFailure | null;
  isExternalServer?: boolean;
  updatePolicyMode: DesktopUpdatePolicyMode;
  manualReleaseUrl: string | null;
  // Release page for this version, preferred over the generic changelog.
  releasePageUrl?: string | null;
  // false fills a shared overlay stack; true self-anchors.
  positioned?: boolean;
  onInstall: () => void;
  onDismiss: () => void;
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
}

const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];
const LEADING_V = /^v/;

function formatVersion(version: string | null | undefined): string {
  if (!version) return "";
  return version.startsWith("v") ? version : `v${version}`;
}

export function UpdateBanner({
  status,
  info,
  dismissed,
  lastFailure,
  isExternalServer = false,
  updatePolicyMode,
  manualReleaseUrl,
  releasePageUrl = null,
  positioned = true,
  onInstall,
  onDismiss,
  onCopyDiagnostics,
}: UpdateBannerProps) {
  const [copying, setCopying] = useState(false);
  const [manualReport, setManualReport] = useState<string | null>(null);
  const [manualMessage, setManualMessage] = useState<string | null>(null);
  // Version whose notes are expanded; a new offer collapses the panel.
  const [notesVersion, setNotesVersion] = useState<string | null>(null);
  const showFailure = Boolean(lastFailure) && !dismissed;
  const showAvailable = status === "available" && !dismissed && !showFailure;
  const show = showFailure || (showAvailable && Boolean(info));
  const isManualLinuxPackage = updatePolicyMode === "manual_linux_package";
  const installDisabled = isManualLinuxPackage
    ? manualReleaseUrl === null
    : isExternalServer;
  const currentVersion = formatVersion(info?.currentVersion);
  const latestVersion = formatVersion(info?.version);
  const Icon = showFailure ? CircleAlert : Download;
  // The Studio version offered. Not a notes key; it scopes the expanded state.
  const notesTargetVersion = info?.version?.replace(LEADING_V, "") ?? null;
  const notesOpen =
    notesTargetVersion !== null && notesVersion === notesTargetVersion;

  async function handleCopyDiagnostics() {
    setCopying(true);
    try {
      const result = await onCopyDiagnostics();
      if (result.ok) {
        setManualReport(null);
        setManualMessage(null);
      } else {
        setManualReport(result.report);
        setManualMessage(
          result.error ??
            "Clipboard copy failed. Select and copy the diagnostics below.",
        );
      }
    } catch (error) {
      setManualReport(null);
      setManualMessage(`Diagnostics copy failed: ${String(error)}`);
    } finally {
      setCopying(false);
    }
  }

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, y: 12, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 8, scale: 0.97 }}
          transition={{ duration: 0.35, ease: EASE_OUT_QUART }}
          className={cn(
            // Wider than the other overlays: notes preview plus three buttons.
            positioned
              ? "fixed bottom-4 right-4 z-[9999] w-[calc(100vw-2rem)] max-w-[448px]"
              : cn(
                  "pointer-events-auto flex w-[calc(100vw-2rem)] max-w-[448px] flex-col",
                  // Floor = the header and the action row, the parts of this
                  // card that cannot give up height. Under it a capped rail
                  // takes the height out of the notes, which clip;
                  // min-height:auto would instead be the whole card, so this
                  // one would yield nothing and clip the banner below it.
                  //
                  // Its own constants, not the browser card's: this card
                  // carries one more status line under the version, worth about
                  // 20px at the default type size and 24px at the largest.
                  // Measured the same way, at every step from 15px to 20px:
                  // 204, 210, 215, 221, 227, 233 at the widths where the action
                  // pair holds together, and 204, 210, 262, 269, 277, 304 below
                  // 384px where it wraps onto a row of its own. See
                  // web/update-banner for why the floor is split into a fixed
                  // and a scaled part, and why the narrow regime needs its own.
                  //
                  // The failure card has no notes to give up, so shrinking it
                  // could only clip the diagnostics and the retry button. It
                  // holds its height and the rail scrolls instead.
                  showFailure
                    ? "shrink-0"
                    : "min-h-[calc(117px+93px*var(--ui-font-scale,1))] max-[383px]:min-h-[calc(24px+224px*var(--ui-font-scale,1))]",
                ),
          )}
          // See the browser card: dismissible, so it may cover the composer.
          data-overlay-dismissible="true"
          data-testid="tauri-update-banner"
        >
          <div className="relative flex max-h-[calc(100dvh_-_2rem)] min-h-0 flex-col overflow-hidden rounded-[24px] bg-white px-5 pb-4 pt-5 shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-[0_8px_28px_-6px_rgba(0,0,0,0.28)]">
            <button
              type="button"
              onClick={onDismiss}
              className="absolute top-2.5 right-3 flex size-6 items-center justify-center rounded-full text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground"
              aria-label="Dismiss app update notification"
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

            <div className="flex min-w-0 shrink-0 items-start gap-4 pr-6">
              <Icon
                aria-hidden="true"
                className="mt-1 size-5 shrink-0 text-foreground"
                strokeWidth={1.75}
              />
              <div className="min-w-0">
                <p className="font-heading text-base font-medium text-foreground">
                  {showFailure
                    ? "App update failed"
                    : `New ${PRODUCT_NAME} version`}
                </p>
                {showFailure ? null : (
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    {currentVersion} &rarr;{" "}
                    <span className="font-medium text-foreground">
                      {latestVersion}
                    </span>
                  </p>
                )}
                <p className="mt-1 text-ui-11 text-muted-foreground/70">
                  {showFailure
                    ? "Backend recovered. Diagnostics are still available."
                    : isManualLinuxPackage
                      ? "Open the GitHub release page to install the Linux package"
                      : isExternalServer
                        ? "Run `unsloth studio update` from your terminal"
                        : "A new app update is available"}
                </p>
              </div>
            </div>

            {showFailure && lastFailure && (
              <p className="mt-3 line-clamp-2 shrink-0 text-xs text-destructive">
                {lastFailure.error}
              </p>
            )}

            {!showFailure && notesTargetVersion ? (
              <ReleaseNotesPanel
                version={notesTargetVersion}
                open={notesOpen}
                className="min-h-0 flex-1"
                releaseNotesUrl={releasePageUrl ?? manualReleaseUrl}
              />
            ) : null}

            <div
              className={cn(
                // Wraps on a narrow card, never compresses on a short one.
                "mt-4 flex shrink-0 flex-wrap items-center gap-x-1 gap-y-2",
                !showFailure && notesTargetVersion
                  ? "justify-between"
                  : "justify-end",
              )}
            >
              {!showFailure && notesTargetVersion ? (
                <Button
                  size="sm"
                  variant="ghost"
                  // same type size as the action buttons
                  className="-ml-2 h-auto whitespace-nowrap rounded-full px-2.5 py-2 text-ui-13 font-medium text-foreground"
                  onClick={() =>
                    setNotesVersion(notesOpen ? null : notesTargetVersion)
                  }
                  aria-expanded={notesOpen}
                  data-testid="tauri-update-release-notes-toggle"
                >
                  {notesOpen ? "Hide release notes" : "Show release notes"}
                </Button>
              ) : null}
              {showFailure ? (
                <>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-auto rounded-full px-3 py-2 text-ui-13 font-medium text-foreground"
                    onClick={() => {
                      handleCopyDiagnostics().catch(console.error);
                    }}
                  >
                    {copying ? "Copying..." : "Copy diagnostics"}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-auto rounded-full px-3 py-2 text-ui-13 font-medium text-foreground"
                    onClick={onDismiss}
                  >
                    Later
                  </Button>
                  <Button
                    size="sm"
                    className="-mr-1 h-auto rounded-full px-3.5 py-2 text-ui-13"
                    onClick={onInstall}
                    disabled={installDisabled}
                  >
                    {isManualLinuxPackage
                      ? "Open release page"
                      : "Retry update"}
                  </Button>
                </>
              ) : (
                // wrap + right-align so the action pair stays together
                <div className="flex flex-wrap items-center justify-end gap-x-1 gap-y-2">
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-auto whitespace-nowrap rounded-full px-2.5 py-2 text-ui-13 font-medium text-foreground"
                    onClick={onDismiss}
                  >
                    Remind me later
                  </Button>
                  <Button
                    size="sm"
                    className="-mr-1 h-auto whitespace-nowrap rounded-full px-3 py-2 text-ui-13"
                    onClick={onInstall}
                    disabled={installDisabled}
                  >
                    {isManualLinuxPackage ? "Open release page" : "Update"}
                  </Button>
                </div>
              )}
            </div>
            {(manualMessage || manualReport) && (
              // The clipboard fallback, and the one region of the failure card
              // that may give up height. The card is capped at the viewport and
              // clips, and the rail cannot scroll to what that cap hides, so
              // without a scroller here the report the reader is being asked to
              // select and copy is the part that goes missing in a short window.
              <div
                className="hover-scrollbar min-h-0 flex-1 overflow-y-auto overscroll-contain"
                data-testid="tauri-update-manual-report"
              >
                {manualMessage && (
                  <p className="mt-3 text-xs text-destructive">
                    {manualMessage}
                  </p>
                )}
                {manualReport && (
                  <textarea
                    readOnly={true}
                    value={manualReport}
                    onFocus={(event) => event.currentTarget.select()}
                    className="mt-2 h-28 w-full resize-none rounded-lg border border-border/50 bg-muted/30 p-2 font-mono text-ui-10 text-muted-foreground"
                  />
                )}
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
