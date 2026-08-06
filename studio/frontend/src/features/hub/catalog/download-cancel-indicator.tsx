


import { Cancel01Icon, PauseIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

/** What stopping this download costs. HTTP keeps the partial to continue from,
 * so it is a pause; Xet has to start over, so it is a cancel. */
export type DownloadStopMode = "cancel" | "pause";

/**
 * The stop glyph beside the percentage during a download. Always visible: it is
 * the only way to stop, so it should not need a hover to be found. Sits in a
 * fixed 16x16 slot so the percentage never shifts.
 */
export function DownloadStopIndicator({ mode }: { mode: DownloadStopMode }) {
  return (
    <span className="hub-cta-indicator">
      <HugeiconsIcon
        icon={mode === "pause" ? PauseIcon : Cancel01Icon}
        strokeWidth={1.75}
      />
    </span>
  );
}
