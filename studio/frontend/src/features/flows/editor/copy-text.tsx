// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import { toast } from "sonner";
import { Copy01Icon } from "./icon-data";

/** Small copy icon for a text value; on a card it shows on hover, elsewhere always. */
export function CopyTextButton({
  text,
  label = "Copy",
  className,
}: {
  text: string;
  label?: string;
  className?: string;
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      data-testid="flow-copy-text"
      className={cn(
        "nodrag nopan flex size-6 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
        className,
      )}
      onClick={(event) => {
        event.stopPropagation();
        void copyToClipboard(text).then((ok) =>
          ok ? toast.success("Copied") : toast.error("Could not copy"),
        );
      }}
      onDoubleClick={(event) => event.stopPropagation()}
    >
      <HugeiconsIcon icon={Copy01Icon} size={13} />
    </button>
  );
}
