// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import type { ReactNode } from "react";

export const ICON_ACTION =
  "size-[calc(32px*var(--ui-space-scale,1))] shrink-0 rounded-full text-muted-foreground hover:text-foreground";

export function IconAction({
  icon,
  label,
  onClick,
  disabled,
  className,
  iconClassName = "size-4",
  children,
}: {
  /** Omit to draw `children` instead, for an icon that is not a Hugeicon. */
  icon?: IconSvgElement;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  className?: string;
  iconClassName?: string;
  children?: ReactNode;
}) {
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      title={label}
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className={cn(ICON_ACTION, className)}
    >
      {icon ? (
        <HugeiconsIcon icon={icon} className={iconClassName} />
      ) : (
        children
      )}
    </Button>
  );
}
