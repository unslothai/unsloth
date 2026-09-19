// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { Star } from "lucide-react";

export function ModelFavoriteAction({
  favorite,
  onToggle,
  className,
}: {
  favorite: boolean;
  onToggle: () => void;
  className?: string;
}) {
  return (
    <button
      type="button"
      aria-label="Favorites"
      aria-pressed={favorite}
      title={favorite ? "Remove from favorites" : "Add to favorites"}
      onClick={(event) => {
        event.stopPropagation();
        onToggle();
      }}
      className={cn(
        "aria-pressed:opacity-100 flex size-5 shrink-0 items-center justify-center rounded-md transition-colors hover:bg-black/5 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:hover:bg-white/10",
        favorite
          ? "text-amber-600 dark:text-amber-400"
          : "text-muted-foreground/60 hover:text-foreground",
        className,
      )}
    >
      <Star
        aria-hidden="true"
        className="size-3.5"
        fill={favorite ? "currentColor" : "none"}
      />
    </button>
  );
}
