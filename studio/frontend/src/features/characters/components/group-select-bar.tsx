// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useT } from "@/i18n";
import { UserGroupIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { Character } from "../types";
import { Portrait } from "./portrait";

/** The floating bar while picking characters for a group chat. */
export function GroupSelectBar({
  selected,
  starting,
  onStart,
}: {
  selected: Character[];
  starting: boolean;
  onStart: () => void;
}) {
  const t = useT();
  return (
    <div className="pointer-events-none sticky bottom-6 z-20 mt-8 flex justify-center">
      <div className="pointer-events-auto flex items-center gap-3 rounded-full bg-popover py-2 pl-5 pr-2 shadow-[0_12px_32px_-12px_rgb(0_0_0/0.35)] ring-1 ring-border">
        <span className="text-ui-14 text-muted-foreground">
          {selected.length < 2
            ? t("characters.groupNeedsTwo")
            : t("characters.selectedCount", { count: selected.length })}
        </span>
        <div className="flex -space-x-2">
          {selected.slice(0, 5).map((character) => (
            <Portrait
              key={character.id}
              id={character.id}
              name={character.name}
              hasImage={character.hasAvatar}
              rev={character.avatarRev}
              className="size-7 rounded-full text-[0.8rem] ring-2 ring-popover"
            />
          ))}
        </div>
        <Button variant="primary" className="rounded-full" disabled={selected.length < 2 || starting} onClick={onStart}>
          <HugeiconsIcon icon={UserGroupIcon} strokeWidth={1.75} className="size-icon" />
          {t("characters.startGroupChat")}
        </Button>
      </div>
    </div>
  );
}
