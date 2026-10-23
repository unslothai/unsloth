// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Layers01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Panel } from "@xyflow/react";
import type { ReactElement } from "react";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { SuperCard } from "../../utils/graph/super-cards";

type SuperCardControlsProps = {
  openCard: SuperCard | null;
  combinableCount: number;
  locked: boolean;
  onCombine: () => void;
  onExit: () => void;
  onRename: (id: string, title: string) => void;
  onUngroup: (id: string) => void;
};

export function SuperCardControls({
  openCard,
  combinableCount,
  locked,
  onCombine,
  onExit,
  onRename,
  onUngroup,
}: SuperCardControlsProps): ReactElement | null {
  if (openCard) {
    return (
      <Panel
        position="top-left"
        className="!mt-14 m-3 flex items-center gap-2 rounded-lg border bg-background/95 px-2 py-1 shadow-sm"
      >
        <Breadcrumb>
          <BreadcrumbList className="flex-nowrap">
            <BreadcrumbItem>
              <BreadcrumbLink asChild={true}>
                <button type="button" onClick={onExit}>
                  Recipe
                </button>
              </BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <HugeiconsIcon
                icon={Layers01Icon}
                className="size-3.5 text-primary"
              />
              <Input
                aria-label="Super card name"
                value={openCard.title}
                disabled={locked}
                onChange={(event) => onRename(openCard.id, event.target.value)}
                className="nodrag h-7 w-48 text-sm text-foreground"
              />
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="corner-squircle"
          disabled={locked}
          onClick={() => onUngroup(openCard.id)}
        >
          Ungroup
        </Button>
      </Panel>
    );
  }
  if (combinableCount < 2) {
    return null;
  }
  return (
    <Panel position="top-left" className="!mt-14 m-3">
      <Button
        type="button"
        size="sm"
        className="corner-squircle"
        disabled={locked}
        onClick={onCombine}
      >
        <HugeiconsIcon icon={Layers01Icon} className="size-4" />
        Combine {combinableCount} into super card
      </Button>
    </Panel>
  );
}
