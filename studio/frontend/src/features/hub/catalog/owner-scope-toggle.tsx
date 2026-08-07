// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HubOptionMenu } from "./hub-option-menu";

export type OwnerScope = "unsloth" | "all";

const OPTIONS: { value: OwnerScope; label: string }[] = [
  { value: "unsloth", label: "Unsloth" },
  { value: "all", label: "All" },
];

/**
 * "Unsloth / All" publisher scope as a compact dropdown pill beside the
 * view-mode tabs. Only shown while browsing a model list, never on the hub feed.
 */
export function OwnerScopeToggle({
  value,
  onChange,
}: {
  value: OwnerScope;
  onChange: (value: OwnerScope) => void;
}) {
  return (
    <HubOptionMenu<OwnerScope>
      value={value}
      options={OPTIONS}
      onValueChange={onChange}
      ariaLabel="Publisher scope"
      align="end"
      // Extra gap before the chevron; min-width keeps the pill readable.
      className="h-8 min-w-[96px] gap-1.5 text-ui-11p5"
    />
  );
}
