// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HubOptionMenu } from "./hub-option-menu";

export type OwnerScope = "unsloth" | "all";

const OPTIONS: { value: OwnerScope; label: string }[] = [
  { value: "unsloth", label: "Unsloth" },
  { value: "all", label: "All" },
];

/**
 * "Unsloth / All" publisher scope as a compact dropdown pill that sits beside
 * the view-mode tabs (including in the narrow split pane) instead of taking its
 * own row. Only appears while browsing a model list, never on the hub feed.
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
      className="h-8 text-[11.5px]"
    />
  );
}
