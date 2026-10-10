// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HubOptionMenu } from "./hub-option-menu";

export type OwnerScope = "unsloth" | "all";

const OPTIONS: { value: OwnerScope; label: string }[] = [
  { value: "unsloth", label: "Unsloth" },
  { value: "all", label: "All" },
];

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
      className="h-8 min-w-[calc(96px*var(--ui-space-scale,1))] gap-1.5 text-ui-11p5"
    />
  );
}
