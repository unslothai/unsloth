// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu";

export type ExportFormatOption<Key extends string> = {
  readonly key: Key;
  readonly label: string;
};

const JSONL_SUFFIX = " JSONL";

/** Export formats for a menu, with the JSONL variants folded into one "JSONL" submenu where the first one sat. */
export function ExportFormatItems<Key extends string>({
  options,
  onSelect,
  subContentClassName = "w-48",
}: {
  options: readonly ExportFormatOption<Key>[];
  onSelect: (key: Key, event: Event) => void;
  subContentClassName?: string;
}) {
  const jsonl = options.filter(({ label }) => label.endsWith(JSONL_SUFFIX));
  const item = ({ key, label }: ExportFormatOption<Key>) => (
    <DropdownMenuItem key={key} onSelect={(event) => onSelect(key, event)}>
      {label}
    </DropdownMenuItem>
  );
  if (jsonl.length < 2) return <>{options.map(item)}</>;
  const firstJsonl = options.indexOf(jsonl[0] as ExportFormatOption<Key>);
  const rest = options.filter((option) => !jsonl.includes(option));
  return (
    <>
      {rest.slice(0, firstJsonl).map(item)}
      <DropdownMenuSub>
        <DropdownMenuSubTrigger>JSONL</DropdownMenuSubTrigger>
        <DropdownMenuSubContent className={subContentClassName}>
          {jsonl.map(item)}
        </DropdownMenuSubContent>
      </DropdownMenuSub>
      {rest.slice(firstJsonl).map(item)}
    </>
  );
}
