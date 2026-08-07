


import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { PICKER_OPTION_FOCUS_VISIBLE_CLASS } from "./picker-focus";

export function SelectablePickerItem({
  active,
  onSelect,
  children,
  className,
  values,
}: {
  active?: boolean;
  onSelect: () => void;
  children: ReactNode;
  className?: string;
  values?: readonly string[];
}) {
  return (
    <button
      type="button"
      data-picker-option="true"
      data-picker-values={values ? JSON.stringify(values) : undefined}
      aria-pressed={active ?? false}
      onClick={onSelect}
      className={cn(
        "flex w-full cursor-pointer select-none items-center gap-2 rounded-[8px] px-2 py-1.5 text-left text-ui-12p5 transition-colors hover:bg-foreground/[0.05] focus-visible:bg-foreground/[0.05]",
        PICKER_OPTION_FOCUS_VISIBLE_CLASS,
        active && "bg-foreground/[0.06]",
        className,
      )}
    >
      {children}
    </button>
  );
}
