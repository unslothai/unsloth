import type { ComponentProps, ReactElement } from "react";
import { Handle, type HandleProps } from "@xyflow/react";

import { cn } from "@/lib/utils";

export type BaseHandleProps = HandleProps;

export function BaseHandle({
  className,
  children,
  ...props
}: ComponentProps<typeof Handle>): ReactElement {
  return (
    <Handle
      {...props}
      className={cn(
        "dark:border-secondary dark:bg-secondary h-[11px] w-[11px] rounded-full border border-slate-300 bg-slate-100 transition",
        className,
      )}
    >
      {children}
    </Handle>
  );
}
