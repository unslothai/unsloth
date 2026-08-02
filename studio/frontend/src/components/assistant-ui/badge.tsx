"use client";

import type { ComponentProps } from "react";
import { Slot } from "radix-ui";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center justify-center gap-1 rounded-md font-medium text-xs transition-colors [&_svg]:size-3 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        outline:
          "border border-input bg-transparent text-muted-foreground hover:bg-accent hover:text-accent-foreground",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        muted:
          "bg-muted text-muted-foreground hover:bg-muted/80 hover:text-foreground",
        ghost:
          "bg-transparent text-muted-foreground hover:bg-accent hover:text-accent-foreground",
        info: "bg-blue-100 text-blue-700 hover:bg-blue-100/80 dark:bg-blue-900/50 dark:text-blue-300",
        warning:
          "bg-amber-100 text-amber-700 hover:bg-amber-100/80 dark:bg-amber-900/50 dark:text-amber-300",
        success:
          "bg-emerald-100 text-emerald-700 hover:bg-emerald-100/80 dark:bg-emerald-900/50 dark:text-emerald-300",
        destructive:
          "bg-red-100 text-red-700 hover:bg-red-100/80 dark:bg-red-900/50 dark:text-red-300",
      },
      size: {
        sm: "px-1.5 py-0.5",
        default: "px-2 py-1",
        lg: "px-2.5 py-1.5 text-sm",
      },
    },
    defaultVariants: {
      variant: "outline",
      size: "default",
    },
  },
);

export type BadgeProps = ComponentProps<"span"> &
  VariantProps<typeof badgeVariants> & {
    asChild?: boolean;
  };

function Badge({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: BadgeProps) {
  const Comp = asChild ? Slot.Root : "span";

  return (
    <Comp
      data-slot="badge"
      data-variant={variant}
      data-size={size}
      className={cn(badgeVariants({ variant, size }), className)}
      {...props}
    />
  );
}

export { Badge, badgeVariants };
