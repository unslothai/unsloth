// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MascotImg } from "@/components/mascot-img";
import { cn } from "@/lib/utils";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { motion } from "motion/react";
import type { ReactNode } from "react";

export function StepHeader({ title, body }: { title: string; body: string }) {
  return (
    <div className="mb-10 space-y-3 text-center">
      <h2 className="font-heading text-[1.75rem] leading-[1.15] font-semibold tracking-[-0.02em]">
        {title}
      </h2>
      <p className="text-muted-foreground mx-auto max-w-sm text-[0.9375rem] leading-relaxed">
        {body}
      </p>
    </div>
  );
}

export function Mascot({
  src,
  className,
}: { src: string; className?: string }) {
  return (
    <motion.div
      aria-hidden="true"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3, delay: 0.05 }}
      className={cn("mx-auto", className)}
    >
      <MascotImg src={src} className="size-full object-contain" />
    </motion.div>
  );
}

export function Group({
  children,
  className,
}: { children: ReactNode; className?: string }) {
  return <div className={cn("space-y-8", className)}>{children}</div>;
}

export function Tile({
  icon,
  tone = "plain",
}: { icon: IconSvgElement; tone?: "plain" | "accent" | "done" }) {
  return (
    <span
      className={cn(
        "flex size-8 shrink-0 items-center justify-center rounded-full",
        tone === "accent" && "bg-control-accent text-control-accent-foreground",
        tone === "done" &&
          "bg-emerald-500/15 text-emerald-600 dark:text-emerald-500",
        tone === "plain" && "bg-muted text-foreground",
      )}
    >
      <HugeiconsIcon icon={icon} className="size-4" strokeWidth={1.75} />
    </span>
  );
}

export function Row({
  lead,
  label,
  description,
  trailing,
  below,
  muted = false,
}: {
  lead?: ReactNode;
  label: ReactNode;
  description?: ReactNode;
  trailing?: ReactNode;
  below?: ReactNode;
  muted?: boolean;
}) {
  return (
    <div>
      <div className="flex items-center gap-4">
        {lead}
        <div className="min-w-0 flex-1">
          <div
            className={cn(
              "truncate text-sm font-medium",
              muted && "text-muted-foreground",
            )}
          >
            {label}
          </div>
          {description ? (
            <div className="text-muted-foreground mt-0.5 text-xs leading-relaxed">
              {description}
            </div>
          ) : null}
        </div>
        {trailing}
      </div>
      {below ? <div className="mt-5">{below}</div> : null}
    </div>
  );
}

export function ChosenRing({
  on,
  className,
}: { on: boolean; className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "ring-control-accent pointer-events-none absolute inset-0 rounded-[inherit] ring-2 transition-opacity duration-150",
        on ? "opacity-100" : "opacity-0",
        className,
      )}
    />
  );
}
