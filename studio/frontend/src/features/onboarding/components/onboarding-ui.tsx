// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MascotImg } from "@/components/mascot-img";
import { cn } from "@/lib/utils";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { motion, useReducedMotion } from "motion/react";
import type { ReactNode } from "react";

export function StepHeader({ title, body }: { title: string; body: string }) {
  return (
    <div className="mb-7 space-y-2 text-center">
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
      initial={{ scale: 0.85, rotate: -6, opacity: 0 }}
      animate={{ scale: 1, rotate: 0, opacity: 1 }}
      transition={{ type: "spring", stiffness: 260, damping: 18, delay: 0.05 }}
      className={cn("mx-auto", className)}
    >
      <MascotImg src={src} className="size-full object-contain" />
    </motion.div>
  );
}

/** One surface holding several rows, the way Settings groups them. */
export function Group({
  children,
  className,
}: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        "bg-muted/60 divide-border/50 divide-y overflow-hidden rounded-2xl",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function Tile({
  icon,
  tone = "plain",
}: { icon: IconSvgElement; tone?: "plain" | "accent" | "done" }) {
  return (
    <span
      className={cn(
        "flex size-9 shrink-0 items-center justify-center rounded-xl",
        tone === "accent" && "bg-control-accent text-control-accent-foreground",
        tone === "done" &&
          "bg-emerald-500/15 text-emerald-600 dark:text-emerald-500",
        tone === "plain" && "bg-background text-foreground shadow-xs",
      )}
    >
      <HugeiconsIcon icon={icon} className="size-4.5" strokeWidth={1.75} />
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
    <div className="px-5 py-4">
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
      {below ? <div className="mt-4">{below}</div> : null}
    </div>
  );
}

/** Two or three choices in one pill, the active one riding a sliding chip. */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  id,
}: {
  value: T;
  options: { value: T; label: string; icon?: IconSvgElement }[];
  onChange: (next: T) => void;
  label: string;
  id: string;
}) {
  const reduced = useReducedMotion();
  return (
    <fieldset className="hub-tab-toggle inline-flex h-9 shrink-0 items-center rounded-full border-0 p-0.5">
      <legend className="sr-only">{label}</legend>
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(option.value)}
            className={cn(
              "relative flex h-8 items-center gap-1.5 rounded-full px-3.5 text-xs font-medium transition-colors",
              active
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active ? (
              <motion.span
                layoutId={id}
                className="hub-tab-toggle-pill absolute inset-0 rounded-full"
                transition={
                  reduced
                    ? { duration: 0 }
                    : { type: "spring", stiffness: 500, damping: 35, mass: 0.5 }
                }
              />
            ) : null}
            {option.icon ? (
              <HugeiconsIcon
                icon={option.icon}
                className="relative z-10 size-3.5"
              />
            ) : null}
            <span className="relative z-10">{option.label}</span>
          </button>
        );
      })}
    </fieldset>
  );
}

/** The chosen ring of a card or tile, drawn over its edge and eased in, never a fill. */
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
