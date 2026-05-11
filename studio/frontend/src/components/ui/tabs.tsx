// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

/* eslint-disable react-refresh/only-export-components */

import { type VariantProps, cva } from "class-variance-authority";
import { motion } from "motion/react";
import { Tabs as TabsPrimitive } from "radix-ui";
import * as React from "react";

import { cn } from "@/lib/utils";

const TabsContext = React.createContext<{ value?: string; id: string }>({
  id: "",
});

export function Tabs({
  className,
  orientation = "horizontal",
  value,
  defaultValue,
  onValueChange,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.Root>): React.ReactElement {
  const [internal, setInternal] = React.useState(defaultValue ?? "");
  const current = value ?? internal;
  const id = React.useId();

  return (
    <TabsContext.Provider value={{ value: current, id }}>
      <TabsPrimitive.Root
        data-slot="tabs"
        data-orientation={orientation}
        value={current}
        onValueChange={(v) => {
          if (value === undefined) {
            setInternal(v);
          }
          onValueChange?.(v);
        }}
        className={cn(
          "gap-2 group/tabs flex data-[orientation=horizontal]:flex-col",
          className,
        )}
        {...props}
      />
    </TabsContext.Provider>
  );
}

export const tabsListVariants = cva(
  "rounded-4xl p-[3px]  group-data-horizontal/tabs:h-9 group-data-vertical/tabs:rounded-2xl data-[variant=line]:rounded-none group/tabs-list text-muted-foreground inline-flex w-fit items-center justify-center group-data-[orientation=vertical]/tabs:h-fit group-data-[orientation=vertical]/tabs:flex-col",
  {
    variants: {
      variant: {
        default: "bg-muted",
        line: "gap-1 bg-transparent",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export function TabsList({
  className,
  variant = "default",
  children,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.List> &
  VariantProps<typeof tabsListVariants>): React.ReactElement {
  const ctx = React.useContext(TabsContext);
  const listRef = React.useRef<HTMLDivElement>(null);
  const [pill, setPill] = React.useState<{
    x: number;
    y: number;
    width: number;
    height: number;
    ready: boolean;
  }>({ x: 0, y: 0, width: 0, height: 0, ready: false });

  React.useLayoutEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const measure = (): void => {
      const active = list.querySelector<HTMLElement>(
        '[role="tab"][data-state="active"]',
      );
      if (!active) {
        setPill((prev) => ({ ...prev, ready: false }));
        return;
      }
      setPill({
        x: active.offsetLeft,
        y: active.offsetTop,
        width: active.offsetWidth,
        height: active.offsetHeight,
        ready: true,
      });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(list);
    list
      .querySelectorAll<HTMLElement>('[role="tab"]')
      .forEach((tab) => observer.observe(tab));
    return () => observer.disconnect();
  }, [ctx.value, variant, children]);

  const showPill = variant !== "line";

  return (
    <TabsPrimitive.List
      ref={listRef}
      data-slot="tabs-list"
      data-variant={variant}
      className={cn(tabsListVariants({ variant }), "relative", className)}
      {...props}
    >
      {showPill && pill.ready ? (
        <motion.span
          aria-hidden="true"
          className="pointer-events-none absolute top-0 left-0 rounded-xl bg-background dark:border dark:border-input dark:bg-input/30"
          initial={false}
          animate={{
            x: pill.x,
            y: pill.y,
            width: pill.width,
            height: pill.height,
          }}
          transition={{
            type: "tween",
            duration: 0.25,
            ease: [0.4, 0, 0.2, 1],
          }}
        />
      ) : null}
      {children}
    </TabsPrimitive.List>
  );
}

export function TabsTrigger({
  className,
  value,
  children,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.Trigger>): React.ReactElement {
  return (
    <TabsPrimitive.Trigger
      data-slot="tabs-trigger"
      value={value}
      className={cn(
        "gap-1.5 rounded-xl corner-squircle border border-transparent px-2 py-1 text-sm font-medium group-data-vertical/tabs:px-2.5 group-data-vertical/tabs:py-1.5 [&_svg:not([class*='size-'])]:size-4 focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:outline-ring text-foreground/60 hover:text-foreground dark:text-muted-foreground dark:hover:text-foreground relative inline-flex h-[calc(100%-1px)] flex-1 items-center justify-center whitespace-nowrap transition-colors group-data-[orientation=vertical]/tabs:w-full group-data-[orientation=vertical]/tabs:justify-start focus-visible:ring-[3px] focus-visible:outline-1 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0",
        "group-data-[variant=line]/tabs-list:bg-transparent group-data-[variant=line]/tabs-list:data-[state=active]:bg-transparent dark:group-data-[variant=line]/tabs-list:data-[state=active]:border-transparent dark:group-data-[variant=line]/tabs-list:data-[state=active]:bg-transparent",
        "data-[state=active]:text-foreground dark:data-[state=active]:text-foreground",
        "after:bg-foreground after:absolute after:opacity-0 after:transition-opacity group-data-[orientation=horizontal]/tabs:after:inset-x-0 group-data-[orientation=horizontal]/tabs:after:bottom-[-5px] group-data-[orientation=horizontal]/tabs:after:h-0.5 group-data-[orientation=vertical]/tabs:after:inset-y-0 group-data-[orientation=vertical]/tabs:after:-right-1 group-data-[orientation=vertical]/tabs:after:w-0.5 group-data-[variant=line]/tabs-list:data-[state=active]:after:opacity-100",
        className,
      )}
      {...props}
    >
      <span className="relative z-10">{children}</span>
    </TabsPrimitive.Trigger>
  );
}

export function TabsContent({
  className,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.Content>): React.ReactElement {
  return (
    <TabsPrimitive.Content
      data-slot="tabs-content"
      className={cn("text-sm flex-1 outline-none", className)}
      {...props}
    />
  );
}
