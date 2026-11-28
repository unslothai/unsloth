// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A textarea that grows with its content, so paired fields stay readable. */
import {
  forwardRef,
  type TextareaHTMLAttributes,
  useLayoutEffect,
  useRef,
} from "react";
import { cn } from "@/lib/utils";

type Props = TextareaHTMLAttributes<HTMLTextAreaElement> & {
  value: string;
  minHeight?: number;
};

export const AutoGrowTextarea = forwardRef<HTMLTextAreaElement, Props>(
  function AutoGrowTextarea(
    { value, minHeight = 80, className, ...props },
    forwardedRef,
  ) {
    const localRef = useRef<HTMLTextAreaElement | null>(null);

    useLayoutEffect(() => {
      const el = localRef.current;
      if (!el) return;
      const fit = () => {
        el.style.height = "auto";
        el.style.height = `${Math.max(el.scrollHeight, minHeight)}px`;
      };
      fit();
      // Width changes rewrap the text, so the height must be measured again.
      const observer = new ResizeObserver(fit);
      observer.observe(el);
      return () => observer.disconnect();
    }, [value, minHeight]);

    return (
      <textarea
        ref={(node) => {
          localRef.current = node;
          if (typeof forwardedRef === "function") forwardedRef(node);
          else if (forwardedRef) forwardedRef.current = node;
        }}
        value={value}
        rows={1}
        className={cn(
          "w-full resize-none overflow-y-auto rounded-lg border border-border/60 bg-transparent px-3 py-2 text-ui-13 leading-relaxed text-foreground outline-none placeholder:text-muted-foreground focus-visible:border-foreground/30",
          className,
        )}
        {...props}
      />
    );
  },
);
