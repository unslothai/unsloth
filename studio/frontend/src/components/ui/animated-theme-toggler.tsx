// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react"
import { Moon, Sun } from "lucide-react"
import { flushSync } from "react-dom"

import { cn } from "@/lib/utils"
import { prefersReducedMotion, setTheme } from "@/features/settings"

interface AnimatedThemeTogglerProps extends React.ComponentPropsWithoutRef<"button"> {
  duration?: number
}

export function useAnimatedThemeToggle(duration = 400) {
  const [isDark, setIsDark] = useState(false)
  const anchorRef = useRef<HTMLElement | null>(null)
  const inFlightRef = useRef(false)

  useEffect(() => {
    const updateTheme = () => {
      setIsDark(document.documentElement.classList.contains("dark"))
    }
    updateTheme()
    const observer = new MutationObserver(updateTheme)
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    })
    return () => observer.disconnect()
  }, [])

  const toggleTheme = useCallback(async () => {
    // One toggle per animation. Clicks during a slow view transition would
    // otherwise queue up and land as invisible back-and-forth flips, so the
    // theme looks stuck until an odd number of clicks gets through.
    if (inFlightRef.current) return
    const anchorRect = anchorRef.current?.getBoundingClientRect() ?? null

    const applyTheme = () => {
      flushSync(() => {
        // Read the live class instead of React state, which can lag the DOM
        // while a transition is being captured.
        const nextDark = !document.documentElement.classList.contains("dark")
        setIsDark(nextDark)
        setTheme(nextDark ? "dark" : "light")
      })
    }

    // Skip the view transition (its clip-path runs via the Web Animations API,
    // which CSS force-reduced-motion cannot reach) when reduced motion is set.
    if (!document.startViewTransition || prefersReducedMotion()) {
      applyTheme()
      return
    }

    inFlightRef.current = true
    try {
      const transition = document.startViewTransition(applyTheme)
      await transition.ready

      if (anchorRect) {
        const { top, left, width, height } = anchorRect
        const x = left + width / 2
        const y = top + height / 2
        const maxRadius = Math.hypot(
          Math.max(left, window.innerWidth - left),
          Math.max(top, window.innerHeight - top)
        )
        document.documentElement.animate(
          {
            clipPath: [
              `circle(0px at ${x}px ${y}px)`,
              `circle(${maxRadius}px at ${x}px ${y}px)`,
            ],
          },
          {
            duration,
            easing: "ease-in-out",
            pseudoElement: "::view-transition-new(root)",
          }
        )
      }
      await transition.finished
    } catch {
      // A skipped transition still applied the theme.
    } finally {
      inFlightRef.current = false
    }
  }, [duration])

  return { isDark, toggleTheme, anchorRef }
}

export const AnimatedThemeToggler = ({
  className,
  duration = 400,
  ...props
}: AnimatedThemeTogglerProps) => {
  const { isDark, toggleTheme, anchorRef } = useAnimatedThemeToggle(duration)

  return (
    <button
      ref={(node) => {
        anchorRef.current = node
      }}
      onClick={toggleTheme}
      className={cn(className)}
      {...props}
    >
      {isDark ? <Sun /> : <Moon />}
      <span className="sr-only">Toggle theme</span>
    </button>
  )
}
