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
    // One toggle per animation, or clicks mid-transition queue up as invisible flips.
    if (inFlightRef.current) return
    const anchorRect = anchorRef.current?.getBoundingClientRect() ?? null

    const applyTheme = () => {
      flushSync(() => {
        // Read the live class: React state can lag the DOM during a transition capture.
        const nextDark = !document.documentElement.classList.contains("dark")
        setIsDark(nextDark)
        setTheme(nextDark ? "dark" : "light")
      })
    }

    // Reduced motion cannot reach the view transition's Web Animations clip-path, so skip it.
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
