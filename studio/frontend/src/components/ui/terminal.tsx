// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils"
import {
  Children,
  cloneElement,
  isValidElement,
  useEffect,
  useRef,
  useState,
} from "react"
import type { ElementType, ReactElement, ReactNode } from "react"

type TerminalProps = {
  children: ReactNode
  className?: string
  sequence?: boolean
  startOnView?: boolean
}

type InternalLineProps = {
  __isActive?: boolean
  __onDone?: () => void
  __sequence?: boolean
}

function useStartOnView(enabled: boolean): {
  ref: React.RefObject<HTMLDivElement | null>
  started: boolean
} {
  const ref = useRef<HTMLDivElement | null>(null)
  const [isInView, setIsInView] = useState(false)
  const started = !enabled || isInView

  useEffect(() => {
    if (!enabled) {
      return
    }

    const node = ref.current
    if (!node) {
      return
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          setIsInView(true)
          observer.disconnect()
        }
      },
      { threshold: 0.2 }
    )

    observer.observe(node)
    return () => observer.disconnect()
  }, [enabled])

  return { ref, started }
}

export function Terminal({
  children,
  className,
  sequence = true,
  startOnView = true,
}: TerminalProps): ReactElement {
  const { ref, started } = useStartOnView(startOnView)
  const childElements = Children.toArray(children).filter(isValidElement)
  const [activeIndex, setActiveIndex] = useState(0)
  const visibleIndex = sequence
    ? started
      ? activeIndex
      : -1
    : Number.MAX_SAFE_INTEGER

  function handleLineDone(index: number): void {
    if (!sequence) {
      return
    }

    setActiveIndex((prev) => {
      if (prev !== index) {
        return prev
      }
      return Math.min(index + 1, childElements.length)
    })
  }

  return (
    <div
      ref={ref}
      className={cn(
        "w-full rounded-2xl border border-border bg-card px-6 py-5 font-mono text-sm text-foreground shadow-2xl",
        className
      )}
    >
      {childElements.map((child, index) =>
        cloneElement(child, {
          __sequence: sequence,
          __isActive: !sequence || visibleIndex >= index,
          __onDone: () => handleLineDone(index),
          key: child.key ?? index,
        } as InternalLineProps)
      )}
    </div>
  )
}

type AnimatedSpanProps = InternalLineProps & {
  children: ReactNode
  className?: string
  delay?: number
  startOnView?: boolean
}

export function AnimatedSpan({
  children,
  className,
  delay = 0,
  startOnView = false,
  __isActive,
  __sequence,
  __onDone,
}: AnimatedSpanProps): ReactElement {
  const { ref, started } = useStartOnView(startOnView)
  const [visible, setVisible] = useState(false)
  const doneRef = useRef(false)
  const onDoneRef = useRef(__onDone)
  const shouldStart = __sequence ? __isActive : started

  useEffect(() => {
    onDoneRef.current = __onDone
  }, [__onDone])

  useEffect(() => {
    if (!shouldStart || doneRef.current) {
      return
    }

    const timeout = window.setTimeout(() => {
      setVisible(true)
      doneRef.current = true
      onDoneRef.current?.()
    }, delay)

    return () => window.clearTimeout(timeout)
  }, [delay, shouldStart])

  return (
    <div
      ref={ref}
      className={cn(
        "min-h-5 transition-opacity duration-300",
        visible ? "opacity-100" : "opacity-0",
        className
      )}
    >
      {children}
    </div>
  )
}

type TypingAnimationProps = InternalLineProps & {
  children: string
  className?: string
  duration?: number
  delay?: number
  as?: ElementType
  startOnView?: boolean
}

export function TypingAnimation({
  children,
  className,
  duration = 60,
  delay = 0,
  as: Component = "span",
  startOnView = true,
  __isActive,
  __sequence,
  __onDone,
}: TypingAnimationProps): ReactElement {
  const { ref, started } = useStartOnView(startOnView)
  const [typed, setTyped] = useState("")
  const doneRef = useRef(false)
  const onDoneRef = useRef(__onDone)
  const shouldStart = __sequence ? __isActive : started

  useEffect(() => {
    onDoneRef.current = __onDone
  }, [__onDone])

  useEffect(() => {
    if (!shouldStart || doneRef.current) {
      return
    }

    let index = 0
    let intervalId: number | null = null
    const startTimer = window.setTimeout(() => {
      intervalId = window.setInterval(() => {
        index += 1
        setTyped(children.slice(0, index))

        if (index >= children.length) {
          if (intervalId) {
            window.clearInterval(intervalId)
          }
          doneRef.current = true
          onDoneRef.current?.()
        }
      }, duration)
    }, delay)

    return () => {
      window.clearTimeout(startTimer)
      if (intervalId) {
        window.clearInterval(intervalId)
      }
    }
  }, [children, delay, duration, shouldStart])

  return (
    <div ref={ref} className="min-h-5">
      <Component className={cn("whitespace-pre-wrap", className)}>{typed}</Component>
    </div>
  )
}
