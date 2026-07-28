// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client"

import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { getClientPlatform } from "@/components/tauri/window-titlebar"
import { useIsMobile } from "@/hooks/use-mobile"
import {
  SIDEBAR_WIDTH_DEFAULT,
  SIDEBAR_WIDTH_MAX,
  SIDEBAR_WIDTH_MIN,
  clampSidebarWidth,
  useSidebarWidth,
} from "@/hooks/use-sidebar-width"
import { HugeiconsIcon } from "@hugeicons/react"
import { LayoutAlignLeftIcon } from "@hugeicons/core-free-icons"

const noop = () => {}

const SIDEBAR_WIDTH = `${SIDEBAR_WIDTH_DEFAULT}px`
const SIDEBAR_WIDTH_ICON = "3rem"
const SIDEBAR_KEYBOARD_SHORTCUT = "b"

type SidebarContextProps = {
  state: "expanded" | "collapsed"
  open: boolean
  setOpen: (open: boolean) => void
  openMobile: boolean
  setOpenMobile: (open: boolean) => void
  isMobile: boolean
  toggleSidebar: () => void
  hasPinMode: boolean
  pinned: boolean
  setPinned: (value: boolean) => void
  togglePinned: () => void
  width: number
  setWidth: (value: number) => void
  resetWidth: () => void
}

const SidebarContext = React.createContext<SidebarContextProps | null>(null)

function useSidebar() {
  const context = React.useContext(SidebarContext)
  if (!context) {
    throw new Error("useSidebar must be used within a SidebarProvider.")
  }

  return context
}

function SidebarProvider({
  defaultOpen = true,
  open: openProp,
  onOpenChange: setOpenProp,
  pinned: pinnedProp,
  setPinned: setPinnedProp,
  togglePinned: togglePinnedProp,
  className,
  style,
  children,
  ...props
}: React.ComponentProps<"div"> & {
  defaultOpen?: boolean
  open?: boolean
  onOpenChange?: (open: boolean) => void
  pinned?: boolean
  setPinned?: (value: boolean) => void
  togglePinned?: () => void
}) {
  const isMobile = useIsMobile()
  const [openMobile, setOpenMobile] = React.useState(false)
  const { width, setWidth, resetWidth } = useSidebarWidth()

  const prevIsMobileRef = React.useRef(isMobile)
  React.useEffect(() => {
    if (prevIsMobileRef.current && !isMobile) {
      setOpenMobile(false)
    }
    prevIsMobileRef.current = isMobile
  }, [isMobile])

  // Whether pin mode is active (caller provides pinned + setPinned + togglePinned).
  const hasPinMode = pinnedProp !== undefined && setPinnedProp !== undefined && togglePinnedProp !== undefined

  // This is the internal state of the sidebar.
  // We use openProp and setOpenProp for control from outside the component.
  const [_open, _setOpen] = React.useState(defaultOpen)

  // When pin mode is active, open is driven entirely by `pinned` (explicit
  // user toggle). Otherwise fall back to the controlled/uncontrolled pattern.
  const open = hasPinMode ? !!pinnedProp : (openProp ?? _open)

  const setOpen = React.useCallback(
    (value: boolean | ((value: boolean) => boolean)) => {
      const openState = typeof value === "function" ? value(open) : value

      if (hasPinMode) {
        // In pin mode, setOpen controls pinned state.
        setPinnedProp?.(openState)
        return
      }

      if (setOpenProp) {
        setOpenProp(openState)
      } else {
        _setOpen(openState)
      }
    },
    [setOpenProp, open, hasPinMode, setPinnedProp]
  )

  // Helper to toggle the sidebar.
  const toggleSidebar = React.useCallback(() => {
    if (isMobile) return setOpenMobile((open) => !open)
    if (hasPinMode && togglePinnedProp) return togglePinnedProp()
    return setOpen((open) => !open)
  }, [isMobile, setOpen, setOpenMobile, hasPinMode, togglePinnedProp])

  // Adds a keyboard shortcut to toggle the sidebar.
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        event.key === SIDEBAR_KEYBOARD_SHORTCUT &&
        (event.metaKey || event.ctrlKey)
      ) {
        event.preventDefault()
        toggleSidebar()
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [toggleSidebar])

  // We add a state so that we can do data-state="expanded" or "collapsed".
  // This makes it easier to style the sidebar with Tailwind classes.
  const state = open ? "expanded" : "collapsed"

  const pinned = pinnedProp ?? false
  const setPinned = setPinnedProp ?? noop
  const togglePinned = togglePinnedProp ?? noop

  const contextValue = React.useMemo<SidebarContextProps>(
    () => ({
      state,
      open,
      setOpen,
      isMobile,
      openMobile,
      setOpenMobile,
      toggleSidebar,
      hasPinMode,
      pinned,
      setPinned,
      togglePinned,
      width,
      setWidth,
      resetWidth,
    }),
    [state, open, setOpen, isMobile, openMobile, setOpenMobile, toggleSidebar, hasPinMode, pinned, setPinned, togglePinned, width, setWidth, resetWidth]
  )

  return (
    <SidebarContext.Provider value={contextValue}>
      <div
        data-slot="sidebar-wrapper"
        style={
          {
            // The drag handle writes this same property live while resizing.
            "--sidebar-width": `${width}px`,
            "--sidebar-width-icon": SIDEBAR_WIDTH_ICON,
            ...style,
          } as React.CSSProperties
        }
        className={cn(
          "group/sidebar-wrapper has-data-[variant=inset]:bg-sidebar flex min-h-svh w-full",
          className
        )}
        {...props}
      >
        {children}
      </div>
    </SidebarContext.Provider>
  )
}

function Sidebar({
  side = "left",
  variant = "sidebar",
  collapsible = "offcanvas",
  className,
  children,
  dir,
  ...props
}: React.ComponentProps<"div"> & {
  side?: "left" | "right"
  variant?: "sidebar" | "floating" | "inset"
  collapsible?: "offcanvas" | "icon" | "none"
}) {
  const { isMobile, state, openMobile, setOpenMobile, hasPinMode, pinned } = useSidebar()

  if (collapsible === "none") {
    return (
      <div
        data-slot="sidebar"
        className={cn(
          "bg-sidebar text-sidebar-foreground flex h-full w-(--sidebar-width) flex-col",
          className
        )}
        {...props}
      >
        {children}
      </div>
    )
  }

  if (isMobile) {
    return (
      <Sheet open={openMobile} onOpenChange={setOpenMobile} {...props}>
        <SheetContent
          dir={dir}
          data-sidebar="sidebar"
          data-slot="sidebar"
          data-mobile="true"
          className="bg-sidebar text-sidebar-foreground w-2/3 max-w-[18rem] p-0 [&>button]:hidden"
          side={side}
        >
          <SheetHeader className="sr-only">
            <SheetTitle>Sidebar</SheetTitle>
            <SheetDescription>Displays the mobile sidebar.</SheetDescription>
          </SheetHeader>
          <div className="flex h-full w-full flex-col">{children}</div>
        </SheetContent>
      </Sheet>
    )
  }

  return (
    <div
      className={cn(
        "group peer text-sidebar-foreground relative shrink-0",
        hasPinMode && pinned && "w-(--sidebar-width)",
        hasPinMode && !pinned && "w-(--sidebar-width-icon)",
      )}
      data-state={state}
      data-collapsible={state === "collapsed" ? collapsible : ""}
      data-variant={variant}
      data-side={side}
      data-slot="sidebar"
    >
      {/* This is what handles the sidebar gap on desktop */}
      <div
        data-slot="sidebar-gap"
        className={cn(
          "relative bg-transparent shrink-0",
          "group-data-[side=right]:rotate-180",
          hasPinMode
            ? cn(
                // Pin mode: always push content. Expanded when pinned.
                pinned
                  ? "w-(--sidebar-width)"
                  : (variant === "floating" || variant === "inset"
                      ? "w-[calc(var(--sidebar-width-icon)+(--spacing(4)))]"
                      : "w-(--sidebar-width-icon)"),
              )
            : cn(
                // Legacy mode: original shadcn behavior.
                "w-(--sidebar-width)",
                "group-data-[collapsible=offcanvas]:w-0",
                variant === "floating" || variant === "inset"
                  ? "group-data-[collapsible=icon]:w-[calc(var(--sidebar-width-icon)+(--spacing(4)))]"
                  : "group-data-[collapsible=icon]:w-(--sidebar-width-icon)",
              ),
        )}
      />
      <div
        data-slot="sidebar-container"
        data-side={side}
        className={cn(
          hasPinMode
            ? cn(
                // Pin mode: always push content, full height.
                "absolute top-0 bottom-0 flex w-(--sidebar-width) data-[side=left]:left-0",
                "group-data-[collapsible=icon]:w-(--sidebar-width-icon)",
              )
            : cn(
                // Legacy mode: fixed to viewport (original shadcn behavior).
                "fixed inset-y-0 z-10 flex h-svh w-(--sidebar-width) data-[side=left]:left-0 data-[side=left]:group-data-[collapsible=offcanvas]:left-[calc(var(--sidebar-width)*-1)] data-[side=right]:right-0 data-[side=right]:group-data-[collapsible=offcanvas]:right-[calc(var(--sidebar-width)*-1)]",
              ),
          // Adjust the padding for floating and inset variants.
          variant === "floating" || variant === "inset"
            ? "p-2 group-data-[collapsible=icon]:w-[calc(var(--sidebar-width-icon)+(--spacing(4))+2px)]"
            : !hasPinMode && "group-data-[collapsible=icon]:w-(--sidebar-width-icon) group-data-[side=left]:border-r group-data-[side=right]:border-l",
          className
        )}
        {...props}
      >
        <div
          data-sidebar="sidebar"
          data-slot="sidebar-inner"
          className={cn(
            "bg-sidebar flex size-full flex-col overflow-hidden border-r border-sidebar-border dark:border-r-0",
            "group-data-[variant=floating]:ring-sidebar-border group-data-[variant=floating]:rounded-lg group-data-[variant=floating]:shadow-sm group-data-[variant=floating]:ring-1",
          )}
        >
          {children}
        </div>
        <SidebarResizeHandle />
      </div>
    </div>
  )
}

/** Pointer travel (px) below which a drag counts as a plain click. */
const SIDEBAR_DRAG_SLOP = 4
/** Arrow-key resize step for keyboard users. */
const SIDEBAR_RESIZE_STEP = 16

type SidebarDragState = {
  startX: number
  startWidth: number
  moved: boolean
}

/**
 * Sidebar edge: drag to resize, click to collapse or expand. Arrow keys
 * resize, Home restores the default. The width is painted straight to the
 * wrapper while dragging and only persisted on release.
 */
function SidebarResizeHandle({ className }: { className?: string }) {
  const { open, toggleSidebar, width, setWidth, resetWidth } = useSidebar()
  const ref = React.useRef<HTMLButtonElement>(null)
  const dragRef = React.useRef<SidebarDragState | null>(null)
  const [dragging, setDragging] = React.useState(false)
  const [hovered, setHovered] = React.useState(false)
  const [isMacPlatform] = React.useState(() => getClientPlatform().includes("mac"))

  // Cached on pointer down so no DOM walk per move.
  const wrapperRef = React.useRef<HTMLElement | null>(null)
  const frameRef = React.useRef(0)
  const pendingRef = React.useRef(0)

  const wrapper = () =>
    wrapperRef.current ??
    ref.current?.closest<HTMLElement>('[data-slot="sidebar-wrapper"]') ??
    null

  // Resizing relayouts the whole shell, and pointermove fires faster than the
  // display refreshes, so coalesce to one paint per frame.
  const paintWidth = React.useCallback((px: number) => {
    pendingRef.current = px
    if (frameRef.current) return
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = 0
      wrapperRef.current?.style.setProperty(
        "--sidebar-width",
        `${pendingRef.current}px`,
      )
    })
  }, [])

  const endDrag = React.useCallback(() => {
    dragRef.current = null
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current)
      frameRef.current = 0
      // Flush the queued frame instead of dropping it.
      wrapperRef.current?.style.setProperty(
        "--sidebar-width",
        `${pendingRef.current}px`,
      )
    }
    wrapperRef.current?.removeAttribute("data-resizing")
    wrapperRef.current = null
    setDragging(false)
    document.body.style.removeProperty("cursor")
    document.body.style.removeProperty("user-select")
  }, [])

  const handlePointerDown = (event: React.PointerEvent<HTMLButtonElement>) => {
    if (event.button !== 0) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    wrapperRef.current =
      ref.current?.closest<HTMLElement>('[data-slot="sidebar-wrapper"]') ?? null
    // Collapsed: grow from the measured rail so the edge tracks the pointer.
    const wrapperLeft = wrapper()?.getBoundingClientRect().left ?? 0
    const railWidth = (ref.current?.getBoundingClientRect().left ?? wrapperLeft) - wrapperLeft
    dragRef.current = {
      startX: event.clientX,
      startWidth: open ? width : railWidth,
      moved: false,
    }
    pendingRef.current = open ? width : railWidth
    wrapperRef.current?.setAttribute("data-resizing", "true")
    setDragging(true)
    document.body.style.setProperty("cursor", "col-resize")
    document.body.style.setProperty("user-select", "none")
  }

  const handlePointerMove = (event: React.PointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current
    if (!drag) return
    const delta = event.clientX - drag.startX
    if (!drag.moved && Math.abs(delta) < SIDEBAR_DRAG_SLOP) return
    drag.moved = true

    const next = drag.startWidth + delta
    if (!open) {
      // Past the minimum, dragging the collapsed rail reopens it.
      if (next >= SIDEBAR_WIDTH_MIN) {
        paintWidth(clampSidebarWidth(next))
        toggleSidebar()
      }
      return
    }
    // Dragging inward stops at the minimum. Collapsing is click or Cmd+B only.
    paintWidth(clampSidebarWidth(next))
  }

  const handlePointerUp = (event: React.PointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current
    if (!drag) return
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    endDrag()

    if (!drag.moved) {
      toggleSidebar()
      return
    }
    // Rail drag that never hit the minimum: leave the stored width alone.
    if (!open) return
    // Commit the last requested width; a frame may still be queued.
    setWidth(pendingRef.current)
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      if (!open) return
      event.preventDefault()
      const step = event.key === "ArrowLeft" ? -SIDEBAR_RESIZE_STEP : SIDEBAR_RESIZE_STEP
      setWidth(width + step)
      return
    }
    if (event.key === "Home") {
      event.preventDefault()
      resetWidth()
    }
  }

  // Clear a stuck cursor override if we unmount mid-drag.
  React.useEffect(() => endDrag, [endDrag])

  return (
    <Tooltip open={hovered && !dragging}>
      <TooltipTrigger asChild>
        <button
          ref={ref}
          type="button"
          data-slot="sidebar-resize-handle"
          data-sidebar="resize-handle"
          data-dragging={dragging || undefined}
          aria-label={open ? "Resize or collapse sidebar" : "Expand sidebar"}
          aria-orientation="vertical"
          aria-valuenow={open ? width : SIDEBAR_WIDTH_MIN}
          aria-valuemin={SIDEBAR_WIDTH_MIN}
          aria-valuemax={SIDEBAR_WIDTH_MAX}
          role="separator"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={endDrag}
          onKeyDown={handleKeyDown}
          onPointerEnter={() => setHovered(true)}
          onPointerLeave={() => setHovered(false)}
          className={cn(
            "absolute inset-y-0 -right-1 z-30 hidden w-2 touch-none select-none sm:block",
            // `!` overrides the app-wide hand cursor on buttons.
            open ? "cursor-col-resize!" : "cursor-e-resize!",
            // Sits exactly on the sidebar border so hover recolours one line.
            "after:absolute after:inset-y-0 after:right-1 after:w-px after:bg-transparent after:transition-colors after:duration-150",
            "hover:after:bg-sidebar-ring/25 data-dragging:after:bg-sidebar-ring/25",
            className
          )}
        />
      </TooltipTrigger>
      <TooltipContent side="right" align="center" className="tooltip-compact">
        <span className="flex flex-col gap-px">
          <span>{open ? "Click to collapse" : "Click to expand"} {isMacPlatform ? "⌘B" : "Ctrl+B"}</span>
          <span className="opacity-70">Drag to resize</span>
        </span>
      </TooltipContent>
    </Tooltip>
  )
}

function SidebarTrigger({
  className,
  onClick,
  ...props
}: React.ComponentProps<typeof Button>) {
  const { toggleSidebar } = useSidebar()

  return (
    <Button
      data-sidebar="trigger"
      data-slot="sidebar-trigger"
      variant="ghost"
      size="icon-sm"
      className={cn(className)}
      onClick={(event) => {
        onClick?.(event)
        toggleSidebar()
      }}
      {...props}
    >
      <HugeiconsIcon icon={LayoutAlignLeftIcon} strokeWidth={1.75} className="size-icon" />
      <span className="sr-only">Toggle Sidebar</span>
    </Button>
  )
}

function SidebarRail({ className, ...props }: React.ComponentProps<"button">) {
  const { toggleSidebar } = useSidebar()

  return (
    <button
      data-sidebar="rail"
      data-slot="sidebar-rail"
      aria-label="Toggle Sidebar"
      tabIndex={-1}
      onClick={toggleSidebar}
      title="Toggle Sidebar"
      className={cn(
        "hover:after:bg-sidebar-border absolute inset-y-0 z-20 hidden w-4 transition-all ease-linear group-data-[side=left]:-right-4 group-data-[side=right]:left-0 after:absolute after:inset-y-0 after:start-1/2 after:w-[2px] sm:flex ltr:-translate-x-1/2 rtl:-translate-x-1/2",
        "in-data-[side=left]:cursor-w-resize in-data-[side=right]:cursor-e-resize",
        "[[data-side=left][data-state=collapsed]_&]:cursor-e-resize [[data-side=right][data-state=collapsed]_&]:cursor-w-resize",
        "hover:group-data-[collapsible=offcanvas]:bg-sidebar group-data-[collapsible=offcanvas]:translate-x-0 group-data-[collapsible=offcanvas]:after:left-full",
        "[[data-side=left][data-collapsible=offcanvas]_&]:-right-2",
        "[[data-side=right][data-collapsible=offcanvas]_&]:-left-2",
        className
      )}
      {...props}
    />
  )
}

function SidebarInset({ className, ...props }: React.ComponentProps<"main">) {
  return (
    <main
      data-slot="sidebar-inset"
      className={cn(
        "bg-background md:peer-data-[variant=inset]:m-2 md:peer-data-[variant=inset]:ml-0 md:peer-data-[variant=inset]:rounded-xl md:peer-data-[variant=inset]:shadow-sm md:peer-data-[variant=inset]:peer-data-[state=collapsed]:ml-2 relative flex min-h-0 w-full flex-1 flex-col",
        className
      )}
      {...props}
    />
  )
}

function SidebarInput({
  className,
  ...props
}: React.ComponentProps<typeof Input>) {
  return (
    <Input
      data-slot="sidebar-input"
      data-sidebar="input"
      className={cn("bg-background h-8 w-full shadow-none", className)}
      {...props}
    />
  )
}

function SidebarHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="sidebar-header"
      data-sidebar="header"
      className={cn("gap-2 p-2 flex flex-col", className)}
      {...props}
    />
  )
}

function SidebarFooter({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="sidebar-footer"
      data-sidebar="footer"
      className={cn("gap-2 p-2 flex flex-col", className)}
      {...props}
    />
  )
}

function SidebarSeparator({
  className,
  ...props
}: React.ComponentProps<typeof Separator>) {
  return (
    <Separator
      data-slot="sidebar-separator"
      data-sidebar="separator"
      className={cn("bg-sidebar-border mx-2 w-auto", className)}
      {...props}
    />
  )
}

function SidebarContent({ className, ref, ...props }: React.ComponentProps<"div"> & { ref?: React.Ref<HTMLDivElement> }) {
  return (
    <div
      ref={ref}
      data-slot="sidebar-content"
      data-sidebar="content"
      className={cn(
        "gap-2 flex min-h-0 flex-1 flex-col overflow-y-auto overflow-x-hidden group-data-[collapsible=icon]:overflow-hidden [&>*]:shrink-0",
        className
      )}
      {...props}
    />
  )
}

function SidebarGroup({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="sidebar-group"
      data-sidebar="group"
      className={cn(
        "p-2 relative flex w-full min-w-0 flex-col",
        className
      )}
      {...props}
    />
  )
}

function SidebarGroupLabel({
  className,
  asChild = false,
  ...props
}: React.ComponentProps<"div"> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "div"

  return (
    <Comp
      data-slot="sidebar-group-label"
      data-sidebar="group-label"
      className={cn(
        "text-[#94a3b8] dark:text-[#666] ring-sidebar-ring h-auto pt-3 pb-2 px-4 rounded-md text-ui-10 font-semibold uppercase tracking-[0em] group-data-[collapsible=icon]:-mt-8 group-data-[collapsible=icon]:opacity-0 focus-visible:ring-1 [&>svg]:size-3 flex shrink-0 items-center outline-hidden [&>svg]:shrink-0",
        className
      )}
      {...props}
    />
  )
}

function SidebarGroupAction({
  className,
  asChild = false,
  ...props
}: React.ComponentProps<"button"> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "button"

  return (
    <Comp
      data-slot="sidebar-group-action"
      data-sidebar="group-action"
      className={cn(
        "text-sidebar-foreground ring-sidebar-ring hover:bg-sidebar-accent hover:text-sidebar-accent-foreground absolute top-3.5 right-3 w-5 rounded-md p-0 focus-visible:ring-1 [&>svg]:size-4 flex aspect-square items-center justify-center outline-hidden transition-transform group-data-[collapsible=icon]:hidden after:absolute after:-inset-2 md:after:hidden [&>svg]:shrink-0",
        className
      )}
      {...props}
    />
  )
}

function SidebarGroupContent({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="sidebar-group-content"
      data-sidebar="group-content"
      className={cn("text-sm w-full", className)}
      {...props}
    />
  )
}

function SidebarMenu({ className, ...props }: React.ComponentProps<"ul">) {
  return (
    <ul
      data-slot="sidebar-menu"
      data-sidebar="menu"
      className={cn("gap-px flex w-full min-w-0 flex-col", className)}
      {...props}
    />
  )
}

function SidebarMenuItem({ className, ...props }: React.ComponentProps<"li">) {
  return (
    <li
      data-slot="sidebar-menu-item"
      data-sidebar="menu-item"
      className={cn("group/menu-item relative", className)}
      {...props}
    />
  )
}

const sidebarMenuButtonVariants = cva(
  "ring-sidebar-ring hover:bg-sidebar-accent hover:text-sidebar-accent-foreground active:bg-sidebar-accent active:text-sidebar-accent-foreground data-active:bg-sidebar-accent data-active:text-sidebar-accent-foreground data-open:hover:bg-sidebar-accent data-open:hover:text-sidebar-accent-foreground gap-2 rounded-md p-2 text-left text-sm cursor-pointer group-has-data-[sidebar=menu-action]/menu-item:pr-8 group-data-[collapsible=icon]:w-full! group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:p-2! data-active:font-medium peer/menu-button flex w-full items-center overflow-hidden outline-hidden group/menu-button disabled:pointer-events-none disabled:opacity-50 aria-disabled:pointer-events-none aria-disabled:opacity-50 [&>span:last-child]:truncate group-data-[collapsible=icon]:[&>span]:hidden [&_svg]:size-4 [&_svg]:shrink-0 group-data-[collapsible=icon]:[&_svg]:size-[var(--icon-size)]",
  {
    variants: {
      variant: {
        default: "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
        outline: "bg-background hover:bg-sidebar-accent hover:text-sidebar-accent-foreground shadow-[0_0_0_1px_hsl(var(--sidebar-border))] hover:shadow-[0_0_0_1px_hsl(var(--sidebar-accent))]",
      },
      size: {
        default: "h-9 text-sm",
        sm: "h-8 text-xs",
        lg: "h-12 text-sm group-data-[collapsible=icon]:p-0!",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function SidebarMenuButton({
  asChild = false,
  isActive = false,
  variant = "default",
  size = "default",
  tooltip,
  className,
  ...props
}: React.ComponentProps<"button"> & {
  asChild?: boolean
  isActive?: boolean
  tooltip?: string | React.ComponentProps<typeof TooltipContent>
} & VariantProps<typeof sidebarMenuButtonVariants>) {
  const Comp = asChild ? Slot.Root : "button"
  const { isMobile, state } = useSidebar()
  const isDisabled = Boolean(props.disabled || props["aria-disabled"])

  const button = (
    <Comp
      data-slot="sidebar-menu-button"
      data-sidebar="menu-button"
      data-size={size}
      data-active={isActive}
      className={cn(sidebarMenuButtonVariants({ variant, size }), className)}
      {...props}
    />
  )

  if (!tooltip) {
    return button
  }

  if (typeof tooltip === "string") {
    tooltip = {
      children: tooltip,
      className: "tooltip-compact",
    }
  }

  // A disabled <button> fires no pointer events (and is not focusable), so the
  // tooltip would never open. Wrap it in a focusable span with a hoverable box
  // so the explanation (e.g. why Train/Export are greyed out) is reachable.
  const trigger = isDisabled ? (
    <span tabIndex={0} className="flex w-full">
      {button}
    </span>
  ) : (
    button
  )

  return (
    <Tooltip>
      <TooltipTrigger asChild>{trigger}</TooltipTrigger>
      <TooltipContent
        side="right"
        align="center"
        // Enabled items only show the tooltip when collapsed (icon labels);
        // a disabled item shows it while expanded too, since it explains why.
        hidden={isMobile || (!isDisabled && state !== "collapsed")}
        {...tooltip}
      />
    </Tooltip>
  )
}

function SidebarMenuAction({
  className,
  asChild = false,
  showOnHover = false,
  ...props
}: React.ComponentProps<"button"> & {
  asChild?: boolean
  showOnHover?: boolean
}) {
  const Comp = asChild ? Slot.Root : "button"

  return (
    <Comp
      data-slot="sidebar-menu-action"
      data-sidebar="menu-action"
      className={cn(
        "text-sidebar-foreground ring-sidebar-ring hover:bg-sidebar-accent hover:text-sidebar-accent-foreground peer-hover/menu-button:text-sidebar-accent-foreground absolute top-1.5 right-1 aspect-square w-5 rounded-md p-0 peer-data-[size=default]/menu-button:top-2 peer-data-[size=lg]/menu-button:top-2.5 peer-data-[size=sm]/menu-button:top-1 focus-visible:ring-1 [&>svg]:size-4 flex items-center justify-center outline-hidden transition-transform group-data-[collapsible=icon]:hidden after:absolute after:-inset-2 md:after:hidden [&>svg]:shrink-0",
        showOnHover &&
          "peer-data-active/menu-button:text-sidebar-accent-foreground group-focus-within/menu-item:opacity-100 group-hover/menu-item:opacity-100 data-open:opacity-100 md:opacity-0",
        className
      )}
      {...props}
    />
  )
}

function SidebarMenuBadge({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="sidebar-menu-badge"
      data-sidebar="menu-badge"
      className={cn(
        "text-sidebar-foreground peer-hover/menu-button:text-sidebar-accent-foreground peer-data-active/menu-button:text-sidebar-accent-foreground pointer-events-none absolute right-1 flex h-5 min-w-5 rounded-md px-1 text-xs font-medium peer-data-[size=default]/menu-button:top-1.5 peer-data-[size=lg]/menu-button:top-2.5 peer-data-[size=sm]/menu-button:top-1 flex items-center justify-center tabular-nums select-none group-data-[collapsible=icon]:hidden",
        className
      )}
      {...props}
    />
  )
}

function SidebarMenuSkeleton({
  className,
  showIcon = false,
  ...props
}: React.ComponentProps<"div"> & {
  showIcon?: boolean
}) {
  // Random width between 50 to 90%.
  const [width] = React.useState(() => {
    return `${Math.floor(Math.random() * 40) + 50}%`
  })

  return (
    <div
      data-slot="sidebar-menu-skeleton"
      data-sidebar="menu-skeleton"
      className={cn("h-8 gap-2 rounded-md px-2 flex items-center", className)}
      {...props}
    >
      {showIcon && (
        <Skeleton
          className="size-4 rounded-md"
          data-sidebar="menu-skeleton-icon"
        />
      )}
      <Skeleton
        className="h-4 max-w-(--skeleton-width) flex-1"
        data-sidebar="menu-skeleton-text"
        style={
          {
            "--skeleton-width": width,
          } as React.CSSProperties
        }
      />
    </div>
  )
}

function SidebarMenuSub({ className, ...props }: React.ComponentProps<"ul">) {
  return (
    <ul
      data-slot="sidebar-menu-sub"
      data-sidebar="menu-sub"
      className={cn("border-sidebar-border mx-3.5 translate-x-px gap-1 border-l px-2.5 py-0.5 group-data-[collapsible=icon]:hidden flex min-w-0 flex-col", className)}
      {...props}
    />
  )
}

function SidebarMenuSubItem({
  className,
  ...props
}: React.ComponentProps<"li">) {
  return (
    <li
      data-slot="sidebar-menu-sub-item"
      data-sidebar="menu-sub-item"
      className={cn("group/menu-sub-item relative", className)}
      {...props}
    />
  )
}

function SidebarMenuSubButton({
  asChild = false,
  size = "md",
  isActive = false,
  className,
  ...props
}: React.ComponentProps<"a"> & {
  asChild?: boolean
  size?: "sm" | "md"
  isActive?: boolean
}) {
  const Comp = asChild ? Slot.Root : "a"

  return (
    <Comp
      data-slot="sidebar-menu-sub-button"
      data-sidebar="menu-sub-button"
      data-size={size}
      data-active={isActive}
      className={cn(
        "text-sidebar-foreground ring-sidebar-ring hover:bg-sidebar-accent hover:text-sidebar-accent-foreground active:bg-sidebar-accent active:text-sidebar-accent-foreground [&>svg]:text-sidebar-accent-foreground data-active:bg-sidebar-accent data-active:text-sidebar-accent-foreground h-7 gap-2 rounded-md px-2 focus-visible:ring-1 data-[size=md]:text-sm data-[size=sm]:text-xs [&>svg]:size-4 flex min-w-0 -translate-x-px items-center overflow-hidden outline-hidden group-data-[collapsible=icon]:hidden disabled:pointer-events-none disabled:opacity-50 aria-disabled:pointer-events-none aria-disabled:opacity-50 [&>span:last-child]:truncate [&>svg]:shrink-0",
        className
      )}
      {...props}
    />
  )
}

export {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupAction,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInput,
  SidebarInset,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSkeleton,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarProvider,
  SidebarRail,
  SidebarResizeHandle,
  SidebarSeparator,
  SidebarTrigger,
  useSidebar,
  SIDEBAR_WIDTH,
  SIDEBAR_WIDTH_ICON,
}
