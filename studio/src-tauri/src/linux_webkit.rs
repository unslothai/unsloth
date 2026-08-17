// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

use std::ffi::{OsStr, OsString};

const APPIMAGE: &str = "APPIMAGE";
const GLES_V2: &[u8] = b"libGLESv2.so.2\0";

const WAYLAND_DISPLAY: &str = "WAYLAND_DISPLAY";
const GDK_BACKEND: &str = "GDK_BACKEND";
const FORCE_SHARED_MEMORY: &str = "WEBKIT_DMABUF_RENDERER_FORCE_SHM";
const DISABLE_DMABUF: &str = "WEBKIT_DISABLE_DMABUF_RENDERER";
const FORCE_SHARED_MEMORY_MIN_VERSION: (u32, u32) = (2, 44);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderingWorkaround {
    ForceSharedMemory,
    DisableDmabuf,
}

impl RenderingWorkaround {
    fn variable(self) -> &'static str {
        match self {
            Self::ForceSharedMemory => FORCE_SHARED_MEMORY,
            Self::DisableDmabuf => DISABLE_DMABUF,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum RenderingPlan {
    Apply(RenderingWorkaround),
    PreserveEnvironment,
}

fn backend_allows_wayland(backends: &OsStr) -> bool {
    backends.to_string_lossy().split(',').any(|backend| {
        let backend = backend.trim();
        backend == "*" || backend.eq_ignore_ascii_case("wayland")
    })
}

fn supports_force_shared_memory((major, minor, _micro): (u32, u32, u32)) -> bool {
    (major, minor) >= FORCE_SHARED_MEMORY_MIN_VERSION
}

fn rendering_plan(
    env: impl Fn(&str) -> Option<OsString>,
    webkit_version: (u32, u32, u32),
    gles_usable: bool,
) -> RenderingPlan {
    // Presence is an operator override, including `=0` and an empty value. Do
    // not combine the modern shared-memory switch with legacy instructions a
    // user may already carry in a launcher or environment.d file.
    if env(FORCE_SHARED_MEMORY).is_some() || env(DISABLE_DMABUF).is_some() {
        return RenderingPlan::PreserveEnvironment;
    }

    let is_appimage = env(APPIMAGE).is_some();
    // libepoxy opens GLES by soname after startup, so it is invisible to the
    // executable's dependency check. Without a usable host copy WebKit aborts;
    // disabling DMA-BUF is the only tested path that still renders (#8343).
    if is_appimage && !gles_usable {
        return RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf);
    }

    let configured_backends = env(GDK_BACKEND);
    let explicit_wayland = configured_backends
        .as_deref()
        .map(backend_allows_wayland)
        .unwrap_or(false);
    let wayland_display = env(WAYLAND_DISPLAY)
        .map(|display| !display.is_empty())
        .unwrap_or(false);

    // GDK can connect to the default wayland-0 socket when its backend is
    // explicitly Wayland even if WAYLAND_DISPLAY is absent. Conversely,
    // WAYLAND_DISPLAY is inherited by XWayland apps, so an X11-only GDK_BACKEND
    // must take precedence.
    if configured_backends.is_some() && !explicit_wayland {
        return RenderingPlan::PreserveEnvironment;
    }
    if !wayland_display && !explicit_wayland {
        return RenderingPlan::PreserveEnvironment;
    }

    let workaround = if is_appimage {
        // FORCE_SHM keeps the web process alive on the affected Steam Deck but
        // still reaches the failing epoxy path. The complete disable rendered
        // there; retain FORCE_SHM for native packages where it is sufficient.
        RenderingWorkaround::DisableDmabuf
    } else if supports_force_shared_memory(webkit_version) {
        RenderingWorkaround::ForceSharedMemory
    } else {
        // FORCE_SHM was added with WebKitGTK 2.44. Older host libraries ignore
        // it, so use the renderer-disable workaround supported by 2.42.
        RenderingWorkaround::DisableDmabuf
    };
    RenderingPlan::Apply(workaround)
}

fn gles_is_usable() -> bool {
    // Test the same operation libepoxy performs instead of only looking for a
    // filename: a present object with an unresolved dependency is equally unusable.
    unsafe {
        let handle = libc::dlopen(GLES_V2.as_ptr().cast(), libc::RTLD_LAZY | libc::RTLD_LOCAL);
        if handle.is_null() {
            false
        } else {
            libc::dlclose(handle);
            true
        }
    }
}

fn runtime_webkit_version() -> (u32, u32, u32) {
    // These accessors read compile-time version constants from the already
    // linked WebKitGTK library; they do not initialize GTK or a web view.
    unsafe {
        (
            webkit2gtk_sys::webkit_get_major_version(),
            webkit2gtk_sys::webkit_get_minor_version(),
            webkit2gtk_sys::webkit_get_micro_version(),
        )
    }
}

/// Select a compatible WebKitGTK rendering transport before GTK initialization.
///
/// Returns the variable applied by this launch, if any.
pub fn configure_renderer() -> Option<&'static str> {
    let gles_usable = std::env::var_os(APPIMAGE).is_none() || gles_is_usable();
    match rendering_plan(
        |name| std::env::var_os(name),
        runtime_webkit_version(),
        gles_usable,
    ) {
        RenderingPlan::Apply(workaround) => {
            let variable = workaround.variable();
            std::env::set_var(variable, "1");
            Some(variable)
        }
        RenderingPlan::PreserveEnvironment => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODERN_WEBKIT: (u32, u32, u32) = (2, 44, 0);

    fn plan_with_version_and_gles(
        vars: &[(&str, &str)],
        webkit_version: (u32, u32, u32),
        gles_usable: bool,
    ) -> RenderingPlan {
        rendering_plan(
            |name| {
                vars.iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| OsString::from(value))
            },
            webkit_version,
            gles_usable,
        )
    }

    fn plan_with_version(vars: &[(&str, &str)], webkit_version: (u32, u32, u32)) -> RenderingPlan {
        plan_with_version_and_gles(vars, webkit_version, true)
    }

    fn plan(vars: &[(&str, &str)]) -> RenderingPlan {
        plan_with_version(vars, MODERN_WEBKIT)
    }

    #[test]
    fn wayland_uses_shared_memory_transport() {
        assert_eq!(
            plan(&[(WAYLAND_DISPLAY, "wayland-0")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory)
        );
    }

    #[test]
    fn explicit_wayland_without_a_display_variable_still_applies() {
        assert_eq!(
            plan(&[(GDK_BACKEND, "wayland")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory)
        );
    }

    #[test]
    fn x11_session_keeps_webkit_defaults() {
        assert_eq!(plan(&[]), RenderingPlan::PreserveEnvironment);
    }

    #[test]
    fn an_empty_wayland_display_is_not_a_wayland_session() {
        assert_eq!(
            plan(&[(WAYLAND_DISPLAY, "")]),
            RenderingPlan::PreserveEnvironment
        );
    }

    #[test]
    fn an_explicit_x11_gdk_backend_wins() {
        assert_eq!(
            plan(&[(WAYLAND_DISPLAY, "wayland-0"), (GDK_BACKEND, "x11")]),
            RenderingPlan::PreserveEnvironment
        );
    }

    #[test]
    fn a_gdk_wildcard_can_select_wayland_without_a_display_variable() {
        assert_eq!(
            plan(&[(GDK_BACKEND, "*")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory)
        );
    }

    #[test]
    fn appimage_without_usable_gles_disables_dmabuf_on_x11() {
        assert_eq!(
            plan_with_version_and_gles(
                &[(APPIMAGE, "/tmp/Unsloth.AppImage")],
                MODERN_WEBKIT,
                false
            ),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf)
        );
    }

    #[test]
    fn native_package_without_gles_keeps_existing_x11_behavior() {
        assert_eq!(
            plan_with_version_and_gles(&[], MODERN_WEBKIT, false),
            RenderingPlan::PreserveEnvironment
        );
    }

    #[test]
    fn appimage_wayland_disables_dmabuf_completely() {
        assert_eq!(
            plan(&[
                (APPIMAGE, "/tmp/Unsloth.AppImage"),
                (WAYLAND_DISPLAY, "wayland-0")
            ]),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf)
        );
    }

    #[test]
    fn old_webkit_uses_the_legacy_renderer_switch() {
        assert_eq!(
            plan_with_version(&[(WAYLAND_DISPLAY, "wayland-0")], (2, 42, 7)),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf)
        );
    }

    #[test]
    fn webkit_244_uses_force_shm() {
        assert_eq!(
            plan_with_version(&[(WAYLAND_DISPLAY, "wayland-0")], (2, 44, 0)),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory)
        );
    }

    #[test]
    fn an_explicit_force_shm_value_is_preserved() {
        assert_eq!(
            plan(&[(WAYLAND_DISPLAY, "wayland-0"), (FORCE_SHARED_MEMORY, "0")]),
            RenderingPlan::PreserveEnvironment
        );
    }

    #[test]
    fn the_legacy_dmabuf_override_is_preserved() {
        assert_eq!(
            plan(&[(WAYLAND_DISPLAY, "wayland-0"), (DISABLE_DMABUF, "1")]),
            RenderingPlan::PreserveEnvironment
        );
    }
}
