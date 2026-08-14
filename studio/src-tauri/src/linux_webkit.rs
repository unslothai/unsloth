// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

use std::ffi::{OsStr, OsString};

const WAYLAND_DISPLAY: &str = "WAYLAND_DISPLAY";
const GDK_BACKEND: &str = "GDK_BACKEND";
const FORCE_SHARED_MEMORY: &str = "WEBKIT_DMABUF_RENDERER_FORCE_SHM";
const DISABLE_DMABUF: &str = "WEBKIT_DISABLE_DMABUF_RENDERER";
const FORCE_SHARED_MEMORY_MIN_VERSION: (u32, u32) = (2, 44);
// both the proprietary and open nvidia modules publish this; nouveau does not and is unaffected
const NVIDIA_DRIVER_VERSION_PATH: &str = "/proc/driver/nvidia/version";

const NVIDIA_REASON: &str = "NVIDIA driver loaded";
const WAYLAND_REASON: &str = "Wayland session";

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
    /// The workaround to apply, and why, for the startup log.
    Apply(RenderingWorkaround, &'static str),
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
    nvidia_driver_loaded: bool,
) -> RenderingPlan {
    // Presence is an operator override, including `=0` and an empty value. Do
    // not combine the modern shared-memory switch with legacy instructions a
    // user may already carry in a launcher or environment.d file.
    if env(FORCE_SHARED_MEMORY).is_some() || env(DISABLE_DMABUF).is_some() {
        return RenderingPlan::PreserveEnvironment;
    }

    // webkit's isNVIDIA fix (bug 262607) is session-independent, so x11 needs this too
    if nvidia_driver_loaded {
        return RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_REASON);
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

    let workaround = if supports_force_shared_memory(webkit_version) {
        RenderingWorkaround::ForceSharedMemory
    } else {
        // FORCE_SHM was added with WebKitGTK 2.44. Older host libraries ignore
        // it, so use the renderer-disable workaround supported by 2.42.
        RenderingWorkaround::DisableDmabuf
    };
    RenderingPlan::Apply(workaround, WAYLAND_REASON)
}

fn nvidia_driver_loaded() -> bool {
    std::path::Path::new(NVIDIA_DRIVER_VERSION_PATH).exists()
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
/// Returns the variable applied by this launch and the reason for it, if any.
pub fn configure_linux_renderer() -> Option<(&'static str, &'static str)> {
    match rendering_plan(
        |name| std::env::var_os(name),
        runtime_webkit_version(),
        nvidia_driver_loaded(),
    ) {
        RenderingPlan::Apply(workaround, reason) => {
            let variable = workaround.variable();
            std::env::set_var(variable, "1");
            Some((variable, reason))
        }
        RenderingPlan::PreserveEnvironment => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODERN_WEBKIT: (u32, u32, u32) = (2, 44, 0);

    fn plan_on_host(
        vars: &[(&str, &str)],
        webkit_version: (u32, u32, u32),
        nvidia_driver_loaded: bool,
    ) -> RenderingPlan {
        rendering_plan(
            |name| {
                vars.iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| OsString::from(value))
            },
            webkit_version,
            nvidia_driver_loaded,
        )
    }

    fn plan_with_version(vars: &[(&str, &str)], webkit_version: (u32, u32, u32)) -> RenderingPlan {
        plan_on_host(vars, webkit_version, false)
    }

    fn plan(vars: &[(&str, &str)]) -> RenderingPlan {
        plan_with_version(vars, MODERN_WEBKIT)
    }

    fn plan_on_nvidia(vars: &[(&str, &str)]) -> RenderingPlan {
        plan_on_host(vars, MODERN_WEBKIT, true)
    }

    #[test]
    fn wayland_uses_shared_memory_transport() {
        assert_eq!(
            plan(&[(WAYLAND_DISPLAY, "wayland-0")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, WAYLAND_REASON)
        );
    }

    #[test]
    fn explicit_wayland_without_a_display_variable_still_applies() {
        assert_eq!(
            plan(&[(GDK_BACKEND, "wayland")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, WAYLAND_REASON)
        );
    }

    #[test]
    fn an_x11_session_off_nvidia_keeps_webkit_defaults() {
        assert_eq!(plan(&[]), RenderingPlan::PreserveEnvironment);
    }

    #[test]
    fn nvidia_on_x11_disables_the_dmabuf_renderer() {
        assert_eq!(
            plan_on_nvidia(&[]),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_REASON)
        );
    }

    #[test]
    fn nvidia_under_an_explicit_x11_backend_still_applies() {
        assert_eq!(
            plan_on_nvidia(&[(GDK_BACKEND, "x11")]),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_REASON)
        );
    }

    #[test]
    fn nvidia_on_wayland_takes_the_stronger_switch_over_shared_memory() {
        // isNVIDIA returns before SharedMemory is added, so FORCE_SHM is not equivalent
        assert_eq!(
            plan_on_nvidia(&[(WAYLAND_DISPLAY, "wayland-0")]),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_REASON)
        );
    }

    #[test]
    fn an_operator_override_outranks_the_nvidia_default() {
        assert_eq!(
            plan_on_nvidia(&[(FORCE_SHARED_MEMORY, "0")]),
            RenderingPlan::PreserveEnvironment
        );
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
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, WAYLAND_REASON)
        );
    }

    #[test]
    fn old_webkit_uses_the_legacy_renderer_switch() {
        assert_eq!(
            plan_with_version(&[(WAYLAND_DISPLAY, "wayland-0")], (2, 42, 7)),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, WAYLAND_REASON)
        );
    }

    #[test]
    fn webkit_244_uses_force_shm() {
        assert_eq!(
            plan_with_version(&[(WAYLAND_DISPLAY, "wayland-0")], (2, 44, 0)),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, WAYLAND_REASON)
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
