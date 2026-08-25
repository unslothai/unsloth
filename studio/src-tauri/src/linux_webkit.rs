// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

use std::ffi::{OsStr, OsString};
use std::os::unix::net::UnixStream;

const APPIMAGE: &str = "APPIMAGE";
const GLES_V2: &[u8] = b"libGLESv2.so.2\0";

const WAYLAND_DISPLAY: &str = "WAYLAND_DISPLAY";
const X11_DISPLAY: &str = "DISPLAY";
const WAYLAND_SOCKET: &str = "WAYLAND_SOCKET";
const X11_SOCKET_DIR: &str = "/tmp/.X11-unix";
const DEFAULT_WAYLAND_SOCKET: &str = "wayland-0";
const GDK_BACKEND: &str = "GDK_BACKEND";
const FORCE_SHARED_MEMORY: &str = "WEBKIT_DMABUF_RENDERER_FORCE_SHM";
const DISABLE_DMABUF: &str = "WEBKIT_DISABLE_DMABUF_RENDERER";
// disable-nvidia-dmabuf.patch's own opt-out, read inside isNVIDIA(). WebKit returns on
// DISABLE_DMABUF first, so it never gets there unless we honour it ourselves.
const FORCE_DMABUF: &str = "WEBKIT_FORCE_DMABUF_RENDERER";
// Comma-joined list of the variables we set, so a relaunch tells our own inherited output
// from an operator's value. Tauri's process::restart does not env_clear. WebKit never reads it.
const APPLIED_WORKAROUND: &str = "UNSLOTH_WEBKIT_RENDERER_WORKAROUND";
const FORCE_SHARED_MEMORY_MIN_VERSION: (u32, u32) = (2, 44);
// both the proprietary and open nvidia modules publish this; nouveau does not and is unaffected
const NVIDIA_DRIVER_VERSION_PATH: &str = "/proc/driver/nvidia/version";

const NVIDIA_REASON: &str = "NVIDIA driver loaded (no Wayland session)";
const NVIDIA_WAYLAND_REASON: &str = "NVIDIA driver loaded (Wayland session)";
const NVIDIA_APPIMAGE_GLES_REASON: &str =
    "NVIDIA driver loaded; AppImage without a usable GLES library";
const WAYLAND_REASON: &str = "Wayland session";
const APPIMAGE_GLES_REASON: &str = "AppImage without a usable GLES library";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderingWorkaround {
    ForceSharedMemory,
    /// Only for a host the probe confirmed is NVIDIA. On a patched library isNVIDIA()
    /// returns before mode.add(SharedMemory) in 2.50.4 and 2.52.6, so FORCE_SHM alone is
    /// never read there; FORCE_DMABUF stands that check down so selection reaches it.
    /// Unpatched libraries ignore the variable.
    ForceSharedMemoryOnNvidia,
    DisableDmabuf,
}

impl RenderingWorkaround {
    fn variables(self) -> &'static [&'static str] {
        match self {
            Self::ForceSharedMemory => &[FORCE_SHARED_MEMORY],
            Self::ForceSharedMemoryOnNvidia => &[FORCE_SHARED_MEMORY, FORCE_DMABUF],
            Self::DisableDmabuf => &[DISABLE_DMABUF],
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum RenderingPlan {
    /// The workaround to apply, and why, for the startup log.
    Apply(RenderingWorkaround, &'static str),
    PreserveEnvironment,
}

/// Resolve GDK_BACKEND the way gdk_display_manager_open_display does: g_strsplit on ',',
/// no trimming, g_str_equal so matching is exact and case sensitive, entries tried in
/// order, and the first backend whose display opens wins. '*' expands to the built-in
/// order, which is wayland before x11 on Linux. Membership is not selection: `x11,wayland`
/// runs on X11 whenever an X display opens, and on NVIDIA that decides between the
/// shared-memory switch and the empty transport set.
///
/// A display cannot be opened here, so `wayland_open` is the socket probe below standing in
/// for GDK's own opener. A named backend that cannot open is not the selection, it is
/// skipped, so wayland,x11 lands on X11 when no compositor answers.
fn selected_backend_is_wayland(backends: &OsStr, wayland_open: bool, x11_display: bool) -> bool {
    let mut named_wayland = false;
    for backend in backends.to_string_lossy().split(',') {
        match backend {
            "wayland" if wayland_open => return true,
            "wayland" => named_wayland = true,
            "x11" if x11_display => return false,
            "*" if wayland_open => return true,
            "*" if x11_display => return false,
            _ => continue,
        }
    }
    // Wayland was asked for and nothing else could open. GTK will fail to start if the
    // compositor really is absent, so the workaround choice is moot either way; keep the
    // Wayland one so a socket we cannot see from here is not treated as an X11 session.
    named_wayland
}

fn supports_force_shared_memory((major, minor, _micro): (u32, u32, u32)) -> bool {
    (major, minor) >= FORCE_SHARED_MEMORY_MIN_VERSION
}

/// `forceDMABuf && *forceDMABuf != '0'` from disable-nvidia-dmabuf.patch: first byte
/// only, so an empty value is a request and `0` is not.
fn force_dmabuf_requested(value: &OsStr) -> bool {
    value.as_encoded_bytes().first() != Some(&b'0')
}

fn rendering_plan(
    env: impl Fn(&str) -> Option<OsString>,
    webkit_version: (u32, u32, u32),
    gles_usable: bool,
    nvidia_driver_loaded: bool,
    wayland_socket: bool,
    x11_open: bool,
) -> RenderingPlan {
    // Either renderer variable is an operator override when present, `=0` and an empty
    // value included, unless the marker says we wrote it. Without that test a launch
    // reads its own inherited output back as an instruction and the first decision is
    // pinned for the life of the process tree, even after the display server changes.
    //
    // DISABLE_DMABUF=0 is how a host this over-triggers on gets its accelerated path
    // back: WebKit reads it as unset (`g_strcmp0(v, "0")`), and a patched library's own
    // GL_VENDOR probe then declines on an iGPU. An empty value is not that request.
    let applied_by_this_app = env(APPLIED_WORKAROUND).unwrap_or_default();
    let ours = |variable: &str| {
        applied_by_this_app
            .to_string_lossy()
            .split(',')
            .any(|claimed| claimed == variable)
    };

    if env(DISABLE_DMABUF).is_some() && !ours(DISABLE_DMABUF) {
        return RenderingPlan::PreserveEnvironment;
    }
    if env(FORCE_SHARED_MEMORY).is_some() && !ours(FORCE_SHARED_MEMORY) {
        return RenderingPlan::PreserveEnvironment;
    }

    // Stands the NVIDIA branch down and nothing else: it answers that patch's question,
    // not the missing-GLES one below, and that launch cannot render without its fallback.
    let force_dmabuf = !ours(FORCE_DMABUF)
        && env(FORCE_DMABUF)
            .as_deref()
            .is_some_and(force_dmabuf_requested);

    let is_appimage = env(APPIMAGE).is_some();
    // What matters is which backend opens, not what WAYLAND_DISPLAY and DISPLAY say: both
    // are inherited by clients of the other server and both outlive the thing they name.
    // Each flag is the answer from a probe of the socket that opener would use. An unset
    // GDK_BACKEND is GDK's own "*", which tries wayland before x11 (gdk_backends[]).
    let wayland_session = selected_backend_is_wayland(
        env(GDK_BACKEND).as_deref().unwrap_or(OsStr::new("*")),
        wayland_socket,
        x11_open,
    );

    // The DMA-BUF transport breaks on the proprietary driver on either display server, so
    // this cannot be gated on a Wayland session. Upstream declined the fix (bug 262607
    // WONTFIX, PR 18614 closed), so Debian and Ubuntu carry disable-nvidia-dmabuf.patch
    // and Fedora, Arch and the tarballs do not. Both shipped artifacts get a patched
    // library anyway (the .deb from the host, the AppImage bundles Ubuntu 22.04's), so
    // here this covers hosts where that patch's GL_VENDOR probe disagrees with the module
    // probe, plus the GBM open and throwaway GL context isNVIDIA() does at startup.
    //
    // The two switches are not interchangeable, so pick per failure mode, not per GPU:
    //   Wayland  DISABLE_DMABUF. The failure is the explicit-sync disconnect, and
    //            FORCE_SHM routes every commit down the wl_shm path that trips it
    //            (bug 315436). It is also the switch reported to fix Error 71.
    //   X11      FORCE_SHM. No explicit-sync protocol there, the failure is hardware
    //            DMA-BUF allocation, and shared memory fixes it without the empty set.
    // The empty set is not just slower: DISABLE_DMABUF returns before the SharedMemory
    // add, so checkRequirements() is false, AcceleratedBackingStore::create() returns
    // nullptr, and webkitWebViewBaseEnterAcceleratedCompositingMode() dereferences it
    // behind an ASSERT release builds drop. block/buzz#3654 hits that SIGSEGV on NVIDIA
    // X11, on the same iGPU-presenting topology the probe below over-triggers on.
    //
    // That probe is module presence, not the GPU that will render, so a PRIME laptop on
    // its iGPU takes a workaround it does not need. Deliberate: reading the rendering GPU
    // needs a GL context and this runs before GTK init so that none exists. Those hosts
    // opt out with WEBKIT_DISABLE_DMABUF_RENDERER=0.
    if nvidia_driver_loaded && !force_dmabuf {
        let missing_appimage_gles = is_appimage && !gles_usable;
        let reason = if missing_appimage_gles {
            NVIDIA_APPIMAGE_GLES_REASON
        } else if wayland_session {
            NVIDIA_WAYLAND_REASON
        } else {
            NVIDIA_REASON
        };
        // FORCE_SHM still reaches the failing libepoxy path in AppImages, and 2.44 is
        // where WebKitGTK started reading it at all, so both keep the stronger switch.
        let workaround = if wayland_session
            || missing_appimage_gles
            || !supports_force_shared_memory(webkit_version)
        {
            RenderingWorkaround::DisableDmabuf
        } else {
            RenderingWorkaround::ForceSharedMemoryOnNvidia
        };
        return RenderingPlan::Apply(workaround, reason);
    }

    // libepoxy loads GLES at runtime; disabling DMA-BUF handles a missing copy (#8343).
    if is_appimage && !gles_usable {
        return RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, APPIMAGE_GLES_REASON);
    }

    if !wayland_session {
        return RenderingPlan::PreserveEnvironment;
    }

    let workaround = if is_appimage {
        // FORCE_SHM still reaches the failing libepoxy path in AppImages.
        RenderingWorkaround::DisableDmabuf
    } else if supports_force_shared_memory(webkit_version) {
        // Plain FORCE_SHM: nothing here confirmed NVIDIA, so leave the library's own
        // detection alone. Standing it down would force wl_shm on a host it would have
        // taken off DMA-BUF itself, which is the commit path bug 315436 disconnects on.
        RenderingWorkaround::ForceSharedMemory
    } else {
        // FORCE_SHM was added with WebKitGTK 2.44. Older host libraries ignore
        // it, so use the renderer-disable workaround supported by 2.42.
        RenderingWorkaround::DisableDmabuf
    };
    RenderingPlan::Apply(workaround, WAYLAND_REASON)
}

/// Whether the socket wl_display_connect would use accepts connections. Existence is not
/// enough and a set WAYLAND_DISPLAY is not either: a dead compositor leaves both behind,
/// and GDK's opener would then fail over to x11 while this said Wayland. Connecting fails
/// the same way GDK does, and touches no GL state, so it is safe before GTK init.
///
/// Same resolution as libwayland: WAYLAND_SOCKET is an inherited fd and settles it on its
/// own, an absolute WAYLAND_DISPLAY is the path (1.15+), otherwise it is a name under
/// XDG_RUNTIME_DIR, defaulting to wayland-0. The fallback is on unset, not on empty, so
/// `WAYLAND_DISPLAY=` resolves to the runtime directory itself and cannot connect. That is
/// how a session forces itself through XWayland, and it must not read as Wayland.
fn wayland_socket_connectable(
    runtime_dir: Option<OsString>,
    display: Option<OsString>,
    inherited_fd: bool,
) -> bool {
    if inherited_fd {
        return true;
    }
    let name = display.unwrap_or_else(|| DEFAULT_WAYLAND_SOCKET.into());
    let path = if std::path::Path::new(&name).is_absolute() {
        std::path::PathBuf::from(name)
    } else {
        match runtime_dir {
            Some(dir) => std::path::Path::new(&dir).join(name),
            None => return false,
        }
    };
    UnixStream::connect(path).is_ok()
}

fn wayland_socket_present() -> bool {
    wayland_socket_connectable(
        std::env::var_os("XDG_RUNTIME_DIR"),
        std::env::var_os(WAYLAND_DISPLAY),
        // An empty value is not an fd; libwayland fails the parse rather than falling back.
        std::env::var_os(WAYLAND_SOCKET).is_some_and(|fd| !fd.is_empty()),
    )
}

/// Whether the X display DISPLAY names accepts connections, so the x11 arm is decided the
/// same way as the wayland one rather than on the variable alone. Deliberately fail-safe:
/// only a local `:N` display is probed, and only both of Xorg's listeners refusing counts
/// as closed. Guessing "closed" for a live server would put DISABLE_DMABUF on an X11
/// webview, which is the empty transport set, so anything unrecognized stays trusted.
fn x11_display_open(display: Option<OsString>, socket_dir: &str) -> bool {
    let Some(display) = display.filter(|display| !display.is_empty()) else {
        return false;
    };
    let text = display.to_string_lossy().into_owned();
    let Some((host, number)) = text.rsplit_once(':') else {
        return true;
    };
    let number = number.split('.').next().unwrap_or_default();
    if !matches!(host, "" | "unix")
        || number.is_empty()
        || !number.bytes().all(|b| b.is_ascii_digit())
    {
        return true; // a TCP or otherwise unusual display; not ours to judge
    }
    let path = format!("{socket_dir}/X{number}");
    if UnixStream::connect(&path).is_ok() {
        return true;
    }
    // Xorg binds the abstract name too, and on some hosts only that one is reachable.
    use std::os::linux::net::SocketAddrExt;
    std::os::unix::net::SocketAddr::from_abstract_name(path.as_bytes())
        .and_then(|address| UnixStream::connect_addr(&address))
        .is_ok()
}

fn nvidia_driver_loaded() -> bool {
    std::path::Path::new(NVIDIA_DRIVER_VERSION_PATH).exists()
}

fn gles_is_usable() -> bool {
    // Match libepoxy's runtime load, including unresolved dependencies.
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

/// Unset the variables an earlier launch of this process tree claimed and `drop` selects,
/// then retire the claim. Only claimed names are touched, never an operator's value.
fn release_claimed(drop: impl Fn(&str) -> bool) {
    let claimed = std::env::var_os(APPLIED_WORKAROUND).unwrap_or_default();
    for name in claimed.to_string_lossy().split(',') {
        if !name.is_empty() && drop(name) {
            std::env::remove_var(name);
        }
    }
    std::env::remove_var(APPLIED_WORKAROUND);
}

/// Select a compatible WebKitGTK rendering transport before GTK initialization.
///
/// Returns the variable applied by this launch and the reason for it, if any.
pub fn configure_renderer() -> Option<(&'static [&'static str], &'static str)> {
    let gles_usable = std::env::var_os(APPIMAGE).is_none() || gles_is_usable();
    match rendering_plan(
        |name| std::env::var_os(name),
        runtime_webkit_version(),
        gles_usable,
        nvidia_driver_loaded(),
        wayland_socket_present(),
        x11_display_open(std::env::var_os(X11_DISPLAY), X11_SOCKET_DIR),
    ) {
        RenderingPlan::Apply(workaround, reason) => {
            let variables = workaround.variables();
            release_claimed(|name| !variables.contains(&name));
            for variable in variables {
                std::env::set_var(variable, "1");
            }
            // Claim them, so the next launch re-decides rather than reading them as overrides.
            std::env::set_var(APPLIED_WORKAROUND, variables.join(","));
            Some((variables, reason))
        }
        RenderingPlan::PreserveEnvironment => {
            // Drop the values too, not just the claim. Leaving one set keeps a workaround
            // this launch decided against, and the next launch, seeing it unmarked, would
            // read it as an operator override and preserve it for good.
            release_claimed(|_| true);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODERN_WEBKIT: (u32, u32, u32) = (2, 44, 0);

    fn plan_on_host(
        vars: &[(&str, &str)],
        webkit_version: (u32, u32, u32),
        gles_usable: bool,
        nvidia_driver_loaded: bool,
    ) -> RenderingPlan {
        // A fixture that names a display means a live one, which is what the probe answers.
        let live = named(vars, WAYLAND_DISPLAY);
        plan_on_host_with_socket(
            vars,
            webkit_version,
            gles_usable,
            nvidia_driver_loaded,
            live,
        )
    }

    fn plan_on_host_with_socket(
        vars: &[(&str, &str)],
        webkit_version: (u32, u32, u32),
        gles_usable: bool,
        nvidia_driver_loaded: bool,
        wayland_socket: bool,
    ) -> RenderingPlan {
        // As above for DISPLAY: a fixture that names one models a server that answers.
        let x11_open = named(vars, X11_DISPLAY);
        plan_on_displays(
            vars,
            webkit_version,
            gles_usable,
            nvidia_driver_loaded,
            wayland_socket,
            x11_open,
        )
    }

    fn plan_on_displays(
        vars: &[(&str, &str)],
        webkit_version: (u32, u32, u32),
        gles_usable: bool,
        nvidia_driver_loaded: bool,
        wayland_socket: bool,
        x11_open: bool,
    ) -> RenderingPlan {
        rendering_plan(
            |name| {
                vars.iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| OsString::from(value))
            },
            webkit_version,
            gles_usable,
            nvidia_driver_loaded,
            wayland_socket,
            x11_open,
        )
    }

    fn named(vars: &[(&str, &str)], variable: &str) -> bool {
        vars.iter()
            .any(|(key, value)| *key == variable && !value.is_empty())
    }

    fn plan_with_version_and_gles(
        vars: &[(&str, &str)],
        webkit_version: (u32, u32, u32),
        gles_usable: bool,
    ) -> RenderingPlan {
        plan_on_host(vars, webkit_version, gles_usable, false)
    }

    fn plan_with_version(vars: &[(&str, &str)], webkit_version: (u32, u32, u32)) -> RenderingPlan {
        plan_with_version_and_gles(vars, webkit_version, true)
    }

    fn plan(vars: &[(&str, &str)]) -> RenderingPlan {
        plan_with_version(vars, MODERN_WEBKIT)
    }

    fn plan_on_nvidia(vars: &[(&str, &str)]) -> RenderingPlan {
        plan_on_host(vars, MODERN_WEBKIT, true, true)
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
    fn nvidia_on_x11_drops_to_shared_memory_rather_than_the_empty_transport_set() {
        // No explicit-sync protocol to violate here, and DISABLE_DMABUF leaves the null
        // backing store release builds dereference (block/buzz#3654).
        assert_eq!(
            plan_on_nvidia(&[]),
            RenderingPlan::Apply(
                RenderingWorkaround::ForceSharedMemoryOnNvidia,
                NVIDIA_REASON
            )
        );
    }

    #[test]
    fn nvidia_under_an_explicit_x11_backend_still_applies() {
        assert_eq!(
            plan_on_nvidia(&[(GDK_BACKEND, "x11")]),
            RenderingPlan::Apply(
                RenderingWorkaround::ForceSharedMemoryOnNvidia,
                NVIDIA_REASON
            )
        );
    }

    #[test]
    fn nvidia_on_x11_below_244_keeps_the_legacy_renderer_switch() {
        // FORCE_SHM arrived in 2.44; older libraries ignore it entirely.
        assert_eq!(
            plan_on_host(&[], (2, 42, 7), true, true),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_REASON)
        );
    }

    #[test]
    fn nvidia_appimage_without_gles_reports_both_causes() {
        assert_eq!(
            plan_on_host(
                &[(APPIMAGE, "/tmp/Unsloth.AppImage")],
                MODERN_WEBKIT,
                false,
                true
            ),
            RenderingPlan::Apply(
                RenderingWorkaround::DisableDmabuf,
                NVIDIA_APPIMAGE_GLES_REASON
            )
        );
    }

    #[test]
    fn the_distribution_force_dmabuf_opt_out_outranks_the_nvidia_default() {
        // Checked inside isNVIDIA(), which WebKit never reaches once DISABLE_DMABUF is set.
        for value in ["1", "", "yes"] {
            assert_eq!(
                plan_on_nvidia(&[(FORCE_DMABUF, value)]),
                RenderingPlan::PreserveEnvironment,
                "WEBKIT_FORCE_DMABUF_RENDERER={value} must stand the workaround down"
            );
        }
    }

    #[test]
    fn force_dmabuf_does_not_defeat_the_missing_gles_fallback() {
        // That fallback answers a packaging failure, not this variable's question, and the
        // AppImage cannot render without it (#8343).
        assert_eq!(
            plan_on_host(
                &[(FORCE_DMABUF, "1"), (APPIMAGE, "/tmp/Unsloth.AppImage")],
                MODERN_WEBKIT,
                false,
                true
            ),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, APPIMAGE_GLES_REASON)
        );
    }

    #[test]
    fn force_dmabuf_still_leaves_the_wayland_workaround_in_place() {
        // It speaks to the NVIDIA patch, not the Wayland rule that predates it.
        assert_eq!(
            plan_on_nvidia(&[(FORCE_DMABUF, "1"), (WAYLAND_DISPLAY, "wayland-0")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, WAYLAND_REASON)
        );
    }

    #[test]
    fn a_zero_force_dmabuf_value_does_not_stand_the_workaround_down() {
        // The patch tests the first byte against '0', so "0" is not a request.
        assert_eq!(
            plan_on_nvidia(&[(FORCE_DMABUF, "0")]),
            RenderingPlan::Apply(
                RenderingWorkaround::ForceSharedMemoryOnNvidia,
                NVIDIA_REASON
            )
        );
    }

    #[test]
    fn nvidia_on_wayland_takes_the_stronger_switch_over_shared_memory() {
        assert_eq!(
            plan_on_nvidia(&[(WAYLAND_DISPLAY, "wayland-0")]),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_WAYLAND_REASON)
        );
    }

    #[test]
    fn an_operator_force_shm_value_is_preserved_on_nvidia_too() {
        // Unmarked, so it came from a launcher or environment.d rather than from us.
        for value in ["0", "1", "", "true"] {
            assert_eq!(
                plan_on_nvidia(&[(FORCE_SHARED_MEMORY, value)]),
                RenderingPlan::PreserveEnvironment,
                "operator FORCE_SHM={value} must not be overridden on NVIDIA"
            );
        }
    }

    #[test]
    fn nvidia_relaunch_re_decides_over_its_own_inherited_force_shm() {
        // Marked, so it is our own state from an earlier launch, and Wayland escalates it.
        assert_eq!(
            plan_on_nvidia(&[
                (WAYLAND_DISPLAY, "wayland-0"),
                (FORCE_SHARED_MEMORY, "1"),
                (APPLIED_WORKAROUND, FORCE_SHARED_MEMORY),
            ]),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_WAYLAND_REASON)
        );
    }

    #[test]
    fn a_relaunch_re_reaches_the_same_decision_it_made_first_time() {
        // process::restart inherits the environment, so the plan must be a fixed point:
        // feed a launch's own output back in and it must not drift.
        let hosts = [
            (&[][..], true),
            (&[(WAYLAND_DISPLAY, "wayland-0")][..], true),
            (&[(GDK_BACKEND, "x11")][..], true),
            (&[][..], false),
            (&[(WAYLAND_DISPLAY, "wayland-0")][..], false),
        ];
        for (session, nvidia) in hosts {
            let first = plan_on_host(session, MODERN_WEBKIT, true, nvidia);
            let RenderingPlan::Apply(workaround, reason) = first else {
                continue;
            };
            let mut inherited = session.to_vec();
            let claimed = workaround.variables().join(",");
            for variable in workaround.variables() {
                inherited.push((variable, "1"));
            }
            inherited.push((APPLIED_WORKAROUND, &claimed));
            assert_eq!(
                plan_on_host(&inherited, MODERN_WEBKIT, true, nvidia),
                RenderingPlan::Apply(workaround, reason),
                "relaunch drifted for session {session:?} nvidia={nvidia}"
            );
        }
    }

    // release_claimed touches the real process environment, so serialise the tests that do.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn standing_down_clears_the_value_we_set_not_only_the_claim() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::set_var(DISABLE_DMABUF, "1");
        std::env::set_var(APPLIED_WORKAROUND, DISABLE_DMABUF);

        release_claimed(|_| true);

        // Leaving it set would keep a workaround this launch decided against, and the
        // next launch would read the unmarked value as an operator override for good.
        assert!(std::env::var_os(DISABLE_DMABUF).is_none());
        assert!(std::env::var_os(APPLIED_WORKAROUND).is_none());
    }

    #[test]
    fn switching_workaround_clears_only_the_variable_being_dropped() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::set_var(FORCE_SHARED_MEMORY, "1");
        std::env::set_var(FORCE_DMABUF, "1");
        std::env::set_var(
            APPLIED_WORKAROUND,
            [FORCE_SHARED_MEMORY, FORCE_DMABUF].join(","),
        );

        let keep = [FORCE_DMABUF];
        release_claimed(|name| !keep.contains(&name));

        assert!(std::env::var_os(FORCE_SHARED_MEMORY).is_none());
        assert_eq!(std::env::var_os(FORCE_DMABUF), Some(OsString::from("1")));
        std::env::remove_var(FORCE_DMABUF);
    }

    #[test]
    fn a_generic_shared_memory_plan_leaves_webkits_own_nvidia_detection_alone() {
        // Nothing here confirmed NVIDIA: /proc/driver/nvidia can be hidden by a container
        // or a confined launch while WebKit's GL probe still sees it. Standing the patch
        // down there would force the wl_shm commit path bug 315436 disconnects on.
        assert_eq!(
            plan(&[(WAYLAND_DISPLAY, "wayland-0")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, WAYLAND_REASON)
        );
        assert_eq!(
            RenderingWorkaround::ForceSharedMemory.variables(),
            &[FORCE_SHARED_MEMORY]
        );
    }

    #[test]
    fn a_named_wayland_that_cannot_open_falls_through_to_the_next_backend() {
        // GDK tries wayland first, its opener fails with no socket to connect to, and the
        // loop continues to x11. Calling that a Wayland session hands an X11 host the
        // empty transport set.
        assert_eq!(
            plan_on_nvidia(&[(GDK_BACKEND, "wayland,x11"), (X11_DISPLAY, ":0")]),
            RenderingPlan::Apply(
                RenderingWorkaround::ForceSharedMemoryOnNvidia,
                NVIDIA_REASON
            )
        );
        // Same list, but the default wayland-0 socket is there, so wayland does open.
        assert_eq!(
            plan_on_host_with_socket(
                &[(GDK_BACKEND, "wayland,x11"), (X11_DISPLAY, ":0")],
                MODERN_WEBKIT,
                true,
                true,
                true
            ),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_WAYLAND_REASON)
        );
    }

    #[test]
    fn a_lone_named_wayland_still_takes_the_wayland_workaround() {
        // No fallback entry, so GTK either finds a compositor or fails to start. Treating
        // it as X11 would be a guess against the only backend the operator named.
        assert_eq!(
            plan_on_nvidia(&[(GDK_BACKEND, "wayland")]),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_WAYLAND_REASON)
        );
    }

    #[test]
    fn forcing_shared_memory_also_opts_out_of_the_distro_patch() {
        // isNVIDIA() returns before mode.add(SharedMemory) on 2.50.4 and 2.52.6, so
        // FORCE_SHM on its own is never read there and the set is empty regardless.
        assert_eq!(
            RenderingWorkaround::ForceSharedMemoryOnNvidia.variables(),
            &[FORCE_SHARED_MEMORY, FORCE_DMABUF]
        );
        assert_eq!(
            RenderingWorkaround::DisableDmabuf.variables(),
            &[DISABLE_DMABUF]
        );
    }

    #[test]
    fn our_own_force_dmabuf_does_not_read_back_as_an_opt_out() {
        // We set it as half of ForceSharedMemory; a relaunch must not see its own output
        // as an operator standing the NVIDIA branch down.
        let claimed = [FORCE_SHARED_MEMORY, FORCE_DMABUF].join(",");
        assert_eq!(
            plan_on_nvidia(&[
                (FORCE_SHARED_MEMORY, "1"),
                (FORCE_DMABUF, "1"),
                (APPLIED_WORKAROUND, &claimed),
            ]),
            RenderingPlan::Apply(
                RenderingWorkaround::ForceSharedMemoryOnNvidia,
                NVIDIA_REASON
            )
        );
    }

    #[test]
    fn a_marker_for_the_other_variable_does_not_release_force_shm() {
        // Only a claim naming FORCE_SHM identifies FORCE_SHM as ours.
        assert_eq!(
            plan_on_nvidia(&[
                (FORCE_SHARED_MEMORY, "1"),
                (APPLIED_WORKAROUND, DISABLE_DMABUF),
            ]),
            RenderingPlan::PreserveEnvironment
        );
    }

    #[test]
    fn explicit_disable_override_preserves_force_shm_on_nvidia() {
        assert_eq!(
            plan_on_nvidia(&[(FORCE_SHARED_MEMORY, "1"), (DISABLE_DMABUF, "0")]),
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
    fn a_gdk_wildcard_selects_wayland_only_when_a_wayland_display_is_there() {
        // '*' expands to the built-in order, wayland before x11, and takes the first that
        // opens. With no WAYLAND_DISPLAY the wayland opener has nothing to connect to.
        assert_eq!(
            plan(&[(GDK_BACKEND, "*"), (WAYLAND_DISPLAY, "wayland-0")]),
            RenderingPlan::Apply(RenderingWorkaround::ForceSharedMemory, WAYLAND_REASON)
        );
        assert_eq!(
            plan(&[(GDK_BACKEND, "*"), (X11_DISPLAY, ":0")]),
            RenderingPlan::PreserveEnvironment
        );
    }

    #[test]
    fn an_ordered_backend_list_takes_the_first_that_opens() {
        // GDK tries entries in order, so x11,wayland runs on X11 whenever DISPLAY opens.
        // Membership said Wayland, which on NVIDIA picked the empty transport set for a
        // session that is actually X11.
        assert_eq!(
            plan_on_nvidia(&[
                (GDK_BACKEND, "x11,wayland"),
                (WAYLAND_DISPLAY, "wayland-0"),
                (X11_DISPLAY, ":0"),
            ]),
            RenderingPlan::Apply(
                RenderingWorkaround::ForceSharedMemoryOnNvidia,
                NVIDIA_REASON
            )
        );
        // Reverse the order and Wayland is what opens first.
        assert_eq!(
            plan_on_nvidia(&[
                (GDK_BACKEND, "wayland,x11"),
                (WAYLAND_DISPLAY, "wayland-0"),
                (X11_DISPLAY, ":0"),
            ]),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_WAYLAND_REASON)
        );
        // x11 named first but no X display: GDK falls through to the next entry.
        assert_eq!(
            plan_on_nvidia(&[(GDK_BACKEND, "x11,wayland"), (WAYLAND_DISPLAY, "wayland-0")]),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_WAYLAND_REASON)
        );
    }

    #[test]
    fn backend_matching_is_exact_like_g_str_equal() {
        // g_strsplit does not trim and g_str_equal is case sensitive, so neither of these
        // names a backend GDK will match; it falls through to the next entry.
        for value in [" wayland", "Wayland", "wayland-egl"] {
            assert_eq!(
                plan(&[(GDK_BACKEND, value), (WAYLAND_DISPLAY, "wayland-0")]),
                RenderingPlan::PreserveEnvironment,
                "GDK_BACKEND={value} matches no GDK backend"
            );
        }
    }

    #[test]
    fn appimage_without_usable_gles_disables_dmabuf_on_x11() {
        assert_eq!(
            plan_with_version_and_gles(
                &[(APPIMAGE, "/tmp/Unsloth.AppImage")],
                MODERN_WEBKIT,
                false
            ),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, APPIMAGE_GLES_REASON)
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
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, WAYLAND_REASON)
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

    // GDK_BACKEND unset is "*", so a live default socket opens wayland even with no
    // WAYLAND_DISPLAY, and this must not be read as the X11 session FORCE_SHM is for.
    #[test]
    fn an_unset_gdk_backend_follows_the_default_socket() {
        assert_eq!(
            plan_on_host_with_socket(&[(X11_DISPLAY, ":0")], MODERN_WEBKIT, true, true, true),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_WAYLAND_REASON)
        );
        assert_eq!(
            plan_on_host_with_socket(&[(X11_DISPLAY, ":0")], MODERN_WEBKIT, true, true, false),
            RenderingPlan::Apply(
                RenderingWorkaround::ForceSharedMemoryOnNvidia,
                NVIDIA_REASON
            )
        );
    }

    fn socket_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("unsloth-wl-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_listening_wayland_socket_is_a_wayland_session() {
        let dir = socket_dir("live");
        let _listener =
            std::os::unix::net::UnixListener::bind(dir.join(DEFAULT_WAYLAND_SOCKET)).unwrap();
        assert!(wayland_socket_connectable(
            Some(dir.clone().into()),
            None,
            false
        ));
        // The same socket named explicitly, and named by absolute path as libwayland allows.
        assert!(wayland_socket_connectable(
            Some(dir.clone().into()),
            Some(DEFAULT_WAYLAND_SOCKET.into()),
            false
        ));
        assert!(wayland_socket_connectable(
            None,
            Some(dir.join(DEFAULT_WAYLAND_SOCKET).into()),
            false
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // A crashed compositor leaves the file behind, and WAYLAND_DISPLAY behind with it;
    // GDK's opener still fails over to x11, so neither may read as a Wayland session.
    #[test]
    fn a_stale_wayland_socket_is_not_a_wayland_session() {
        let dir = socket_dir("stale");
        let socket = dir.join(DEFAULT_WAYLAND_SOCKET);
        std::os::unix::net::UnixListener::bind(&socket).unwrap();
        assert!(socket.exists());
        for display in [None, Some(OsString::from(DEFAULT_WAYLAND_SOCKET))] {
            assert!(!wayland_socket_connectable(
                Some(dir.clone().into()),
                display,
                false
            ));
        }
        assert!(!wayland_socket_connectable(None, None, false));
        // A name with no runtime dir to resolve it against cannot open either.
        assert!(!wayland_socket_connectable(
            None,
            Some(DEFAULT_WAYLAND_SOCKET.into()),
            false
        ));
        // WAYLAND_SOCKET is an inherited fd: the connection is already made.
        assert!(wayland_socket_connectable(None, None, true));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // WAYLAND_DISPLAY= is how a session forces itself through XWayland. libwayland falls
    // back to wayland-0 on unset, not on empty, so the empty value must not reach the
    // still-live default socket and read as Wayland.
    #[test]
    fn an_explicitly_empty_wayland_display_is_not_the_default_socket() {
        let dir = socket_dir("empty-display");
        let _listener =
            std::os::unix::net::UnixListener::bind(dir.join(DEFAULT_WAYLAND_SOCKET)).unwrap();
        assert!(wayland_socket_connectable(
            Some(dir.clone().into()),
            None,
            false
        ));
        assert!(!wayland_socket_connectable(
            Some(dir.clone().into()),
            Some(OsString::new()),
            false
        ));
        // Nor is an empty WAYLAND_SOCKET an inherited fd.
        assert!(!wayland_socket_connectable(
            None,
            Some(OsString::new()),
            false
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // DISPLAY names a server that may be gone; only a local display is judged, and only
    // when both of Xorg's listeners refuse, since guessing closed for a live server would
    // put DISABLE_DMABUF on an X11 webview.
    #[test]
    fn only_a_local_x_display_that_answers_nothing_reads_as_closed() {
        let dir = socket_dir("x11");
        let sockets = dir.to_string_lossy().into_owned();
        let probe = |display: &str| x11_display_open(Some(display.into()), &sockets);

        assert!(!x11_display_open(None, &sockets));
        assert!(!x11_display_open(Some(OsString::new()), &sockets));
        // Nothing has bound :0 in this directory yet.
        assert!(!probe(":0"));
        let _listener = std::os::unix::net::UnixListener::bind(dir.join("X0")).unwrap();
        assert!(probe(":0"));
        assert!(probe(":0.1"), "the screen suffix is not part of the socket");
        // Abstract only, which is all some hosts publish.
        use std::os::linux::net::SocketAddrExt;
        let abstract_name =
            std::os::unix::net::SocketAddr::from_abstract_name(format!("{sockets}/X1")).unwrap();
        let _abstract = std::os::unix::net::UnixListener::bind_addr(&abstract_name).unwrap();
        assert!(probe(":1"));
        assert!(!probe(":2"));
        // Not ours to judge: a remote display, a screen number that is not one, a path.
        assert!(probe("somehost:2"));
        assert!(probe(":abc"));
        assert!(probe("/tmp/.X11-unix/X2"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // A dead X server with an explicit x11-first order: GDK skips the opener that fails and
    // lands on wayland, so the plan has to land there too.
    #[test]
    fn a_dead_x_server_lets_an_x11_first_order_fall_through_to_wayland() {
        let session = &[(GDK_BACKEND, "x11,wayland"), (X11_DISPLAY, ":0")];
        assert_eq!(
            plan_on_displays(session, MODERN_WEBKIT, true, true, true, false),
            RenderingPlan::Apply(RenderingWorkaround::DisableDmabuf, NVIDIA_WAYLAND_REASON)
        );
        assert_eq!(
            plan_on_displays(session, MODERN_WEBKIT, true, true, true, true),
            RenderingPlan::Apply(
                RenderingWorkaround::ForceSharedMemoryOnNvidia,
                NVIDIA_REASON
            )
        );
    }

    // The X11 arm of the same failure: WAYLAND_DISPLAY outlives its compositor, DISPLAY
    // works, and DISABLE_DMABUF on an X11 webview is the empty-transport-set crash.
    #[test]
    fn a_dead_compositor_named_by_wayland_display_falls_back_to_x11() {
        let session = &[(WAYLAND_DISPLAY, "wayland-0"), (X11_DISPLAY, ":0")];
        assert_eq!(
            plan_on_host_with_socket(session, MODERN_WEBKIT, true, true, false),
            RenderingPlan::Apply(
                RenderingWorkaround::ForceSharedMemoryOnNvidia,
                NVIDIA_REASON
            )
        );
        assert_eq!(
            plan_on_host_with_socket(session, MODERN_WEBKIT, true, false, false),
            RenderingPlan::PreserveEnvironment
        );
    }
}
