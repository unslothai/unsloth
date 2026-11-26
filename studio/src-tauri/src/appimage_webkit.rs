// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Which WebKitGTK an AppImage runs on. `linux/appimage-apprun.sh` probes the host's
//! GTK3/webkit2gtk-4.1 stack with [`PROBE_FLAG`] and, when it is complete and new enough,
//! starts the app on it instead of the bundled copy, exporting [`MODE_ENV`]`=host`.

use std::ffi::OsString;
#[cfg(target_os = "linux")]
use std::path::{Path, PathBuf};

/// Run by AppRun on the unbundled copy of this binary: report the WebKitGTK that loaded.
pub const PROBE_FLAG: &str = "--unsloth-webkit-probe";
/// Set by AppRun to `host` or `bundled`.
pub const MODE_ENV: &str = "UNSLOTH_APPIMAGE_WEBKIT_MODE";
const APPIMAGE: &str = "APPIMAGE";
/// No tray library: setup_tray would panic (libappindicator-sys dlopens one at startup).
#[cfg(target_os = "linux")]
const EXIT_NO_APPINDICATOR: i32 = 3;
/// libappindicator-sys's own order.
#[cfg(target_os = "linux")]
const APPINDICATOR_LIBRARIES: [&[u8]; 2] = [
    b"libayatana-appindicator3.so.1\0",
    b"libappindicator3.so.1\0",
];
/// No `autoaudiosink` (gst-plugins-good): WebKitGTK aborts the page on its first media element.
#[cfg(target_os = "linux")]
const EXIT_NO_AUDIO_SINK: i32 = 4;
#[cfg(target_os = "linux")]
const AUDIO_SINK_PLUGIN: &str = "libgstautodetect.so";
/// The plugins a generated mp4 (H.264 video, AAC audio) needs, with the Debian package of each:
/// `avdec_h264` / `avdec_aac` (gst-libav) and the mp4 demuxer `qtdemux` (gst-plugins-good).
/// Without them WebKitGTK shows the video as unplayable.
#[cfg(target_os = "linux")]
const MP4_PLAYBACK_PLUGINS: [(&str, &str); 2] = [
    ("libgstlibav.so", "gstreamer1.0-libav"),
    ("libgstisomp4.so", "gstreamer1.0-plugins-good"),
];

/// True when this AppImage runs on its own bundled libraries. A host-mode AppImage loads the
/// system's stack, so renderer decisions treat it like a native install.
pub fn bundled_runtime(env: impl Fn(&str) -> Option<OsString>) -> bool {
    env(APPIMAGE).is_some() && env(MODE_ENV).as_deref() != Some("host".as_ref())
}

/// For diagnostics: never echoes the variable, so only these three values can appear.
pub fn mode_label(env: impl Fn(&str) -> Option<OsString>) -> &'static str {
    if env(APPIMAGE).is_none() {
        "none"
    } else if bundled_runtime(env) {
        "bundled"
    } else {
        "host"
    }
}

/// The probe: `Some(exit code)` when argv is exactly `[exe, PROBE_FLAG]`. Runs before any
/// GTK or logging. The process only starts when every linked library and symbol resolved
/// (the binary is linked BIND_NOW), so reaching this code already proves the stack loads.
#[cfg(target_os = "linux")]
pub fn run_probe() -> Option<i32> {
    let mut args = std::env::args_os().skip(1);
    if args.next().as_deref() != Some(PROBE_FLAG.as_ref()) || args.next().is_some() {
        return None;
    }
    if !APPINDICATOR_LIBRARIES
        .iter()
        .any(|name| library_loads(name))
    {
        return Some(EXIT_NO_APPINDICATOR);
    }
    if !plugin_present(
        gstreamer_library_dir().as_deref(),
        |name| std::env::var_os(name),
        AUDIO_SINK_PLUGIN,
    ) {
        return Some(EXIT_NO_AUDIO_SINK);
    }
    let (major, minor, micro) = linked_webkit_version();
    println!("webkit={major}.{minor}.{micro}");
    Some(0)
}

/// For the log: the system's GStreamer is in use and has no [`AUDIO_SINK_PLUGIN`].
#[cfg(target_os = "linux")]
pub fn host_lacks_audio_sink() -> bool {
    let env = |name: &str| std::env::var_os(name);
    !bundled_runtime(env)
        && !plugin_present(gstreamer_library_dir().as_deref(), env, AUDIO_SINK_PLUGIN)
}

/// The Debian packages whose GStreamer plugins are missing for mp4 playback, in install order.
/// Empty on a bundled AppImage (it ships gst-libav) and when GStreamer's directory is unknown.
#[cfg(target_os = "linux")]
fn missing_mp4_playback_packages(
    bundled: bool,
    library_dir: Option<&Path>,
    env: impl Fn(&str) -> Option<OsString>,
) -> Vec<&'static str> {
    if bundled {
        return Vec::new();
    }
    MP4_PLAYBACK_PLUGINS
        .iter()
        .filter(|(plugin, _)| !plugin_present(library_dir, &env, plugin))
        .map(|(_, package)| *package)
        .collect()
}

/// For the Video page: the packages to install so generated videos play. Always empty off Linux.
#[tauri::command]
pub fn missing_media_codec_packages() -> Vec<&'static str> {
    #[cfg(target_os = "linux")]
    {
        let env = |name: &str| std::env::var_os(name);
        // The notice is an apt command: say nothing on rpm / pacman systems.
        if !Path::new("/etc/debian_version").is_file() {
            return Vec::new();
        }
        missing_mp4_playback_packages(
            bundled_runtime(env),
            gstreamer_library_dir().as_deref(),
            env,
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

/// GStreamer's own search (gst_registry_scan_plugins): `*_1_0` beats the plain name, and a set
/// system path replaces the default directory. Read from disk: building the registry can outlast
/// the probe's timeout. An unknown default directory counts as present.
#[cfg(target_os = "linux")]
fn plugin_present(
    library_dir: Option<&Path>,
    env: impl Fn(&str) -> Option<OsString>,
    plugin: &str,
) -> bool {
    let var = |name: &str| env(&format!("{name}_1_0")).or_else(|| env(name));
    let mut directories: Vec<PathBuf> = var("GST_PLUGIN_PATH")
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default();
    match var("GST_PLUGIN_SYSTEM_PATH") {
        Some(paths) => directories.extend(std::env::split_paths(&paths)),
        None => match library_dir {
            Some(library_dir) => directories.push(library_dir.join("gstreamer-1.0")),
            None => return true,
        },
    }
    directories
        .iter()
        .any(|directory| directory.is_absolute() && directory.join(plugin).is_file())
}

/// The directory of the libgstreamer the linked WebKitGTK loaded.
#[cfg(target_os = "linux")]
fn gstreamer_library_dir() -> Option<PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    // SAFETY: NUL-terminated names; the file name is copied before the handle closes.
    let library = unsafe {
        let handle = libc::dlopen(
            b"libgstreamer-1.0.so.0\0".as_ptr().cast(),
            libc::RTLD_LAZY | libc::RTLD_LOCAL,
        );
        if handle.is_null() {
            return None;
        }
        let symbol = libc::dlsym(handle, b"gst_init\0".as_ptr().cast());
        let mut info: libc::Dl_info = std::mem::zeroed();
        let library = (!symbol.is_null()
            && libc::dladdr(symbol, &mut info) != 0
            && !info.dli_fname.is_null())
        .then(|| {
            let name = std::ffi::CStr::from_ptr(info.dli_fname).to_bytes();
            PathBuf::from(std::ffi::OsStr::from_bytes(name))
        });
        libc::dlclose(handle);
        library
    }?;
    Some(library.canonicalize().ok()?.parent()?.to_path_buf())
}

#[cfg(target_os = "linux")]
fn library_loads(name: &[u8]) -> bool {
    // SAFETY: a NUL-terminated name; the handle is closed straight away.
    unsafe {
        let handle = libc::dlopen(name.as_ptr().cast(), libc::RTLD_LAZY | libc::RTLD_LOCAL);
        if handle.is_null() {
            return false;
        }
        libc::dlclose(handle);
        true
    }
}

/// The version of the libwebkit2gtk that actually loaded.
#[cfg(target_os = "linux")]
pub fn linked_webkit_version() -> (u32, u32, u32) {
    // SAFETY: plain getters with no preconditions; no GTK initialization needed.
    unsafe {
        (
            webkit2gtk_sys::webkit_get_major_version(),
            webkit2gtk_sys::webkit_get_minor_version(),
            webkit2gtk_sys::webkit_get_micro_version(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |name| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| OsString::from(value))
        }
    }

    #[test]
    fn only_an_exact_host_mode_leaves_the_bundled_runtime() {
        assert!(!bundled_runtime(env(&[])));
        assert!(!bundled_runtime(env(&[(MODE_ENV, "host")])));
        for mode in [
            None,
            Some("bundled"),
            Some("HOST"),
            Some("host "),
            Some(""),
            Some("x"),
        ] {
            let mut vars = vec![(APPIMAGE, "/tmp/Unsloth.AppImage")];
            if let Some(mode) = mode {
                vars.push((MODE_ENV, mode));
            }
            assert!(bundled_runtime(env(&vars)), "{mode:?}");
        }
        assert!(!bundled_runtime(env(&[
            (APPIMAGE, "/tmp/Unsloth.AppImage"),
            (MODE_ENV, "host")
        ])));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_audio_sink_plugin_is_looked_up_where_gstreamer_scans() {
        let host = tempfile::tempdir().unwrap();
        let library_dir = host.path().join("lib");
        let default = library_dir.join("gstreamer-1.0");
        let extra = host.path().join("extra");
        for directory in [&default, &extra] {
            std::fs::create_dir_all(directory).unwrap();
        }
        let extra_path = format!("/nonexistent:{}", extra.display());
        let present = |vars: &[(&str, &str)]| {
            plugin_present(Some(&library_dir), env(vars), AUDIO_SINK_PLUGIN)
        };

        assert!(plugin_present(None, env(&[]), AUDIO_SINK_PLUGIN));
        assert!(!present(&[]));
        std::fs::write(extra.join(AUDIO_SINK_PLUGIN), b"").unwrap();
        assert!(!present(&[]));
        for name in [
            "GST_PLUGIN_PATH",
            "GST_PLUGIN_PATH_1_0",
            "GST_PLUGIN_SYSTEM_PATH",
            "GST_PLUGIN_SYSTEM_PATH_1_0",
        ] {
            assert!(present(&[(name, &extra_path)]), "{name}");
        }
        assert!(!present(&[
            ("GST_PLUGIN_PATH_1_0", ""),
            ("GST_PLUGIN_PATH", &extra_path)
        ]));

        std::fs::remove_file(extra.join(AUDIO_SINK_PLUGIN)).unwrap();
        std::fs::write(default.join(AUDIO_SINK_PLUGIN), b"").unwrap();
        assert!(present(&[]));
        assert!(present(&[("GST_PLUGIN_PATH", &extra_path)]));
        assert!(!present(&[("GST_PLUGIN_SYSTEM_PATH", &extra_path)]));
        assert!(!present(&[
            ("GST_PLUGIN_SYSTEM_PATH_1_0", ""),
            ("GST_PLUGIN_SYSTEM_PATH", "")
        ]));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn missing_mp4_playback_plugins_name_their_debian_packages() {
        let host = tempfile::tempdir().unwrap();
        let library_dir = host.path().join("lib");
        let plugins = library_dir.join("gstreamer-1.0");
        std::fs::create_dir_all(&plugins).unwrap();
        let missing =
            |bundled| missing_mp4_playback_packages(bundled, Some(&library_dir), env(&[]));

        // Only gst-plugins-bad installed: the state of a deb that recommended plugins-bad alone.
        std::fs::write(plugins.join("libgstvideoparsersbad.so"), b"").unwrap();
        assert_eq!(
            missing(false),
            vec!["gstreamer1.0-libav", "gstreamer1.0-plugins-good"]
        );
        assert!(missing(true).is_empty(), "a bundled AppImage ships its own");
        std::fs::write(plugins.join("libgstisomp4.so"), b"").unwrap();
        assert_eq!(missing(false), vec!["gstreamer1.0-libav"]);
        std::fs::write(plugins.join("libgstlibav.so"), b"").unwrap();
        assert!(missing(false).is_empty());
        assert!(missing_mp4_playback_packages(false, None, env(&[])).is_empty());
    }

    #[test]
    fn the_deb_recommends_the_mp4_playback_packages() {
        let conf: serde_json::Value =
            serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let recommends = &conf["bundle"]["linux"]["deb"]["recommends"];
        for package in [
            "gstreamer1.0-plugins-bad",
            "gstreamer1.0-plugins-good",
            "gstreamer1.0-libav",
        ] {
            assert!(
                recommends.as_array().unwrap().iter().any(|p| p == package),
                "{package} missing from {recommends}"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_gstreamer_webkit_loaded_is_found_on_disk() {
        let library_dir = gstreamer_library_dir().expect("WebKitGTK links GStreamer");
        assert!(library_dir.join("libgstreamer-1.0.so.0").exists());
    }

    #[test]
    fn the_diagnostics_label_is_one_of_three_values() {
        assert_eq!(mode_label(env(&[])), "none");
        assert_eq!(
            mode_label(env(&[(APPIMAGE, "/a"), (MODE_ENV, "host")])),
            "host"
        );
        for mode in ["bundled", "<script>", "host\nx"] {
            assert_eq!(
                mode_label(env(&[(APPIMAGE, "/a"), (MODE_ENV, mode)])),
                "bundled"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_linked_webkit_reports_a_plausible_version() {
        let (major, minor, _) = linked_webkit_version();
        assert_eq!(major, 2);
        assert!(minor >= 36, "{minor}");
    }
}
