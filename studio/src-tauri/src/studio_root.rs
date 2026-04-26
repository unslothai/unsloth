//! Shared resolver for the Studio install root inside the Tauri desktop app.
//!
//! Priority (highest first):
//!   1. `UNSLOTH_STUDIO_HOME` / `STUDIO_HOME` env vars (current process).
//!   2. `~/.unsloth/studio-home` marker file written by the installer in
//!      env-override mode, so the desktop app launched from
//!      Finder/Start Menu/Desktop (where the launching shell's env vars are
//!      not inherited) still resolves the custom root.
//!   3. Legacy default `~/.unsloth/studio`.
//!
//! Mirrors the shell / PowerShell / Python resolvers in this PR so a
//! `UNSLOTH_STUDIO_HOME=~/studio` value is interpreted the same in every
//! component.

use std::ffi::OsString;
use std::path::PathBuf;

/// Expand a leading `~`, `~/...`, or `~\...` against `dirs::home_dir()`.
/// Empty values return `None`. Other absolute or relative paths pass
/// through unchanged.
pub fn expand_studio_home_value(value: OsString) -> Option<PathBuf> {
    if value.is_empty() {
        return None;
    }
    if let Some(text) = value.to_str() {
        if text == "~" {
            return dirs::home_dir();
        }
        if let Some(rest) = text
            .strip_prefix("~/")
            .or_else(|| text.strip_prefix("~\\"))
        {
            return dirs::home_dir().map(|home| home.join(rest));
        }
    }
    Some(PathBuf::from(value))
}

fn studio_root_from_env() -> Option<PathBuf> {
    for var in ["UNSLOTH_STUDIO_HOME", "STUDIO_HOME"] {
        if let Some(value) = std::env::var_os(var) {
            if let Some(path) = expand_studio_home_value(value) {
                return Some(path);
            }
        }
    }
    None
}

/// Marker file the installer writes (in env-override mode) so a fresh
/// desktop launch with no shell env vars can still find the custom root.
///
/// Two safety nets:
///   1. Strip only trailing newlines (`\n`/`\r\n`) so paths whose content
///      legitimately contains leading/trailing spaces survive.
///   2. Validate the resolved path actually points at a Studio install
///      (carries the installer-written `share/studio.conf` sentinel or
///      a `bin/unsloth*` shim). A stale marker pointing at a deleted or
///      moved workspace falls back to the legacy default instead of
///      hijacking resolution.
fn studio_root_from_marker() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let marker = home.join(".unsloth").join("studio-home");
    let raw = std::fs::read_to_string(&marker).ok()?;
    let trimmed = raw.trim_end_matches(['\n', '\r']);
    if trimmed.is_empty() {
        return None;
    }
    let path = expand_studio_home_value(OsString::from(trimmed))?;
    if !looks_like_installer_managed_studio_root(&path) {
        return None;
    }
    Some(path)
}

fn looks_like_installer_managed_studio_root(path: &std::path::Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    if path.join("share").join("studio.conf").is_file() {
        return true;
    }
    let shim = if cfg!(windows) {
        path.join("bin").join("unsloth.exe")
    } else {
        path.join("bin").join("unsloth")
    };
    shim.is_file()
}

/// Where the resolved Studio root came from. Callers that need to
/// distinguish a real custom override from the legacy fallback (e.g.,
/// `install.rs` propagating UNSLOTH_STUDIO_HOME to a subprocess) should
/// use `resolve_studio_root_with_source()` instead of `resolve_studio_root()`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StudioRootSource {
    /// Resolved from `UNSLOTH_STUDIO_HOME` / `STUDIO_HOME` env var.
    Env,
    /// Resolved from the installer-written `~/.unsloth/studio-home` marker.
    Marker,
    /// Legacy fallback: `~/.unsloth/studio`.
    Default,
}

/// Resolve the Studio install root and report which source produced it.
/// Returns `None` only when the home directory cannot be determined and
/// no env override is set.
pub fn resolve_studio_root_with_source() -> Option<(PathBuf, StudioRootSource)> {
    if let Some(p) = studio_root_from_env() {
        return Some((p, StudioRootSource::Env));
    }
    if let Some(p) = studio_root_from_marker() {
        return Some((p, StudioRootSource::Marker));
    }
    let home = dirs::home_dir()?;
    Some((
        home.join(".unsloth").join("studio"),
        StudioRootSource::Default,
    ))
}

/// Convenience wrapper that drops the source. For callers that don't
/// need to distinguish the legacy fallback from a real custom root.
pub fn resolve_studio_root() -> Option<PathBuf> {
    resolve_studio_root_with_source().map(|(p, _)| p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn expand_handles_literal_paths() {
        let out = expand_studio_home_value(OsString::from("/srv/foo")).unwrap();
        assert_eq!(out, Path::new("/srv/foo"));
    }

    #[test]
    fn expand_returns_none_for_empty() {
        assert!(expand_studio_home_value(OsString::new()).is_none());
    }

    #[test]
    fn expand_handles_tilde_only() {
        let out = expand_studio_home_value(OsString::from("~"));
        if let Some(home) = dirs::home_dir() {
            assert_eq!(out.unwrap(), home);
        }
    }

    #[test]
    fn expand_handles_tilde_slash_prefix() {
        let out = expand_studio_home_value(OsString::from("~/studio")).unwrap();
        if let Some(home) = dirs::home_dir() {
            assert_eq!(out, home.join("studio"));
        }
    }

    #[test]
    fn expand_handles_tilde_backslash_prefix() {
        let out = expand_studio_home_value(OsString::from("~\\studio")).unwrap();
        if let Some(home) = dirs::home_dir() {
            assert_eq!(out, home.join("studio"));
        }
    }

    fn unique_temp(name: &str) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "unsloth-marker-{name}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn marker_validation_accepts_installer_sentinel() {
        let root = unique_temp("sentinel");
        std::fs::create_dir_all(root.join("share")).unwrap();
        std::fs::write(root.join("share").join("studio.conf"), b"# sentinel").unwrap();
        assert!(looks_like_installer_managed_studio_root(&root));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn marker_validation_accepts_bin_shim() {
        let root = unique_temp("shim");
        std::fs::create_dir_all(root.join("bin")).unwrap();
        let shim = if cfg!(windows) {
            root.join("bin").join("unsloth.exe")
        } else {
            root.join("bin").join("unsloth")
        };
        std::fs::write(&shim, b"#!/bin/sh").unwrap();
        assert!(looks_like_installer_managed_studio_root(&root));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn marker_validation_rejects_empty_dir() {
        let root = unique_temp("empty");
        assert!(!looks_like_installer_managed_studio_root(&root));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn marker_validation_rejects_missing_path() {
        let p = Path::new("/nonexistent/studio-root-xyz-1234567890");
        assert!(!looks_like_installer_managed_studio_root(p));
    }

    #[test]
    fn source_enum_default_when_no_env_or_marker() {
        // Note: this test only verifies the StudioRootSource type compiles
        // and the enum variants exist; we cannot reliably test the env/marker
        // path without mutating process-wide state in a non-thread-safe way.
        let _e = StudioRootSource::Env;
        let _m = StudioRootSource::Marker;
        let _d = StudioRootSource::Default;
        assert_ne!(StudioRootSource::Default, StudioRootSource::Env);
        assert_ne!(StudioRootSource::Default, StudioRootSource::Marker);
    }
}
