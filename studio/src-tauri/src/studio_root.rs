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
fn studio_root_from_marker() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let marker = home.join(".unsloth").join("studio-home");
    let raw = std::fs::read_to_string(&marker).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    expand_studio_home_value(OsString::from(trimmed))
}

/// Resolve the Studio install root using the priority chain. Returns
/// `None` only when the home directory cannot be determined and no env
/// override is set (extremely rare on supported platforms).
pub fn resolve_studio_root() -> Option<PathBuf> {
    if let Some(p) = studio_root_from_env() {
        return Some(p);
    }
    if let Some(p) = studio_root_from_marker() {
        return Some(p);
    }
    let home = dirs::home_dir()?;
    Some(home.join(".unsloth").join("studio"))
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
}
