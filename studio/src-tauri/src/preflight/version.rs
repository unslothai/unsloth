pub(crate) const DESKTOP_PROTOCOL_VERSION: u16 = 1;
pub(crate) const DESKTOP_MANAGEABILITY_VERSION: u16 = 1;
// Explicit backend package minimum, not the desktop app Cargo version: backend
// and app releases can diverge. When bumping, verify this package exists on PyPI.
pub(super) const MIN_DESKTOP_BACKEND_VERSION: &str = "2026.5.2";

pub(super) fn parse_version(value: &str) -> Option<[u64; 3]> {
    let value = value.trim();
    let mut parts = value.splitn(3, '.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch_and_suffix = parts.next()?;
    let patch_len = patch_and_suffix
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(patch_and_suffix.len());
    if patch_len == 0 {
        return None;
    }
    let suffix = &patch_and_suffix[patch_len..];
    if !version_suffix_allowed(suffix) {
        return None;
    }
    Some([major, minor, patch_and_suffix[..patch_len].parse().ok()?])
}

pub(super) fn version_suffix_allowed(suffix: &str) -> bool {
    suffix.is_empty()
        || suffix.starts_with('+')
        || suffix.starts_with(".post")
        || suffix.starts_with(".dev")
        || suffix.starts_with(".rc")
        || suffix.starts_with("post")
        || suffix.starts_with("dev")
        || suffix.starts_with("rc")
        || suffix.starts_with('a')
        || suffix.starts_with('b')
}

pub(super) fn backend_version_compatible(version: Option<&str>) -> bool {
    let Some(version) = version else {
        return false;
    };
    if cfg!(debug_assertions) && version == "dev" {
        return true;
    }
    let Some(actual) = parse_version(version) else {
        return false;
    };
    let Some(minimum) = parse_version(MIN_DESKTOP_BACKEND_VERSION) else {
        return false;
    };
    actual >= minimum
}

pub(crate) fn backend_version_stale_reason(version: Option<&str>) -> Option<String> {
    if backend_version_compatible(version) {
        return None;
    }
    match version {
        None | Some("") => Some("desktop_backend_version_missing".to_string()),
        Some("dev") if !cfg!(debug_assertions) => {
            Some("desktop_backend_version_invalid".to_string())
        }
        Some(value) if parse_version(value).is_none() => {
            Some("desktop_backend_version_invalid".to_string())
        }
        Some(_) => Some("desktop_backend_version_too_old".to_string()),
    }
}
