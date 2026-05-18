use std::cmp::Ordering;

pub(crate) const DESKTOP_PROTOCOL_VERSION: u16 = 1;
pub(crate) const DESKTOP_MANAGEABILITY_VERSION: u16 = 1;
// Explicit backend package minimum, not the desktop app Cargo version: backend
// and app releases can diverge. When bumping, verify this package exists on PyPI.
pub(super) const MIN_DESKTOP_BACKEND_VERSION: &str = "2026.5.3";

#[derive(Debug, Eq, PartialEq)]
pub(super) struct ParsedVersion {
    release: [u64; 3],
    suffix: VersionSuffix,
}

#[derive(Debug, Eq, PartialEq)]
enum VersionSuffix {
    Dev(u64),
    Alpha(u64),
    Beta(u64),
    Rc(u64),
    PreRelease(String),
    Stable,
    Build,
    Post(u64),
}

pub(super) fn parse_version(value: &str) -> Option<ParsedVersion> {
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
    let suffix = parse_version_suffix(&patch_and_suffix[patch_len..])?;
    Some(ParsedVersion {
        release: [major, minor, patch_and_suffix[..patch_len].parse().ok()?],
        suffix,
    })
}

fn parse_version_suffix(suffix: &str) -> Option<VersionSuffix> {
    if suffix.is_empty() {
        return Some(VersionSuffix::Stable);
    }

    if let Some(value) = suffix.strip_prefix('+') {
        return suffix_part_valid(value).then_some(VersionSuffix::Build);
    }
    if let Some(value) = suffix.strip_prefix('-') {
        return suffix_part_valid(value)
            .then(|| VersionSuffix::PreRelease(value.to_ascii_lowercase()));
    }
    if let Some(number) = parse_numbered_suffix(suffix, &[".post", "post"]) {
        return number.map(VersionSuffix::Post);
    }
    if let Some(number) = parse_numbered_suffix(suffix, &[".dev", "dev"]) {
        return number.map(VersionSuffix::Dev);
    }
    if let Some(number) = parse_numbered_suffix(suffix, &[".rc", "rc"]) {
        return number.map(VersionSuffix::Rc);
    }
    if let Some(number) = parse_numbered_suffix(suffix, &["a"]) {
        return number.map(VersionSuffix::Alpha);
    }
    if let Some(number) = parse_numbered_suffix(suffix, &["b"]) {
        return number.map(VersionSuffix::Beta);
    }

    None
}

fn parse_numbered_suffix(suffix: &str, prefixes: &[&str]) -> Option<Option<u64>> {
    for prefix in prefixes {
        let Some(number) = suffix.strip_prefix(prefix) else {
            continue;
        };
        if number.is_empty() {
            return Some(Some(0));
        }
        return Some(number.parse().ok());
    }
    None
}

fn suffix_part_valid(value: &str) -> bool {
    !value.is_empty()
        && value.split('.').all(|part| {
            !part.is_empty() && part.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        })
}

fn compare_versions(left: &ParsedVersion, right: &ParsedVersion) -> Ordering {
    match left.release.cmp(&right.release) {
        Ordering::Equal => compare_suffixes(&left.suffix, &right.suffix),
        ordering => ordering,
    }
}

fn compare_suffixes(left: &VersionSuffix, right: &VersionSuffix) -> Ordering {
    let left_precedence = suffix_precedence(left);
    let right_precedence = suffix_precedence(right);
    if left_precedence != right_precedence {
        return left_precedence.cmp(&right_precedence);
    }

    match (left, right) {
        (VersionSuffix::Dev(left), VersionSuffix::Dev(right))
        | (VersionSuffix::Alpha(left), VersionSuffix::Alpha(right))
        | (VersionSuffix::Beta(left), VersionSuffix::Beta(right))
        | (VersionSuffix::Rc(left), VersionSuffix::Rc(right))
        | (VersionSuffix::Post(left), VersionSuffix::Post(right)) => left.cmp(right),
        (VersionSuffix::PreRelease(left), VersionSuffix::PreRelease(right)) => left.cmp(right),
        _ => Ordering::Equal,
    }
}

fn suffix_precedence(suffix: &VersionSuffix) -> u8 {
    match suffix {
        VersionSuffix::Dev(_) => 0,
        VersionSuffix::Alpha(_) => 1,
        VersionSuffix::Beta(_) => 2,
        VersionSuffix::Rc(_) | VersionSuffix::PreRelease(_) => 3,
        VersionSuffix::Stable | VersionSuffix::Build => 4,
        VersionSuffix::Post(_) => 5,
    }
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
    compare_versions(&actual, &minimum) != Ordering::Less
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
