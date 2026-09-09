use super::types::ManagedProbe;
use super::version::{
    managed_backend_version_stale_reason, DESKTOP_MANAGEABILITY_VERSION, DESKTOP_PROTOCOL_VERSION,
};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant, UNIX_EPOCH};
use tokio::io::AsyncReadExt;

// 3: the cached capability gained studio_install_ok / studio_install_reason.
// 4: the fingerprint gained the llama.cpp runtime, so a quarantined file
//    invalidates the entry instead of being served a stale Ready.
const MANAGED_CAPABILITY_CACHE_SCHEMA: u16 = 4;

/// The install is fine; the directory its children must run from is not reachable.
pub(super) const WORKING_DIRECTORY_UNAVAILABLE: &str = "working_directory_unavailable";
/// The profile is reachable but a user-written path setting is not resolvable,
/// so reinstalling hits the same wall. Mirrored in the frontend message map.
pub(super) const PATH_SETTING_UNRESOLVABLE: &str = "path_setting_unresolvable";

/// The reason a managed context failure is reported under, with the setting that
/// caused it where there is one: "one of Unsloth's folder settings" is not
/// something a user can act on, and every pin failure names the setting it could
/// not preserve. The name only, never the value, since this reaches the window.
pub(super) fn context_reason(error: &crate::process::ManagedContextError) -> String {
    match error {
        crate::process::ManagedContextError::WorkingDirectory(_) => {
            WORKING_DIRECTORY_UNAVAILABLE.to_string()
        }
        crate::process::ManagedContextError::PathSetting(detail) => {
            match setting_name(detail) {
                Some(name) => format!("{PATH_SETTING_UNRESOLVABLE}:{name}"),
                None => PATH_SETTING_UNRESOLVABLE.to_string(),
            }
        }
    }
}

/// The leading token of a pin failure, when it looks like an environment name.
fn setting_name(detail: &str) -> Option<&str> {
    let name = detail.split_whitespace().next()?;
    let named = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
    named.then_some(name)
}

/// Whether the reason is a context the app cannot build, not a repairable install.
pub(super) fn is_context_reason(reason: &str) -> bool {
    let head = reason.split(':').next().unwrap_or(reason);
    head == WORKING_DIRECTORY_UNAVAILABLE || head == PATH_SETTING_UNRESOLVABLE
}

const FNV64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV64_PRIME: u64 = 0x100000001b3;
const HASHED_MARKER_MAX_BYTES: u64 = 64 * 1024;

const FALLBACK_MARKER_NAMES: &[&str] = &[
    // In the fingerprint, not just the cached answer: a repair touching only
    // studio.txt leaves every other marker alone, so a cache entry written
    // while healthy would outlive the dropped manifest. Mirrors MANIFEST_NAME.
    "unsloth_install_manifest.json",
    "pyvenv.cfg",
    "uv.lock",
    "requirements.txt",
    "python.exe",
    "python",
];

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DesktopCapability {
    desktop_protocol_version: Option<u16>,
    desktop_manageability_version: Option<u16>,
    supports_api_only: Option<bool>,
    supports_provision_desktop_auth: Option<bool>,
    supports_desktop_backend_ownership: Option<bool>,
    desktop_auth_stale_reason: Option<String>,
    // A part-way install leaves a CLI that answers `-h` and a backend that dies
    // on `import structlog`, so a running CLI does not mean ready.
    studio_install_ok: Option<bool>,
    studio_install_reason: Option<String>,
    // Quarantine takes files out of an otherwise present llama.cpp tree, which
    // staleness on the managed Python alone missed: the desktop launched and the
    // model load failed later looking like a bad GGUF. None when the CLI is too
    // old to answer or nothing is installed yet.
    llama_runtime_ok: Option<bool>,
    llama_runtime_reason: Option<String>,
    version: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ManagedCapabilityCache {
    schema: u16,
    bin_path: String,
    bin_size: u64,
    bin_mtime_ms: u64,
    studio_root_id: Option<String>,
    marker_path: Option<String>,
    marker_size: Option<u64>,
    marker_mtime_ms: Option<u64>,
    llama_runtime: Option<String>,
    desktop_protocol_version: u16,
    desktop_manageability_version: u16,
    capability: DesktopCapability,
}

#[derive(Debug, Clone)]
struct MarkerFingerprint {
    path: String,
    size: u64,
    mtime_ms: u64,
    content_hash: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ManagedBinFingerprint {
    bin_path: String,
    bin_size: u64,
    bin_mtime_ms: u64,
    studio_root_id: Option<String>,
    marker_path: Option<String>,
    marker_size: Option<u64>,
    marker_mtime_ms: Option<u64>,
    llama_runtime: Option<String>,
}

fn modified_ms(metadata: &fs::Metadata) -> Option<u64> {
    metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
}
fn hash_bytes(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |mut next, byte| {
        next ^= u64::from(*byte);
        next.wrapping_mul(FNV64_PRIME)
    })
}

fn marker_content_hash(path: &Path, metadata: &fs::Metadata) -> Option<u64> {
    if metadata.len() > HASHED_MARKER_MAX_BYTES {
        return None;
    }
    fs::read(path)
        .ok()
        .map(|bytes| hash_bytes(FNV64_OFFSET_BASIS, &bytes))
}

fn site_packages_dirs(venv_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    #[cfg(unix)]
    {
        if let Ok(lib_dir) = fs::read_dir(venv_dir.join("lib")) {
            for entry in lib_dir.flatten() {
                out.push(entry.path().join("site-packages"));
            }
        }
    }
    out.push(venv_dir.join("Lib").join("site-packages"));
    // read_dir order is unspecified and the hashes below fold in order.
    out.sort();
    out
}

/// Hash of the .dist-info / .egg-info names present, version included.
///
/// pip uninstall rewrites nothing else that is fingerprinted, so a venv that
/// lost a studio.txt dependency would keep serving the healthy verdict.
fn installed_distributions_hash(site_packages: &Path) -> Option<u64> {
    let mut names: Vec<String> = fs::read_dir(site_packages)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            (name.ends_with(".dist-info") || name.ends_with(".egg-info")).then_some(name)
        })
        .collect();
    names.sort();
    Some(names.iter().fold(FNV64_OFFSET_BASIS, |hash, name| {
        hash_bytes(hash, name.as_bytes())
    }))
}

fn marker_candidates_for_bin(bin: &Path) -> Vec<PathBuf> {
    let Some(scripts_dir) = bin.parent() else {
        return Vec::new();
    };
    let Some(venv_dir) = scripts_dir.parent() else {
        return Vec::new();
    };
    let mut out = Vec::new();

    for site_packages in site_packages_dirs(venv_dir) {
        out.push(
            site_packages
                .join("unsloth_cli")
                .join("commands")
                .join("studio.py"),
        );
    }
    for marker_name in FALLBACK_MARKER_NAMES {
        out.push(venv_dir.join(marker_name));
        out.push(scripts_dir.join(marker_name));
    }
    out
}

/// The file whose size and mtime stand for "this CLI", which is the launcher when there
/// is one.
///
/// On Windows there may not be. Antivirus quarantine deletes the generated unsloth.exe
/// and leaves a venv that still runs through its interpreter, and since that is now a
/// supported layout, find_unsloth_binary_in_studio_dir hands back a launcher path that
/// does not exist. Keying the fingerprint on it would fail fs::metadata, so the
/// capability cache could be neither read nor written, and every preflight would pay
/// both the -h and the desktop-capabilities subprocess with their own 10s ceilings.
/// python.exe is the right stand-in: it is what actually starts the CLI there, and an
/// update replaces the whole venv, so it moves when the launcher would have.
fn fingerprint_identity_file(bin: &Path) -> Option<PathBuf> {
    if bin.exists() {
        return Some(bin.to_path_buf());
    }
    #[cfg(windows)]
    {
        let interpreter = bin.parent()?.join("python.exe");
        if interpreter.exists() {
            return Some(interpreter);
        }
    }
    None
}

fn managed_bin_fingerprint(bin: &Path) -> Option<ManagedBinFingerprint> {
    // Metadata from whatever identifies this CLI, but the cache key below stays the
    // launcher path, so the two layouts of one install cannot collide.
    let bin_metadata = fs::metadata(fingerprint_identity_file(bin)?).ok()?;
    let bin_path = bin
        .canonicalize()
        .unwrap_or_else(|_| bin.to_path_buf())
        .to_string_lossy()
        .into_owned();

    let studio_root_id = crate::desktop_backend_owner::read_expected_studio_root_id();
    let mut marker_entries: Vec<MarkerFingerprint> = marker_candidates_for_bin(bin)
        .into_iter()
        .filter_map(|path| {
            let metadata = fs::metadata(&path).ok()?;
            Some(MarkerFingerprint {
                path: path
                    .canonicalize()
                    .unwrap_or(path.clone())
                    .to_string_lossy()
                    .into_owned(),
                size: metadata.len(),
                mtime_ms: modified_ms(&metadata)?,
                content_hash: marker_content_hash(&path, &metadata),
            })
        })
        .collect();
    marker_entries.sort_by(|left, right| left.path.cmp(&right.path));
    let mut marker_hash = marker_entries
        .iter()
        .fold(FNV64_OFFSET_BASIS, |hash, marker| {
            let next = hash_bytes(hash, marker.path.as_bytes());
            let next = hash_bytes(next, &marker.size.to_le_bytes());
            let next = hash_bytes(next, &marker.mtime_ms.to_le_bytes());
            if let Some(content_hash) = marker.content_hash {
                hash_bytes(next, &content_hash.to_le_bytes())
            } else {
                next
            }
        });
    let mut tracked = marker_entries.len();
    if let Some(venv_dir) = bin.parent().and_then(Path::parent) {
        for site_packages in site_packages_dirs(venv_dir) {
            let Some(dist_hash) = installed_distributions_hash(&site_packages) else {
                continue;
            };
            marker_hash = hash_bytes(marker_hash, site_packages.to_string_lossy().as_bytes());
            marker_hash = hash_bytes(marker_hash, &dist_hash.to_le_bytes());
            tracked += 1;
        }
    }
    let marker_path = (tracked > 0).then(|| "markers".to_string());
    let marker_size = (tracked > 0).then_some(tracked as u64);
    let marker_mtime_ms = (tracked > 0).then_some(marker_hash);

    Some(ManagedBinFingerprint {
        bin_path,
        bin_size: bin_metadata.len(),
        bin_mtime_ms: modified_ms(&bin_metadata)?,
        studio_root_id,
        marker_path,
        marker_size,
        marker_mtime_ms,
        llama_runtime: llama_runtime_fingerprint(bin),
    })
}

/// The folder a leading `~` names, resolved the way the CLI child resolves it.
///
/// `ntpath.expanduser` answers USERPROFILE, while `dirs::home_dir()` reads the
/// known folder, which a portable or overridden profile moves. Taking the latter
/// would fingerprint a tree the child never looks at.
///
/// process.rs calls this rather than keeping its own order: it used to pass
/// `dirs::home_dir()` straight into `expand_windows_user`, so on a Windows box with
/// an overridden USERPROFILE the child was pinned to the known folder while this
/// fingerprinted the profile, and quarantine in the tree actually in use never
/// invalidated a cached healthy result.
pub(crate) fn tilde_home() -> Option<PathBuf> {
    if cfg!(windows) {
        if let Some(profile) = std::env::var_os("USERPROFILE") {
            if !profile.is_empty() {
                return Some(PathBuf::from(profile));
            }
        }
    }
    dirs::home_dir()
}

/// `UNSLOTH_LLAMA_CPP_PATH` resolved the way the CLI child sees it: trimmed, `~`
/// expanded (`default_managed_llama_dir` calls expanduser), and a relative value
/// anchored to this process's working directory (`relative_override_pins` pins it
/// from exactly there before the spawn).
///
/// `~name` is left alone, since nothing here can resolve another user's home; it
/// is then relative, so it is anchored like any other relative value.
///
/// UNSLOTH_STUDIO_HOME is deliberately not consulted: managed spawns scrub it
/// (MANAGED_CHILD_SCRUBBED_ENV), so the CLI falls through to the legacy root too.
#[cfg_attr(test, allow(dead_code))]
fn llama_runtime_override() -> Option<PathBuf> {
    llama_runtime_override_from(
        std::env::var("UNSLOTH_LLAMA_CPP_PATH").ok().as_deref(),
        tilde_home().as_deref(),
        std::env::current_dir().ok().as_deref(),
    )
}

/// Split out from the reads so tests can drive it without a process-wide variable.
pub(crate) fn llama_runtime_override_from(
    value: Option<&str>,
    home: Option<&Path>,
    cwd: Option<&Path>,
) -> Option<PathBuf> {
    let value = value?.trim();
    if value.is_empty() {
        return None;
    }
    if value == "~" {
        return home.map(Path::to_path_buf);
    }
    if let Some(rest) = value
        .strip_prefix("~/")
        .or_else(|| cfg!(windows).then(|| value.strip_prefix("~\\")).flatten())
    {
        return Some(home?.join(rest));
    }
    if let Some(named) = named_user_home(value, home) {
        return Some(named);
    }
    let path = PathBuf::from(value);
    if path.is_absolute() {
        return Some(path);
    }
    // Anchored, not discarded: dropping it makes both halves of a launch pair
    // report None, so the cache keeps hitting while the real runtime rots. The
    // child joins this value to this same directory, so joining it here watches
    // the tree the child was told about.
    Some(cwd?.join(path))
}

/// `~alice/llama.cpp` resolved to Alice's home, the way `Path.expanduser()` does.
///
/// posixpath.expanduser answers this out of the password database, so leaving it
/// alone made the value relative, anchored it under the desktop's own directory,
/// and fingerprinted a tree the child never opens. getpwnam is the same lookup
/// Python makes, so the two agree for LDAP and SSSD users as well as local ones.
/// None when the name is unknown, which leaves the value to be anchored like any
/// other relative path rather than guessed at. Windows has its own rule and its
/// own arm below.
#[cfg(unix)]
pub(crate) fn named_user_home(value: &str, _home: Option<&Path>) -> Option<PathBuf> {
    use std::os::unix::ffi::OsStrExt;

    let rest = value.strip_prefix('~')?;
    if rest.is_empty() {
        return None;
    }
    let (name, tail) = match rest.find('/') {
        Some(at) => (&rest[..at], &rest[at + 1..]),
        None => (rest, ""),
    };
    if name.is_empty() {
        return None;
    }
    let c_name = std::ffi::CString::new(name).ok()?;
    // getpwnam_r, not getpwnam: this runs on a Tauri worker thread, and getpwnam
    // hands back a pointer into storage shared by the whole process, so a
    // concurrent passwd lookup anywhere else could overwrite pw_dir while it was
    // being copied. Copying promptly narrows that window, it does not close it,
    // and a torn read here fingerprints an unrelated tree. The _r form writes
    // into a buffer this call owns.
    let mut buffer = vec![0i8; 1024];
    let mut passwd: libc::passwd = unsafe { std::mem::zeroed() };
    let home = loop {
        let mut found: *mut libc::passwd = std::ptr::null_mut();
        let code = unsafe {
            libc::getpwnam_r(
                c_name.as_ptr(),
                &mut passwd,
                buffer.as_mut_ptr() as *mut libc::c_char,
                buffer.len(),
                &mut found,
            )
        };
        if code == libc::ERANGE && buffer.len() < 1 << 20 {
            // The name resolves, the buffer was too small for its record. Only
            // grow so far, so a hostile or broken passwd source cannot make this
            // allocate without end.
            buffer.resize(buffer.len() * 2, 0);
            continue;
        }
        // A nonzero code is a lookup error and a null found is "no such user".
        // Both leave the value to be anchored like any other relative path.
        if code != 0 || found.is_null() {
            return None;
        }
        let dir = passwd.pw_dir;
        if dir.is_null() {
            return None;
        }
        break PathBuf::from(std::ffi::OsStr::from_bytes(unsafe {
            std::ffi::CStr::from_ptr(dir).to_bytes()
        }));
    };
    Some(if tail.is_empty() { home } else { home.join(tail) })
}

#[cfg(windows)]
pub(crate) fn named_user_home(value: &str, home: Option<&Path>) -> Option<PathBuf> {
    named_windows_user_home(value, home?, std::env::var("USERNAME").ok().as_deref())
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn named_user_home(_value: &str, _home: Option<&Path>) -> Option<PathBuf> {
    None
}

/// `~name` on Windows, the way ntpath.expanduser resolves it: the sibling of this
/// profile, and only where ntpath is willing to guess at all.
///
/// Leaving it alone made the value relative, so the fingerprint watched
/// `<cwd>\~other\llama.cpp` while process.rs pins the variable for the child
/// through its own `expand_windows_user` and the CLI's `Path.expanduser()` reads
/// it the same way, both landing on the real profile. Quarantine under that
/// profile then never invalidated a cached healthy result.
///
/// Compiled on every platform so the rule is testable off Windows; only the
/// Windows arm above calls it. None when ntpath would decline (a profile folder
/// not named after the current user, an unknown USERNAME), which leaves the value
/// anchored like any other relative path rather than guessed at.
#[cfg_attr(not(windows), allow(dead_code))]
fn named_windows_user_home(value: &str, home: &Path, username: Option<&str>) -> Option<PathBuf> {
    let rest = value.strip_prefix('~')?;
    let end = rest.find(['\\', '/']).unwrap_or(rest.len());
    let (name, tail) = (&rest[..end], &rest[end..]);
    if name.is_empty() {
        return None;
    }
    let home = home.to_string_lossy();
    // Split on the string, not with Path::parent: these are Windows paths
    // whichever platform is reading them.
    let cut = home.rfind(['\\', '/'])?;
    let this_profile = &home[cut + 1..];
    let base = match username {
        Some(user) if user == name => home.clone().into_owned(),
        // C:\Users\alice.DOMAIN is not alice's sibling, so ntpath refuses unless
        // this profile is named after the current user.
        Some(user) if user == this_profile => format!("{}{}", &home[..cut + 1], name),
        _ => return None,
    };
    Some(PathBuf::from(format!("{base}{tail}")))
}

/// The markers install.ps1 writes into its generated `unsloth.cmd`, and the size
/// past which the file is not that shim. Same pair `_is_managed_cmd_shim` reads,
/// so a hand rolled wrapper that happens to call the CLI is not mistaken for one.
#[cfg(any(windows, test))]
const CMD_SHIM_MARKERS: [&[u8]; 2] = [b"unsloth-studio-managed-launcher", b"from unsloth_cli import app"];
#[cfg(any(windows, test))]
const CMD_SHIM_MAX_BYTES: u64 = 8192;

/// Whether a directory carries the sentinel an installer-managed Studio root has,
/// mirroring `_looks_like_installer_managed_studio_home` in the CLI.
///
/// Compiled on every platform so the rule is testable off Windows; the runtime
/// answer follows the platform this is built for, as it does in Python.
fn looks_like_installer_managed_studio_home(candidate: &Path) -> bool {
    if candidate.join("share").join("studio.conf").is_file() {
        return true;
    }
    #[cfg(not(windows))]
    {
        candidate.join("bin").join("unsloth").is_file()
    }
    #[cfg(windows)]
    {
        if candidate.join("bin").join("unsloth.exe").is_file() {
            return true;
        }
        is_managed_cmd_shim(&candidate.join("bin").join("unsloth.cmd"))
    }
}

/// Whether a path is the `.cmd` shim this installer generates.
#[cfg(any(windows, test))]
fn is_managed_cmd_shim(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() || metadata.len() > CMD_SHIM_MAX_BYTES {
        return false;
    }
    let Ok(body) = fs::read(path) else {
        return false;
    };
    CMD_SHIM_MARKERS
        .iter()
        .all(|marker| body.windows(marker.len()).any(|window| window == *marker))
}

/// The llama.cpp root an installer-managed venv implies, with no ambient
/// override to say so.
///
/// The desktop strips UNSLOTH_STUDIO_HOME before spawning the CLI, but the CLI
/// puts it back: `_resolve_studio_home` re-infers the root from `sys.prefix` when
/// the venv is named `unsloth_studio` and the root carries an installer sentinel,
/// and `_ensure_studio_env_exported` then points UNSLOTH_LLAMA_CPP_PATH at that
/// root's llama.cpp. Reading only the ambient environment here fingerprinted the
/// legacy ~/.unsloth/llama.cpp instead, so quarantine inside the custom runtime
/// never moved the fingerprint and a cached Ready kept being served for a tree
/// the new health check never got to grade.
///
/// `bin` is the launcher file inside the venv, so its grandparent is `sys.prefix`.
fn inferred_studio_llama_root(bin: &Path) -> Option<PathBuf> {
    let prefix = bin.parent()?.parent()?;
    if prefix.file_name()? != "unsloth_studio" {
        return None;
    }
    let root = prefix.parent()?;
    // Python resolves both sides before comparing, so a symlinked or relative
    // path to the legacy root is still the legacy root, and the answer there is
    // the legacy llama.cpp rather than <root>/llama.cpp.
    let legacy = dirs::home_dir()?.join(".unsloth").join("studio");
    let same = match (root.canonicalize(), legacy.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => root == legacy,
    };
    if same {
        return None;
    }
    if !looks_like_installer_managed_studio_home(root) {
        return None;
    }
    Some(root.join("llama.cpp"))
}

/// Whether a folder holds a llama-server in one of the layouts the backend
/// accepts, which is the question that decides whether discovery stops there.
///
/// Presence, not usability, and the CLI half agrees: `_scan_pinned` returns a hit
/// for an executable candidate and an unavailable path for one that is merely
/// present, and both end the search. Only a layout holding no server at all is
/// walked past, so asking for the execute bit here would fingerprint a different
/// tree than the CLI grades in exactly the case where the pinned one is damaged.
fn holds_a_llama_server(root: &Path) -> bool {
    let name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let mut candidates = vec![root.join(name), root.join("build").join("bin").join(name)];
    if cfg!(windows) {
        candidates.push(root.join("build").join("bin").join("Release").join(name));
    }
    candidates.iter().any(|candidate| {
        let Ok(metadata) = fs::metadata(candidate) else {
            return false;
        };
        // Presence, not usability. _scan_pinned answers "non_executable" for a
        // candidate without the bit and the caller turns that into _unavailable,
        // which ends the search rather than falling through, so a server the
        // loader could not run still stops discovery at this folder.
        metadata.is_file()
    })
}

/// The managed llama.cpp install root, the same one default_managed_llama_dir
/// picks in Python.
fn llama_runtime_root(bin: &Path) -> Option<PathBuf> {
    // Hermetic under test: this walks a real directory, so otherwise a developer
    // with a runtime installed and a CI runner without one run different tests.
    // Unset means no runtime at all. Mirrors capability_cache_path()'s hook.
    #[cfg(test)]
    {
        let _ = bin;
        return std::env::var_os("UNSLOTH_TEST_LLAMA_RUNTIME_ROOT").map(PathBuf::from);
    }
    // The override is asked first and its answer is final: falling back to the
    // legacy tree when it is set but unresolvable would fingerprint a directory
    // the CLI is not reporting on.
    // Trimmed before deciding, because default_managed_llama_dir strips the value
    // and a whitespace-only one therefore sends the child to the legacy tree. A
    // raw is_empty test called that an override, took this branch, resolved to
    // None, and left the tree the child actually grades with no fingerprint at
    // all, so a library removed after a healthy result never invalidated Ready.
    #[cfg(not(test))]
    if std::env::var("UNSLOTH_LLAMA_CPP_PATH")
        .is_ok_and(|value| !value.trim().is_empty())
    {
        // An override the desktop wrote points at the managed tree and the CLI
        // grades it whatever it holds, so it is taken as given. A user-written one
        // only stops _scan_pinned when it actually holds a server; when it does
        // not, the CLI walks past it and grades the tree behind it, so stopping
        // here would fingerprint a folder nobody loads.
        let desktop_wrote_it =
            std::env::var("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH").as_deref() == Ok("1");
        let override_root = llama_runtime_override();
        if desktop_wrote_it {
            return override_root;
        }
        if let Some(root) = override_root {
            if holds_a_llama_server(&root) {
                return Some(root);
            }
        }
    }
    // No ambient override, so ask the venv the same question the CLI asks itself
    // before falling back to the legacy tree.
    #[cfg(not(test))]
    if let Some(inferred) = inferred_studio_llama_root(bin) {
        return Some(inferred);
    }
    #[cfg(not(test))]
    return Some(dirs::home_dir()?.join(".unsloth").join("llama.cpp"));
}

/// A cheap stand-in for "the llama.cpp runtime tree is unchanged": how many files
/// sit in its binary directory and how many bytes they total.
///
/// The rest of this fingerprint covers the managed venv only, so a quarantined
/// runtime file left it identical and preflight answered Ready from the cache
/// without ever asking the CLI. None when no runtime is installed, which is
/// NotInstalled rather than broken.
fn llama_runtime_fingerprint(bin: &Path) -> Option<String> {
    llama_runtime_fingerprint_at(&llama_runtime_root(bin)?)
}

/// The walk itself, against a given root, so the tests need no shared state.
fn llama_runtime_fingerprint_at(root: &Path) -> Option<String> {
    let mut bin = root.join("build").join("bin");
    if cfg!(windows) {
        bin = bin.join("Release");
    }
    let root_part = root_entrypoints(root);
    match fs::read_dir(&bin) {
        Ok(entries) => Some(format!("bin:{}|root:{root_part}", counted(entries))),
        // build/bin is gone but something is still there. Answering None would
        // make that identical to "nothing was ever installed", so a Ready cached
        // while no runtime existed kept matching once a marker appeared over a
        // missing build/bin, and the CLI never got to say llama_runtime_dir_missing.
        // Fingerprinting the root's own entries makes the marker's arrival move it.
        Err(_) => fs::read_dir(root)
            .ok()
            .map(|entries| format!("nobin:{}|root:{root_part}", counted(entries))),
    }
}

/// The two entrypoints at the install root, which the walk above cannot see.
///
/// `_find_llama_server_binary` reaches `<root>/llama-server` before `build/bin`, and
/// `create_exec_entrypoint` writes a real wrapper there when it cannot make a symlink, so
/// its mode can move while `build/bin` stays byte for byte identical. That is what
/// `installed_runtime_health` now grades, and without it here the cached Ready survived the
/// damage and the CLI was never asked. Following links deliberately: the CLI follows them
/// too, and a dangling one is absent to both.
fn root_entrypoints(root: &Path) -> String {
    let ext = if cfg!(windows) { ".exe" } else { "" };
    let mut out = String::new();
    for name in ["llama-server", "llama-quantize"] {
        if !out.is_empty() {
            out.push(',');
        }
        match fs::metadata(root.join(format!("{name}{ext}"))) {
            Ok(meta) => {
                #[cfg(unix)]
                let mode = {
                    use std::os::unix::fs::PermissionsExt;
                    u64::from(meta.permissions().mode() & 0o7777)
                };
                #[cfg(not(unix))]
                let mode = 0u64;
                out.push_str(&format!("{}:{}:{mode}", u64::from(meta.is_file()), meta.len()));
            }
            // Absent, or a link whose target went. Not a pin either way, which is
            // what installed_runtime_health does with it.
            Err(_) => out.push('-'),
        }
    }
    out
}

/// How many files a directory holds, how many bytes they total, how many links
/// sit beside them and what their permission bits add up to. A subdirectory is
/// not a file, so it cannot read as a binary.
///
/// The last two counters are here because the first two answered the same for
/// trees the CLI grades differently. DirEntry::metadata does not follow a link,
/// so a link is not a file and neither its presence nor its loss moved a count
/// or a byte total, yet losing the install-name link (libggml.0.dylib, and the
/// SONAME link on Linux) is exactly what installed_runtime_health calls a broken
/// payload. Clearing the execute bit on llama-server moves neither the count nor
/// the length either, and that is the other thing health rejects. Either state
/// used to leave a cached Ready matching, so the repair was never offered and
/// the launch failed instead.
fn counted(entries: fs::ReadDir) -> String {
    let mut count: u64 = 0;
    let mut bytes: u64 = 0;
    let mut links: u64 = 0;
    let mut modes: u64 = 0;
    let mut names: u64 = 0;
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        // Every entry this walk counts, link or file, contributes its name. A
        // subdirectory still does not, matching the counters.
        if meta.is_symlink() {
            links += 1;
            names = names.wrapping_add(name_hash(&entry.file_name()));
            continue;
        }
        if meta.is_file() {
            count += 1;
            names = names.wrapping_add(name_hash(&entry.file_name()));
            bytes += meta.len();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                // The permission bits themselves, not a count of "has some execute
                // bit". os.access(X_OK) asks whether THIS user may run the file,
                // and the owner's bits decide that on their own: 0755 -> 0655
                // leaves group and other executable while the owner can no longer
                // run it, so a counter of any-bit-set never moved for the state
                // health rejects. Summed rather than folded, since read_dir order
                // is unspecified. Windows has no bits to read and stays at zero,
                // where the other three carry the tree.
                modes += u64::from(meta.permissions().mode() & 0o7777);
            }
        }
    }
    format!("{count}:{bytes}:{links}:{modes}:{names}")
}

/// FNV-1a over one entry's name, summed into the fingerprint by the caller.
///
/// Names are here because the four counters are all aggregates, and a rename in
/// place moves none of them: security software that renames ggml-base.dll to a
/// quarantine suffix beside itself leaves the count, the byte total, the link
/// count and the mode sum identical, so installed_runtime_health rejected the
/// tree while the cached Ready still matched and preflight answered Ready without
/// ever running the capability probe. The launch then failed with no repair
/// offered, which is the exact hole the runtime half of this fingerprint exists
/// to close.
///
/// Summed rather than folded in sequence, because read_dir order is unspecified
/// and the counters beside it are order-independent for that reason; sorting
/// would cost an allocation per entry on a walk that runs before the window
/// opens. FNV-1a rather than DefaultHasher so the value is stable across Rust
/// releases: a hash that moved on a toolchain bump would invalidate every cached
/// verdict on the first launch after an upgrade. A sum admits collisions in
/// principle, but the thing being detected is a rename, and a renamed file has to
/// collide with the name it replaced, not with any name.
fn name_hash(name: &std::ffi::OsStr) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in name.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

/// The fingerprint an answer may be stored under, or None when it may not be stored.
///
/// The one taken before the probe, and only while the tree still matches it. Reading
/// the fingerprint afresh afterwards instead meant a file quarantined while
/// desktop-capabilities was running was cached as healthy under its own damaged
/// fingerprint, which then matched on every later launch, so the CLI was never asked
/// again and the repair was never offered. A mismatch describes a tree that no longer
/// exists; there is nothing worth keeping and the next launch asks again.
fn fingerprint_to_cache_under<'a>(
    before: Option<&'a ManagedBinFingerprint>,
    after: Option<ManagedBinFingerprint>,
) -> Option<&'a ManagedBinFingerprint> {
    let before = before?;
    (after.as_ref() == Some(before)).then_some(before)
}

fn capability_cache_path() -> Option<PathBuf> {
    #[cfg(test)]
    if let Some(home) = std::env::var_os("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME") {
        return Some(
            PathBuf::from(home)
                .join(".unsloth")
                .join("studio")
                .join("desktop_capability_cache.json"),
        );
    }

    dirs::home_dir().map(|home| {
        home.join(".unsloth")
            .join("studio")
            .join("desktop_capability_cache.json")
    })
}

fn cache_matches(cache: &ManagedCapabilityCache, fingerprint: &ManagedBinFingerprint) -> bool {
    cache.schema == MANAGED_CAPABILITY_CACHE_SCHEMA
        && cache.desktop_protocol_version == DESKTOP_PROTOCOL_VERSION
        && cache.desktop_manageability_version == DESKTOP_MANAGEABILITY_VERSION
        && cache.bin_path == fingerprint.bin_path
        && cache.bin_size == fingerprint.bin_size
        && cache.bin_mtime_ms == fingerprint.bin_mtime_ms
        && cache.studio_root_id == fingerprint.studio_root_id
        && cache.marker_path == fingerprint.marker_path
        && cache.marker_size == fingerprint.marker_size
        && cache.marker_mtime_ms == fingerprint.marker_mtime_ms
        && cache.llama_runtime == fingerprint.llama_runtime
        && desktop_capability_ready(&cache.capability)
}

fn read_cached_capability(fingerprint: &ManagedBinFingerprint) -> Option<DesktopCapability> {
    let path = capability_cache_path()?;
    let bytes = fs::read(path).ok()?;
    let cache = serde_json::from_slice::<ManagedCapabilityCache>(&bytes).ok()?;
    if cache_matches(&cache, fingerprint) {
        Some(cache.capability)
    } else {
        None
    }
}

/// The CLI's word for "another runtime is the active one, so the managed tree is
/// not mine to grade". Its null verdict is about a selection, not about the tree.
const LLAMA_RUNTIME_NOT_MANAGED: &str = "llama_runtime_not_managed";

/// And its word for "the probe itself raised", which is about one attempt.
const LLAMA_RUNTIME_PROBE_FAILED: &str = "llama_runtime_probe_failed";

/// Whether the runtime verdict in this answer was skipped rather than reached.
///
/// Both reasons mean no verdict was reached, and neither is a fact this fingerprint
/// watches: the selection lives in the settings database and the environment, and a
/// probe that raised shares the damaged tree's fingerprint exactly. The third null,
/// nothing installed at all, is a fact about the tree and stays cacheable.
fn llama_runtime_verdict_was_skipped(capability: &DesktopCapability) -> bool {
    capability.llama_runtime_ok.is_none()
        && matches!(
            capability.llama_runtime_reason.as_deref(),
            Some(LLAMA_RUNTIME_NOT_MANAGED) | Some(LLAMA_RUNTIME_PROBE_FAILED)
        )
}

fn write_cached_capability(fingerprint: &ManagedBinFingerprint, capability: &DesktopCapability) {
    if llama_runtime_verdict_was_skipped(capability) {
        // Nothing in this fingerprint watches which runtime is selected: the
        // stored custom folder lives in the settings database and LLAMA_SERVER_PATH
        // in the environment. Caching the skip meant that clearing the selection
        // made a damaged managed tree active with its fingerprint unchanged, so
        // the cache kept serving the Ready it was never graded for and repair was
        // never offered. Re-probing costs one CLI call, and only for the users who
        // run their own build.
        return;
    }
    let Some(path) = capability_cache_path() else {
        return;
    };
    let cache = ManagedCapabilityCache {
        schema: MANAGED_CAPABILITY_CACHE_SCHEMA,
        bin_path: fingerprint.bin_path.clone(),
        bin_size: fingerprint.bin_size,
        bin_mtime_ms: fingerprint.bin_mtime_ms,
        studio_root_id: fingerprint.studio_root_id.clone(),
        marker_path: fingerprint.marker_path.clone(),
        marker_size: fingerprint.marker_size,
        marker_mtime_ms: fingerprint.marker_mtime_ms,
        llama_runtime: fingerprint.llama_runtime.clone(),
        desktop_protocol_version: DESKTOP_PROTOCOL_VERSION,
        desktop_manageability_version: DESKTOP_MANAGEABILITY_VERSION,
        capability: capability.clone(),
    };
    if let Some(parent) = path.parent() {
        if fs::create_dir_all(parent).is_err() {
            return;
        }
    }
    let Ok(bytes) = serde_json::to_vec_pretty(&cache) else {
        return;
    };
    if let Err(error) = fs::write(&path, bytes) {
        warn!(
            "Managed preflight: could not write capability cache: {}",
            error
        );
    }
}

async fn run_cli_probe(bin: &Path, args: &[&str]) -> Result<bool, String> {
    let started = Instant::now();
    let Ok(mut cmd) = crate::process::build_managed_cli_command_tokio(bin, args) else {
        info!(
            "Managed preflight probe {:?} has no managed interpreter to run",
            args
        );
        // Ok, not Err: main's Err arm means "the probe could not be set up and the
        // install is untested". A venv with no interpreter beside the launcher IS a
        // result, and the same one this arm always gave.
        return Ok(false);
    };
    cmd.stdout(Stdio::null()).stderr(Stdio::null());

    // Reported, not folded into `false`: the CLI never ran, so calling it broken
    // would start a repair needing the same context. Re-checking afterwards is not
    // enough: a context that recovers in between makes an untested install look bad.
    if let Err(error) = crate::process::apply_managed_cli_context_tokio(&mut cmd) {
        info!(
            "Managed preflight probe {:?} has no usable working directory: {}",
            args, error
        );
        return Err(error);
    }

    #[cfg(target_os = "linux")]
    crate::process::scrub_appimage_python_env_tokio(&mut cmd);

    // Tauri uses the legacy root regardless of UNSLOTH_STUDIO_HOME / STUDIO_HOME;
    // probe subprocesses must follow the same isolation as process.rs.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
    }

    let Ok(mut child) = crate::process::with_studio_runtime_launch_guard(|| {
        cmd.spawn().map_err(|error| error.to_string())
    }) else {
        info!(
            "Managed preflight probe {:?} failed to spawn in {}ms",
            args,
            started.elapsed().as_millis()
        );
        return Ok(false);
    };

    let ok = match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
        Ok(Ok(status)) => status.success(),
        _ => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            false
        }
    };
    info!(
        "Managed preflight probe {:?} finished ok={} in {}ms",
        args,
        ok,
        started.elapsed().as_millis()
    );
    Ok(ok)
}

async fn probe_cli_capability(bin: &Path) -> Result<Option<DesktopCapability>, String> {
    let started = Instant::now();
    let Ok(mut cmd) = crate::process::build_managed_cli_command_tokio(
        bin,
        &["studio", "desktop-capabilities", "--json"],
    ) else {
        info!("Managed desktop-capabilities probe has no managed interpreter to run");
        // As above: a missing interpreter is a verdict, not a failure to ask.
        return Ok(None);
    };
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());

    // As above: a context that cannot be built is not a probe result.
    if let Err(error) = crate::process::apply_managed_cli_context_tokio(&mut cmd) {
        info!(
            "Managed desktop-capabilities probe has no usable working directory: {}",
            error
        );
        return Err(error);
    }

    #[cfg(target_os = "linux")]
    crate::process::scrub_appimage_python_env_tokio(&mut cmd);

    // Tauri uses the legacy root regardless of UNSLOTH_STUDIO_HOME / STUDIO_HOME;
    // probe subprocesses must follow the same isolation as process.rs.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
    }

    let Ok(mut child) = crate::process::with_studio_runtime_launch_guard(|| {
        cmd.spawn().map_err(|error| error.to_string())
    }) else {
        info!(
            "Managed desktop-capabilities probe failed to spawn in {}ms",
            started.elapsed().as_millis()
        );
        return Ok(None);
    };
    let Some(mut stdout) = child.stdout.take() else {
        return Ok(None);
    };

    match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
        Ok(Ok(status)) if status.success() => {}
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            info!(
                "Managed desktop-capabilities probe timed out in {}ms",
                started.elapsed().as_millis()
            );
            return Ok(None);
        }
        _ => {
            info!(
                "Managed desktop-capabilities probe exited unsuccessfully in {}ms",
                started.elapsed().as_millis()
            );
            return Ok(None);
        }
    }

    let mut output = Vec::new();
    if stdout.read_to_end(&mut output).await.is_err() {
        return Ok(None);
    }

    let capability = serde_json::from_slice::<DesktopCapability>(&output).ok();
    info!(
        "Managed desktop-capabilities probe finished ok={} in {}ms",
        capability.is_some(),
        started.elapsed().as_millis()
    );
    Ok(capability)
}

fn desktop_capability_stale_reason(capability: &DesktopCapability) -> Option<String> {
    if capability.desktop_protocol_version != Some(DESKTOP_PROTOCOL_VERSION) {
        return Some("desktop_protocol_incompatible".to_string());
    }
    if capability.supports_api_only != Some(true) {
        return Some("desktop_api_only_unsupported".to_string());
    }
    if capability.supports_provision_desktop_auth != Some(true) {
        return capability
            .desktop_auth_stale_reason
            .clone()
            .or_else(|| Some("desktop_auth_unsupported".to_string()));
    }
    if capability.desktop_manageability_version.unwrap_or(0) < DESKTOP_MANAGEABILITY_VERSION {
        return Some("desktop_manageability_unsupported".to_string());
    }
    if capability.supports_desktop_backend_ownership != Some(true) {
        return Some("desktop_backend_ownership_unsupported".to_string());
    }
    // Half-installed is Stale, not Ready: starting the backend just crashes it.
    // A CLI too old to answer is already rejected above on manageability.
    if capability.studio_install_ok != Some(true) {
        return Some(
            capability
                .studio_install_reason
                .clone()
                .unwrap_or_else(|| "studio_install_incomplete".to_string()),
        );
    }
    // Only an explicit false. None means the CLI predates the field or nothing is
    // installed yet, and calling either stale would put every older install into
    // repair on its first launch after an upgrade.
    if capability.llama_runtime_ok == Some(false) {
        return Some(
            capability
                .llama_runtime_reason
                .clone()
                .filter(|reason| !reason.is_empty())
                .unwrap_or_else(|| "llama_runtime_incomplete".to_string()),
        );
    }
    managed_backend_version_stale_reason(capability.version.as_deref())
}

fn desktop_capability_ready(capability: &DesktopCapability) -> bool {
    desktop_capability_stale_reason(capability).is_none()
}

/// The reason an unbuildable context is reported under, if that is what went
/// wrong. Checked after a probe that did run and failed anyway.
fn working_directory_reason() -> Option<String> {
    // The whole context: an unresolvable override fails the same spawn, and is a
    // different thing to fix.
    let error = crate::process::managed_cli_context_error()?;
    info!("Managed preflight: managed context unavailable: {error}");
    Some(context_reason(&error))
}

pub(super) async fn probe_managed_bin(bin: PathBuf) -> ManagedProbe {
    let started = Instant::now();

    // An unmounted roaming profile fails every probe below, which is not a broken
    // install: "cli_unusable" would start a repair needing the same directory.
    if let Some(error) = crate::process::managed_cli_context_error() {
        info!(
            "Managed preflight: no usable managed context for {:?}: {}",
            bin, error
        );
        return ManagedProbe::Stale {
            bin,
            reason: context_reason(&error),
        };
    }

    // Always verify the managed CLI actually launches before trusting the cache.
    // A matching capability fingerprint does not prove the binary can still run:
    // its venv interpreter or a runtime dependency can be broken while the
    // path/size/mtime/markers are unchanged, so the -h probe runs first and a
    // non-launchable install is reported Stale for repair. The capability cache
    // below still skips the heavier desktop-capabilities probe on a hit.
    match run_cli_probe(&bin, &["-h"]).await {
        // The CLI was never asked, so do not report a broken install.
        Err(_) => {
            info!(
                "Managed preflight: no usable managed context for {:?} in {}ms",
                bin,
                started.elapsed().as_millis()
            );
            return ManagedProbe::Stale {
                bin,
                reason: working_directory_reason()
                    .unwrap_or_else(|| WORKING_DIRECTORY_UNAVAILABLE.to_string()),
            };
        }
        Ok(false) => {
            info!(
                "Managed preflight: cli unusable for {:?} in {}ms",
                bin,
                started.elapsed().as_millis()
            );
            // The profile can drop between the check above and the probe, so ask again.
            return ManagedProbe::Stale {
                bin,
                reason: working_directory_reason().unwrap_or_else(|| "cli_unusable".to_string()),
            };
        }
        Ok(true) => {}
    }

    // Taken before the probe and kept, so the answer is stored against the tree the
    // CLI actually read. Re-reading it afterwards instead let a file quarantined
    // during the probe be cached as healthy under its own damaged fingerprint, which
    // then matched on every later launch and the CLI was never asked again.
    let fingerprint_before = managed_bin_fingerprint(&bin);
    if let Some(fingerprint) = fingerprint_before.as_ref() {
        if read_cached_capability(fingerprint).is_some() {
            info!(
                "Managed preflight: using cached desktop capability for {:?} in {}ms",
                bin,
                started.elapsed().as_millis()
            );
            return ManagedProbe::Ready { bin };
        }
    }

    let capability = match probe_cli_capability(&bin).await {
        Ok(capability) => capability,
        Err(_) => {
            info!(
                "Managed preflight: no usable managed context for {:?} in {}ms",
                bin,
                started.elapsed().as_millis()
            );
            return ManagedProbe::Stale {
                bin,
                reason: working_directory_reason()
                    .unwrap_or_else(|| WORKING_DIRECTORY_UNAVAILABLE.to_string()),
            };
        }
    };
    if let Some(capability) = capability {
        if let Some(fingerprint) = fingerprint_to_cache_under(
            fingerprint_before.as_ref(),
            managed_bin_fingerprint(&bin),
        ) {
            write_cached_capability(fingerprint, &capability);
        }
        if desktop_capability_ready(&capability) {
            info!(
                "Managed preflight: cli ready for {:?} in {}ms",
                bin,
                started.elapsed().as_millis()
            );
            return ManagedProbe::Ready { bin };
        }
        info!(
            "Managed preflight: cli stale for {:?} in {}ms",
            bin,
            started.elapsed().as_millis()
        );
        return ManagedProbe::Stale {
            bin,
            reason: desktop_capability_stale_reason(&capability)
                .unwrap_or_else(|| "desktop_capability_incompatible".to_string()),
        };
    }

    info!(
        "Managed preflight: desktop capability probe failed for {:?} in {}ms",
        bin,
        started.elapsed().as_millis()
    );
    ManagedProbe::Stale {
        bin,
        reason: working_directory_reason()
            .unwrap_or_else(|| "desktop_capability_probe_failed".to_string()),
    }
}

pub(super) async fn probe_managed_install() -> ManagedProbe {
    let started = Instant::now();
    let result = match crate::process::find_unsloth_binary() {
        Some(bin) => probe_managed_bin(bin).await,
        // The managed install lives under the profile, so an unreachable one looks
        // like no install. Say which, or a late network profile sends them to reinstall.
        None => match crate::process::home_dir_available() {
            Ok(()) => ManagedProbe::Missing,
            Err(error) => {
                info!("Managed preflight: {}", error);
                ManagedProbe::Unavailable {
                    reason: WORKING_DIRECTORY_UNAVAILABLE.to_string(),
                }
            }
        },
    };
    info!(
        "Managed preflight: install probe result {:?} in {}ms",
        result,
        started.elapsed().as_millis()
    );
    result
}

pub async fn managed_install_ready() -> bool {
    matches!(probe_managed_install().await, ManagedProbe::Ready { .. })
}

#[cfg(test)]
mod tests {
    use super::super::version::MIN_DESKTOP_BACKEND_VERSION;
    use super::*;

    fn healthy_capability() -> DesktopCapability {
        DesktopCapability {
            desktop_protocol_version: Some(DESKTOP_PROTOCOL_VERSION),
            desktop_manageability_version: Some(DESKTOP_MANAGEABILITY_VERSION),
            supports_api_only: Some(true),
            supports_provision_desktop_auth: Some(true),
            supports_desktop_backend_ownership: Some(true),
            desktop_auth_stale_reason: None,
            studio_install_ok: Some(true),
            studio_install_reason: None,
            llama_runtime_ok: Some(true),
            llama_runtime_reason: None,
            version: Some(MIN_DESKTOP_BACKEND_VERSION.to_string()),
        }
    }

    #[test]
    fn complete_install_is_ready() {
        assert_eq!(desktop_capability_stale_reason(&healthy_capability()), None);
        assert!(desktop_capability_ready(&healthy_capability()));
    }

    #[test]
    fn a_venv_behind_the_desktop_backend_version_is_stale() {
        // The CLI installer shares this venv, so its package version is the only
        // thing that pulls an old-but-launchable install forward via repair.
        let mut capability = healthy_capability();
        capability.version = Some("2026.5.2".to_string());
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("desktop_backend_version_too_old")
        );
        assert!(!desktop_capability_ready(&capability));
    }

    #[test]
    fn incomplete_install_is_stale_with_the_cli_reason() {
        // The venv has the CLI but not structlog, so preflight must repair
        // rather than spawn a backend that cannot import.
        let mut capability = healthy_capability();
        capability.studio_install_ok = Some(false);
        capability.studio_install_reason = Some("studio_install_incomplete".to_string());
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("studio_install_incomplete")
        );
        assert!(!desktop_capability_ready(&capability));
    }

    #[test]
    fn deps_removed_after_install_is_stale() {
        let mut capability = healthy_capability();
        capability.studio_install_ok = Some(false);
        capability.studio_install_reason = Some("studio_deps_missing".to_string());
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("studio_deps_missing")
        );
    }

    #[test]
    fn missing_install_field_falls_back_to_a_generic_reason() {
        let mut capability = healthy_capability();
        capability.studio_install_ok = None;
        capability.studio_install_reason = None;
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("studio_install_incomplete")
        );
    }

    #[test]
    fn a_quarantined_llama_runtime_is_stale() {
        // The venv is fine and the marker says installed; quarantine took a DLL out
        // of the tree. Repairing here stops it surfacing as a model load failure.
        let mut capability = healthy_capability();
        capability.llama_runtime_ok = Some(false);
        capability.llama_runtime_reason = Some("llama_runtime_payload_incomplete".to_string());
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("llama_runtime_payload_incomplete")
        );
        assert!(!desktop_capability_ready(&capability));
    }

    #[test]
    fn a_broken_runtime_without_a_reason_falls_back_to_a_generic_one() {
        let mut capability = healthy_capability();
        capability.llama_runtime_ok = Some(false);
        capability.llama_runtime_reason = Some(String::new());
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("llama_runtime_incomplete")
        );
    }

    #[test]
    fn an_unknown_llama_runtime_is_not_stale() {
        // None is both "no runtime yet" and "this CLI predates the field", and
        // calling either stale would send every install through repair.
        let mut capability = healthy_capability();
        capability.llama_runtime_ok = None;
        capability.llama_runtime_reason = None;
        assert_eq!(desktop_capability_stale_reason(&capability), None);
        assert!(desktop_capability_ready(&capability));
    }

    /// The payload a CLI without this PR prints. JSON, not a struct literal, which
    /// cannot show that a missing key deserializes.
    fn pre_pr_capability_json() -> String {
        format!(
            r#"{{
              "desktop_protocol_version": {protocol},
              "desktop_manageability_version": {manageability},
              "supports_api_only": true,
              "supports_provision_desktop_auth": true,
              "supports_desktop_backend_ownership": true,
              "desktop_auth_stale_reason": null,
              "studio_install_ok": true,
              "studio_install_reason": null,
              "version": "{version}"
            }}"#,
            protocol = DESKTOP_PROTOCOL_VERSION,
            manageability = DESKTOP_MANAGEABILITY_VERSION,
            version = MIN_DESKTOP_BACKEND_VERSION,
        )
    }

    #[test]
    fn a_capability_payload_without_the_new_keys_still_parses_and_is_ready() {
        // Every install whose CLI predates this prints a payload with neither key.
        // Failing to parse, or parsing to something stale, would send every
        // existing user into repair on their next launch.
        let capability: DesktopCapability =
            serde_json::from_str(&pre_pr_capability_json()).expect("pre-PR payload must parse");
        assert_eq!(capability.llama_runtime_ok, None);
        assert_eq!(capability.llama_runtime_reason, None);
        assert_eq!(desktop_capability_stale_reason(&capability), None);
        assert!(desktop_capability_ready(&capability));
    }

    #[test]
    fn a_payload_from_a_newer_cli_ignores_keys_this_desktop_does_not_know() {
        // The CLI and the desktop shell update separately, so a newer CLI can answer
        // an older desktop. An unknown key must be ignored, not fatal.
        let json = pre_pr_capability_json().replace(
            "\"studio_install_ok\": true,",
            "\"studio_install_ok\": true, \"a_field_from_the_future\": {\"nested\": [1]},",
        );
        let capability: DesktopCapability =
            serde_json::from_str(&json).expect("an unknown key must not be fatal");
        assert!(desktop_capability_ready(&capability));
    }

    #[test]
    fn a_null_llama_runtime_ok_is_not_a_broken_runtime() {
        // What the CLI prints when the probe itself failed, which is not a verdict.
        // Explicit nulls, not absent keys, since the CLI seeds both before trying.
        let json = pre_pr_capability_json().replace(
            "\"studio_install_ok\": true,",
            "\"llama_runtime_ok\": null, \"llama_runtime_reason\": \"\", \"studio_install_ok\": true,",
        );
        let capability: DesktopCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(desktop_capability_stale_reason(&capability), None);
    }

    #[test]
    fn a_broken_runtime_survives_the_json_round_trip() {
        // The frontend switches its message on this exact string.
        let json = pre_pr_capability_json().replace(
            "\"studio_install_ok\": true,",
            "\"llama_runtime_ok\": false, \"llama_runtime_reason\": \"llama_runtime_binaries_missing\", \"studio_install_ok\": true,",
        );
        let capability: DesktopCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("llama_runtime_binaries_missing")
        );
    }

    #[test]
    fn a_cache_file_written_before_this_field_is_ignored_rather_than_fatal() {
        // On disk in every existing install. Schema 3 predates the runtime
        // fingerprint, so its Ready was reached without looking at the runtime: it
        // must miss and be rewritten, not panic and not be served.
        let old = format!(
            r#"{{
              "schema": 3,
              "bin_path": "/managed/unsloth",
              "bin_size": 1,
              "bin_mtime_ms": 1,
              "studio_root_id": null,
              "marker_path": null,
              "marker_size": null,
              "marker_mtime_ms": null,
              "desktop_protocol_version": {protocol},
              "desktop_manageability_version": {manageability},
              "capability": {capability}
            }}"#,
            protocol = DESKTOP_PROTOCOL_VERSION,
            manageability = DESKTOP_MANAGEABILITY_VERSION,
            capability = pre_pr_capability_json(),
        );
        let cache: ManagedCapabilityCache =
            serde_json::from_str(&old).expect("an old cache file must still parse");
        assert_eq!(cache.llama_runtime, None);
        assert_ne!(cache.schema, MANAGED_CAPABILITY_CACHE_SCHEMA);

        let fingerprint = ManagedBinFingerprint {
            bin_path: "/managed/unsloth".to_string(),
            bin_size: 1,
            bin_mtime_ms: 1,
            studio_root_id: None,
            marker_path: None,
            marker_size: None,
            marker_mtime_ms: None,
            llama_runtime: Some("12:345".to_string()),
        };
        assert!(
            !cache_matches(&cache, &fingerprint),
            "a cache from before the runtime was fingerprinted must not be served"
        );
    }

    #[test]
    fn a_quarantined_runtime_file_changes_the_fingerprint() {
        // Without the runtime in the fingerprint the cache hits, preflight answers
        // Ready from it, and the CLI is never asked whether the runtime is intact.
        let root = std::env::temp_dir().join(format!(
            "unsloth-llama-fingerprint-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        let mut bin = root.join("build").join("bin");
        if cfg!(windows) {
            bin = bin.join("Release");
        }
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("llama.dll"), vec![0u8; 1024]).unwrap();
        fs::write(bin.join("ggml-base.dll"), vec![0u8; 2048]).unwrap();

        let intact =
            llama_runtime_fingerprint_at(&root).expect("an installed runtime must fingerprint");
        assert_eq!(
            intact,
            llama_runtime_fingerprint_at(&root).unwrap(),
            "the same tree must fingerprint the same twice, or every launch misses its own cache"
        );

        fs::remove_file(bin.join("ggml-base.dll")).unwrap();
        let quarantined = llama_runtime_fingerprint_at(&root).unwrap();
        assert_ne!(intact, quarantined);

        // A directory is not a file: a stray subfolder must not read as a binary.
        fs::create_dir(bin.join("some-subdir")).unwrap();
        assert_eq!(quarantined, llama_runtime_fingerprint_at(&root).unwrap());

        // A file replaced by one of a different size is caught by the byte total
        // even though the count is unchanged.
        fs::write(bin.join("llama.dll"), vec![0u8; 4096]).unwrap();
        assert_ne!(quarantined, llama_runtime_fingerprint_at(&root).unwrap());

        // The whole tree going is distinct from an empty one: only one of the two
        // means nothing was ever installed.
        fs::remove_dir_all(&root).unwrap();
        assert_eq!(llama_runtime_fingerprint_at(&root), None);
        fs::create_dir_all(&bin).unwrap();
        assert_eq!(
            llama_runtime_fingerprint_at(&root).as_deref(),
            Some("bin:0:0:0:0:0|root:-,-")
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_marker_left_over_a_missing_build_dir_changes_the_fingerprint() {
        // The transition a read_dir(bin).ok()? alone cannot see: nothing installed
        // and a marker over a tree with no build/bin both fail that read, so both
        // fingerprinted as None, while the CLI's verdict moves from null (not
        // stale) to llama_runtime_dir_missing. The cached Ready outlived it.
        let root = std::env::temp_dir().join(format!(
            "unsloth-llama-nobin-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);

        assert_eq!(
            llama_runtime_fingerprint_at(&root),
            None,
            "no tree at all is the NotInstalled case, and stays None"
        );

        fs::create_dir_all(&root).unwrap();
        let empty_tree = llama_runtime_fingerprint_at(&root);
        assert!(
            empty_tree.is_some(),
            "a runtime root with no build/bin is a broken install, not an absent one"
        );

        fs::write(root.join("unsloth_llama_prebuilt.json"), b"{}").unwrap();
        assert_ne!(
            empty_tree,
            llama_runtime_fingerprint_at(&root),
            "a marker arriving over a missing build/bin must invalidate the cached Ready"
        );

        // Still distinct from a tree whose build/bin exists and is empty.
        let mut bin = root.join("build").join("bin");
        if cfg!(windows) {
            bin = bin.join("Release");
        }
        fs::create_dir_all(&bin).unwrap();
        assert_ne!(
            llama_runtime_fingerprint_at(&root),
            empty_tree,
            "an empty binary directory is not the same state as a missing one"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_runtime_override_is_resolved_the_way_python_resolves_it() {
        // default_managed_llama_dir() strips the value and calls expanduser. Reading
        // it raw would fingerprint a folder literally named "~" and silently drop the
        // runtime out for every user who wrote the override that way.
        let home = PathBuf::from(if cfg!(windows) {
            "C:\\Users\\me"
        } else {
            "/home/me"
        });
        let cwd = PathBuf::from(if cfg!(windows) { "C:\\work" } else { "/work" });
        let absolute = if cfg!(windows) {
            "C:\\opt\\llama.cpp"
        } else {
            "/opt/llama.cpp"
        };
        let cases: Vec<(Option<&str>, Option<PathBuf>)> = vec![
            (None, None),
            (Some(""), None),
            (Some("   "), None),
            (Some("~"), Some(home.clone())),
            (Some("~/llama.cpp"), Some(home.join("llama.cpp"))),
            (Some("  ~/llama.cpp  "), Some(home.join("llama.cpp"))),
            (Some(absolute), Some(PathBuf::from(absolute))),
            // Not expanded, since nothing here can resolve another user's home, so
            // it is anchored like any other relative value.
            (
                Some("~someone/llama.cpp"),
                Some(cwd.join("~someone/llama.cpp")),
            ),
            // Anchored to this process's directory, the one
            // relative_override_pins joins the child's copy against.
            (Some("llama.cpp"), Some(cwd.join("llama.cpp"))),
            (Some("./llama.cpp"), Some(cwd.join("./llama.cpp"))),
            (Some("../llama.cpp"), Some(cwd.join("../llama.cpp"))),
        ];
        for (value, expected) in cases {
            assert_eq!(
                llama_runtime_override_from(value, Some(&home), Some(&cwd)),
                expected,
                "value {value:?}"
            );
        }
    }

    #[test]
    fn an_override_that_cannot_be_anchored_is_dropped_rather_than_guessed() {
        // No home to expand against and no directory to anchor to. Guessing would
        // make the cached verdict track a folder the child was never pointed at.
        assert_eq!(llama_runtime_override_from(Some("~/x"), None, None), None);
        assert_eq!(llama_runtime_override_from(Some("x"), None, None), None);
        // An absolute value needs neither, so it still resolves.
        let absolute = if cfg!(windows) {
            "C:\\opt\\llama.cpp"
        } else {
            "/opt/llama.cpp"
        };
        assert_eq!(
            llama_runtime_override_from(Some(absolute), None, None),
            Some(PathBuf::from(absolute))
        );
    }

    #[cfg(windows)]
    #[test]
    fn the_windows_tilde_resolves_through_the_same_profile_the_child_uses() {
        // ntpath.expanduser answers USERPROFILE, while dirs::home_dir() reads the
        // known folder, which a portable or overridden profile moves. Taking the
        // known folder would fingerprint a tree the child never looks at.
        // process.rs carries the same note at its own read.
        let _guard = crate::native_path_policy::PROCESS_ENV_LOCK.lock();
        let previous = std::env::var_os("USERPROFILE");
        std::env::set_var("USERPROFILE", "C:\\Portable\\Profile");
        let resolved = tilde_home();
        match previous {
            Some(value) => std::env::set_var("USERPROFILE", value),
            None => std::env::remove_var("USERPROFILE"),
        }
        assert_eq!(resolved, Some(PathBuf::from("C:\\Portable\\Profile")));
    }

    #[test]
    fn older_cli_is_rejected_on_manageability_before_the_install_check() {
        // A CLI predating this feature cannot answer studio_install_ok, so the
        // more specific manageability reason must win in the diagnostics.
        let mut capability = healthy_capability();
        capability.desktop_manageability_version = Some(1);
        capability.studio_install_ok = None;
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("desktop_manageability_unsupported")
        );
    }

    #[test]
    fn a_stale_capability_is_never_served_from_cache() {
        // write_cached_capability runs before the ready check, so an incomplete
        // install does get cached; reusing it would outlive the repair.
        let mut capability = healthy_capability();
        capability.studio_install_ok = Some(false);
        let cache = ManagedCapabilityCache {
            schema: MANAGED_CAPABILITY_CACHE_SCHEMA,
            bin_path: "/managed/unsloth".to_string(),
            bin_size: 1,
            bin_mtime_ms: 1,
            studio_root_id: None,
            marker_path: None,
            marker_size: None,
            marker_mtime_ms: None,
            llama_runtime: None,
            desktop_protocol_version: DESKTOP_PROTOCOL_VERSION,
            desktop_manageability_version: DESKTOP_MANAGEABILITY_VERSION,
            capability,
        };
        let fingerprint = ManagedBinFingerprint {
            bin_path: "/managed/unsloth".to_string(),
            bin_size: 1,
            bin_mtime_ms: 1,
            studio_root_id: None,
            marker_path: None,
            marker_size: None,
            marker_mtime_ms: None,
            llama_runtime: None,
        };
        assert!(!cache_matches(&cache, &fingerprint));
    }

    // Quarantine deletes the generated unsloth.exe, so the supported stubless layout
    // hands back a launcher path that is not on disk. Without a stand-in the
    // fingerprint is None, the capability cache can be neither read nor written, and
    // every preflight pays both probe subprocesses again.
    #[cfg(windows)]
    #[test]
    fn a_quarantined_launcher_is_fingerprinted_through_its_interpreter() {
        let venv = std::env::temp_dir().join(format!(
            "unsloth-fingerprint-quarantined-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let scripts = venv.join("Scripts");
        fs::create_dir_all(&scripts).unwrap();
        let bin = scripts.join("unsloth.exe");
        let interpreter = scripts.join("python.exe");
        fs::write(&interpreter, "python").unwrap();
        assert!(!bin.exists(), "this case is about the launcher being gone");

        let fingerprint = managed_bin_fingerprint(&bin)
            .expect("a stubless venv must still fingerprint, through python.exe");
        // The identity stays the launcher path, so the two layouts of one install
        // cannot share a cache entry.
        assert!(fingerprint.bin_path.ends_with("unsloth.exe"));
        // And it tracks the interpreter, so a venv replaced by an update invalidates.
        fs::write(&interpreter, "python-after-an-update").unwrap();
        let after = managed_bin_fingerprint(&bin).unwrap();
        assert_ne!(fingerprint.bin_size, after.bin_size);

        // With no interpreter either there is nothing to stand in, and None is right.
        fs::remove_file(&interpreter).unwrap();
        assert!(managed_bin_fingerprint(&bin).is_none());

        let _ = fs::remove_dir_all(&venv);
    }

    #[test]
    fn dropping_the_manifest_changes_the_fingerprint() {
        // Otherwise a cache entry written while healthy outlives the manifest,
        // and the probe returns Ready on the very venv this is meant to catch.
        let venv = std::env::temp_dir().join(format!(
            "unsloth-fingerprint-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let scripts = venv.join("bin");
        fs::create_dir_all(&scripts).unwrap();
        let bin = scripts.join("unsloth");
        fs::write(&bin, "#!/bin/sh\nexit 0\n").unwrap();
        let manifest = venv.join("unsloth_install_manifest.json");
        fs::write(&manifest, "{}").unwrap();

        let with_manifest = managed_bin_fingerprint(&bin).unwrap();
        fs::remove_file(&manifest).unwrap();
        let without_manifest = managed_bin_fingerprint(&bin).unwrap();

        assert_ne!(with_manifest, without_manifest);
        let _ = fs::remove_dir_all(&venv);
    }

    fn cache_for(fingerprint: &ManagedBinFingerprint) -> ManagedCapabilityCache {
        ManagedCapabilityCache {
            schema: MANAGED_CAPABILITY_CACHE_SCHEMA,
            bin_path: fingerprint.bin_path.clone(),
            bin_size: fingerprint.bin_size,
            bin_mtime_ms: fingerprint.bin_mtime_ms,
            studio_root_id: fingerprint.studio_root_id.clone(),
            marker_path: fingerprint.marker_path.clone(),
            marker_size: fingerprint.marker_size,
            marker_mtime_ms: fingerprint.marker_mtime_ms,
            llama_runtime: fingerprint.llama_runtime.clone(),
            desktop_protocol_version: DESKTOP_PROTOCOL_VERSION,
            desktop_manageability_version: DESKTOP_MANAGEABILITY_VERSION,
            capability: healthy_capability(),
        }
    }

    #[test]
    fn losing_a_studio_package_changes_the_fingerprint() {
        // pip uninstall rewrites no fingerprinted file: the manifest, pyvenv.cfg
        // and the launcher survive and `unsloth -h` still exits 0. Without the
        // installed distributions in the fingerprint the healthy answer sticks.
        let venv = std::env::temp_dir().join(format!(
            "unsloth-fingerprint-deps-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&venv);
        let scripts = venv.join("bin");
        fs::create_dir_all(&scripts).unwrap();
        let bin = scripts.join("unsloth");
        fs::write(&bin, "#!/bin/sh\nexit 0\n").unwrap();
        fs::write(venv.join("pyvenv.cfg"), "home = /usr/bin\n").unwrap();
        fs::write(venv.join("unsloth_install_manifest.json"), "{}").unwrap();

        // site_packages_dirs() only walks lib/<pyver>/site-packages on unix; on
        // Windows it looks at Lib/site-packages. Building the posix layout
        // everywhere left the dist-info invisible to the fingerprint on Windows,
        // so removing it changed nothing and the assert_ne below could not hold.
        let site_packages = if cfg!(windows) {
            venv.join("Lib").join("site-packages")
        } else {
            venv.join("lib").join("python3.11").join("site-packages")
        };
        fs::create_dir_all(site_packages.join("unsloth_cli").join("commands")).unwrap();
        fs::write(
            site_packages
                .join("unsloth_cli")
                .join("commands")
                .join("studio.py"),
            "# cli\n",
        )
        .unwrap();
        let dist_info = site_packages.join("fastmcp-3.0.2.dist-info");
        fs::create_dir_all(&dist_info).unwrap();
        fs::write(dist_info.join("METADATA"), "Name: fastmcp\n").unwrap();

        let with_dep = managed_bin_fingerprint(&bin).unwrap();
        let healthy_cache = cache_for(&with_dep);
        // read_dir order is unspecified, so an unsorted walk would miss its own
        // cache every launch and the entry would never be worth writing.
        assert_eq!(with_dep, managed_bin_fingerprint(&bin).unwrap());
        assert!(cache_matches(&healthy_cache, &with_dep));

        fs::remove_dir_all(&dist_info).unwrap();
        let without_dep = managed_bin_fingerprint(&bin).unwrap();

        assert_ne!(with_dep, without_dep);
        assert!(
            !cache_matches(&healthy_cache, &without_dep),
            "a removed studio package must not keep serving the cached Ready answer"
        );
        let _ = fs::remove_dir_all(&venv);
    }

    #[test]
    fn a_context_reason_names_the_setting_it_could_not_preserve() {
        use crate::process::ManagedContextError;
        // Every pin failure names the setting first, and the window needs that
        // name: "one of Unsloth's folder settings" is not something to act on.
        let reason = context_reason(&ManagedContextError::PathSetting(
            "HF_HOME names a path this machine cannot resolve".to_string(),
        ));
        assert_eq!(reason, "path_setting_unresolvable:HF_HOME");
        assert!(is_context_reason(&reason));
        // The name is carried, never the value or the sentence around it.
        assert!(!reason.contains("cannot resolve"));
        // A failure that does not start with a setting name still classifies.
        let bare = context_reason(&ManagedContextError::PathSetting(
            "the environment block is too long".to_string(),
        ));
        assert_eq!(bare, PATH_SETTING_UNRESOLVABLE);
        assert!(is_context_reason(&bare));
        assert!(is_context_reason(WORKING_DIRECTORY_UNAVAILABLE));
        assert!(!is_context_reason("cli_unusable"));
    }

    /// A scratch directory of this test's own, emptied before it is handed back.
    ///
    /// These tests walk real files on parallel threads, so two sharing one path
    /// would delete each other's tree mid-walk. Process id plus thread id keys it,
    /// as above, and keeps a leaked directory from being reused.
    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "unsloth-managed-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The one directory llama_runtime_fingerprint_at walks, kept in one place so
    /// the tests and the walk cannot drift apart on the Windows-only Release level.
    fn runtime_bin_dir(root: &Path) -> PathBuf {
        let bin = root.join("build").join("bin");
        if cfg!(windows) {
            bin.join("Release")
        } else {
            bin
        }
    }

    /// A tree shaped like a prebuilt install: a server binary and one shared
    /// library, the smallest thing quarantine can take a file out of.
    /// The names half of an expected fingerprint, so the counters beside it stay
    /// readable as literals.
    fn names_sum(names: &[&str]) -> u64 {
        names.iter().fold(0u64, |acc, name| {
            acc.wrapping_add(name_hash(std::ffi::OsStr::new(name)))
        })
    }

    /// The entrypoint name `holds_a_llama_server` looks for on this platform.
    fn server_file_name() -> &'static str {
        if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        }
    }

    fn install_fake_runtime(root: &Path) -> PathBuf {
        let bin = runtime_bin_dir(root);
        fs::create_dir_all(&bin).unwrap();
        let server = bin.join("llama-server");
        fs::write(&server, vec![0u8; 4096]).unwrap();
        // Executable, as the installer leaves it, so the counter that watches the
        // bit starts where a real install starts.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut mode = fs::metadata(&server).unwrap().permissions();
            mode.set_mode(0o755);
            fs::set_permissions(&server, mode).unwrap();
        }
        let library = bin.join("libggml-base.so");
        fs::write(&library, vec![0u8; 2048]).unwrap();
        // Explicit, not umask-dependent: the permission bits are part of the
        // fingerprint now, so a runner with a different umask would otherwise
        // read a different string for the same tree.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut mode = fs::metadata(&library).unwrap().permissions();
            mode.set_mode(0o644);
            fs::set_permissions(&library, mode).unwrap();
        }
        bin
    }

    /// A fingerprint whose venv half is fixed and whose runtime half is read off
    /// disk now, which is what each launch does. Built here rather than through
    /// managed_bin_fingerprint to keep the process-wide
    /// UNSLOTH_TEST_LLAMA_RUNTIME_ROOT out of tests running beside these.
    fn fingerprint_for_runtime(runtime_root: &Path) -> ManagedBinFingerprint {
        ManagedBinFingerprint {
            bin_path: "/managed/unsloth".to_string(),
            bin_size: 42,
            bin_mtime_ms: 1_700_000_000_000,
            studio_root_id: None,
            marker_path: None,
            marker_size: None,
            marker_mtime_ms: None,
            llama_runtime: llama_runtime_fingerprint_at(runtime_root),
        }
    }

    /// Points capability_cache_path() at a directory of this test's own for as long
    /// as the guard lives.
    ///
    /// The hook is a process-wide variable, so the guard holds PROCESS_ENV_LOCK for
    /// the whole test: the same lock preflight.rs takes around its own
    /// capability-cache tests and main.rs around XDG_DATA_HOME. The per-test
    /// directory means a file left by a panicking test cannot be read by another.
    struct CapabilityCacheHome {
        home: PathBuf,
        previous: Option<std::ffi::OsString>,
        /// Declared last so it is released after the restore below has run.
        _env: std::sync::MutexGuard<'static, ()>,
    }

    impl CapabilityCacheHome {
        fn new(test_name: &str) -> Self {
            let _env = crate::native_path_policy::PROCESS_ENV_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let home = std::env::temp_dir().join(format!(
                "unsloth-managed-cache-{test_name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = fs::remove_dir_all(&home);
            fs::create_dir_all(&home).unwrap();
            let previous = std::env::var_os("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME");
            std::env::set_var("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME", &home);
            Self {
                home,
                previous,
                _env,
            }
        }

        /// Resolved through the hook rather than rebuilt here, so a layout change
        /// cannot leave these tests asserting against a path nothing writes.
        fn cache_file(&self) -> PathBuf {
            let path = capability_cache_path().expect("the test hook must resolve a cache path");
            assert!(
                path.starts_with(&self.home),
                "the hook must resolve inside this test's home, not the real one: {path:?}"
            );
            path
        }
    }

    impl Drop for CapabilityCacheHome {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => {
                    std::env::set_var("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME", previous)
                }
                None => std::env::remove_var("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME"),
            }
            let _ = fs::remove_dir_all(&self.home);
        }
    }

    #[test]
    fn an_unchanged_runtime_serves_the_cached_capability_back_from_disk() {
        // A healthy install must hit on its second launch, or the desktop pays both
        // probe subprocesses and their 10s ceilings every time the window opens.
        let home = CapabilityCacheHome::new("hit");
        let root = scratch_dir("runtime-hit");
        install_fake_runtime(&root);

        let fingerprint = fingerprint_for_runtime(&root);
        assert!(
            read_cached_capability(&fingerprint).is_none(),
            "nothing is cached yet, so the first launch must miss"
        );
        write_cached_capability(&fingerprint, &healthy_capability());
        assert!(
            home.cache_file().exists(),
            "the cache must reach disk, not just the struct"
        );

        // The next launch recomputes the fingerprint against the same tree.
        let relaunch = fingerprint_for_runtime(&root);
        let cached = read_cached_capability(&relaunch)
            .expect("an install nothing touched must hit its own cache");
        assert_eq!(cached.llama_runtime_ok, Some(true));
        assert!(desktop_capability_ready(&cached));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_file_quarantined_out_of_the_runtime_misses_the_cache() {
        // Quarantine takes one file out of an otherwise present tree, leaving the
        // venv, markers and launcher untouched. If this hit, preflight would answer
        // Ready from a cache written while the tree was intact, and the user would
        // meet the damage as a model load failure.
        let home = CapabilityCacheHome::new("quarantine");
        let root = scratch_dir("runtime-quarantine");
        let bin = install_fake_runtime(&root);

        let healthy = fingerprint_for_runtime(&root);
        write_cached_capability(&healthy, &healthy_capability());
        assert!(read_cached_capability(&healthy).is_some());

        fs::remove_file(bin.join("libggml-base.so")).unwrap();
        let quarantined = fingerprint_for_runtime(&root);
        assert_ne!(healthy.llama_runtime, quarantined.llama_runtime);
        assert!(
            read_cached_capability(&quarantined).is_none(),
            "a quarantined runtime file must not keep serving a cached Ready"
        );
        // The stale entry is still on disk, so the miss came from the fingerprint.
        assert!(home.cache_file().exists());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_file_added_to_the_runtime_misses_the_cache() {
        // What a half-finished repair or partial download leaves behind: a tree that
        // gained a file is not the tree the CLI was asked about.
        let _home = CapabilityCacheHome::new("added");
        let root = scratch_dir("runtime-added");
        let bin = install_fake_runtime(&root);

        let before = fingerprint_for_runtime(&root);
        write_cached_capability(&before, &healthy_capability());

        fs::write(bin.join("llama-quantize"), vec![0u8; 512]).unwrap();
        let after = fingerprint_for_runtime(&root);
        assert!(
            read_cached_capability(&after).is_none(),
            "a runtime that gained a file must be asked about again"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_runtime_directory_that_is_gone_entirely_misses_the_cache() {
        // An uninstall between launches. The fingerprint drops to None, which must
        // not compare equal to the string written while the tree was there.
        let _home = CapabilityCacheHome::new("removed");
        let root = scratch_dir("runtime-removed");
        install_fake_runtime(&root);

        let installed = fingerprint_for_runtime(&root);
        write_cached_capability(&installed, &healthy_capability());

        fs::remove_dir_all(&root).unwrap();
        let gone = fingerprint_for_runtime(&root);
        assert_eq!(gone.llama_runtime, None);
        assert!(
            read_cached_capability(&gone).is_none(),
            "a runtime that was removed must not be served the entry it had while installed"
        );
    }

    #[test]
    fn an_install_with_no_runtime_at_all_still_hits_its_cache() {
        // A user who never installed a runtime has None on both sides, and treating
        // that as a change would make every launch pay both subprocesses forever.
        // The runtime appearing later is a real change and must miss.
        let _home = CapabilityCacheHome::new("no-runtime");
        // A path that does not exist, not merely an empty one: a root present
        // without a build/bin is now its own state, so "no runtime at all" is the
        // absence of the tree itself.
        let parent = scratch_dir("runtime-absent");
        let root = parent.join("never-installed");

        let without = fingerprint_for_runtime(&root);
        assert_eq!(without.llama_runtime, None);
        write_cached_capability(&without, &healthy_capability());
        assert!(
            read_cached_capability(&fingerprint_for_runtime(&root)).is_some(),
            "an install with no runtime must not miss its own cache every launch"
        );

        install_fake_runtime(&root);
        let with = fingerprint_for_runtime(&root);
        assert!(with.llama_runtime.is_some());
        assert!(
            read_cached_capability(&with).is_none(),
            "a runtime installed after the cache was written must be probed"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_cached_capability_that_is_itself_broken_is_never_served() {
        // write_cached_capability runs before the ready check, so a broken runtime's
        // entry does reach disk. Reading it back must refuse it even though the
        // fingerprint matches, or the next launch undoes the repair.
        let home = CapabilityCacheHome::new("broken");
        let root = scratch_dir("runtime-broken");
        install_fake_runtime(&root);

        let fingerprint = fingerprint_for_runtime(&root);
        let mut capability = healthy_capability();
        capability.llama_runtime_ok = Some(false);
        capability.llama_runtime_reason = Some("llama_runtime_payload_incomplete".to_string());
        write_cached_capability(&fingerprint, &capability);

        let raw =
            fs::read(home.cache_file()).expect("the broken verdict is written like any other");
        assert!(String::from_utf8_lossy(&raw).contains("llama_runtime_payload_incomplete"));
        assert!(
            read_cached_capability(&fingerprint).is_none(),
            "a broken runtime must be re-probed even when nothing on disk moved"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_runtime_verdict_the_cli_skipped_is_never_cached() {
        // Codex 3959620616, P2. The CLI declines to grade the managed tree while a
        // custom runtime is selected, and nothing in this fingerprint watches that
        // selection: it lives in the settings database and in LLAMA_SERVER_PATH.
        // Caching the skip meant that clearing the selection made a damaged managed
        // tree active with its fingerprint unchanged, so the cache kept answering
        // Ready for a tree nobody had graded.
        let home = CapabilityCacheHome::new("skipped-verdict");
        let root = scratch_dir("runtime-skipped-verdict");
        install_fake_runtime(&root);
        let fingerprint = fingerprint_for_runtime(&root);

        let mut skipped = healthy_capability();
        skipped.llama_runtime_ok = None;
        skipped.llama_runtime_reason = Some(LLAMA_RUNTIME_NOT_MANAGED.to_string());
        // Still Ready, so this is about the cache and not about the verdict.
        assert!(desktop_capability_ready(&skipped));
        write_cached_capability(&fingerprint, &skipped);
        assert!(
            !home.cache_file().exists(),
            "a verdict that was never reached must not reach the cache"
        );
        assert!(read_cached_capability(&fingerprint).is_none());

        // The other null, "nothing is installed", is a fact about the tree that the
        // fingerprint does watch, so it still caches.
        let mut not_installed = healthy_capability();
        not_installed.llama_runtime_ok = None;
        not_installed.llama_runtime_reason = Some(String::new());
        write_cached_capability(&fingerprint, &not_installed);
        assert!(
            read_cached_capability(&fingerprint).is_some(),
            "only the skipped verdict is uncacheable, or every launch pays a probe"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_verdict_is_only_cached_against_the_tree_the_probe_read() {
        // Codex 3973890115, P2. The fingerprint used to be re-read after the probe, so
        // a file quarantined while desktop-capabilities was running was stored as
        // healthy under its own damaged fingerprint. That entry then matched on every
        // later launch, the CLI was never asked again, and the repair was never
        // offered, which is worse than the race it came from.
        let parent = scratch_dir("verdict-snapshot");
        let root = parent.join("llama.cpp");
        install_fake_runtime(&root);
        let before = fingerprint_for_runtime(&root);

        assert_eq!(
            fingerprint_to_cache_under(Some(&before), Some(fingerprint_for_runtime(&root))),
            Some(&before),
            "an unchanged tree is what the answer describes, so it is cacheable"
        );

        // Quarantine lands mid-probe.
        fs::remove_file(runtime_bin_dir(&root).join("libggml-base.so")).unwrap();
        assert_eq!(
            fingerprint_to_cache_under(Some(&before), Some(fingerprint_for_runtime(&root))),
            None,
            "a verdict about a tree that no longer exists must not be kept"
        );
        assert_eq!(fingerprint_to_cache_under(Some(&before), None), None);
        assert_eq!(fingerprint_to_cache_under(None, Some(before)), None);

        let _ = fs::remove_dir_all(&parent);
    }

    #[test]
    fn a_runtime_probe_that_raised_is_never_cached_either() {
        // Codex 3973660789, P2. A probe that throws leaves a null verdict, which is
        // Ready, and it carries the damaged tree's own fingerprint, so caching it
        // froze a Ready that was never reached and every later launch skipped the
        // probe. Its reason tells it apart from the two nulls that are facts.
        let home = CapabilityCacheHome::new("probe-failed-verdict");
        let root = scratch_dir("runtime-probe-failed");
        install_fake_runtime(&root);
        let fingerprint = fingerprint_for_runtime(&root);

        let mut failed = healthy_capability();
        failed.llama_runtime_ok = None;
        failed.llama_runtime_reason = Some(LLAMA_RUNTIME_PROBE_FAILED.to_string());
        assert!(desktop_capability_ready(&failed), "still Ready: this is about the cache");
        write_cached_capability(&fingerprint, &failed);
        assert!(
            !home.cache_file().exists(),
            "an attempt that raised must not freeze a Ready over a damaged tree"
        );
        assert!(read_cached_capability(&fingerprint).is_none());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_named_windows_profile_resolves_where_the_child_will_look() {
        // Codex 3959620595, P2. process.rs pins UNSLOTH_LLAMA_CPP_PATH for the child
        // through expand_windows_user, and the CLI reads it with ntpath.expanduser,
        // so both land on the sibling profile while this fingerprinted
        // <cwd>\~other\llama.cpp. Quarantine under the real profile then never
        // invalidated a cached healthy result. Same cases as process.rs's own test,
        // so the two readers cannot drift apart.
        let home = Path::new("C:\\Users\\me");
        assert_eq!(
            named_windows_user_home("~other\\llama.cpp", home, Some("me")),
            Some(PathBuf::from("C:\\Users\\other\\llama.cpp")),
        );
        assert_eq!(
            named_windows_user_home("~other/llama.cpp", home, Some("me")),
            Some(PathBuf::from("C:\\Users\\other/llama.cpp")),
        );
        assert_eq!(
            named_windows_user_home("~me\\llama.cpp", home, Some("me")),
            Some(PathBuf::from("C:\\Users\\me\\llama.cpp")),
        );
        assert_eq!(
            named_windows_user_home("~other", home, Some("me")),
            Some(PathBuf::from("C:\\Users\\other")),
        );
        // ntpath declines to guess when this profile is not named after the current
        // user, since C:\Users\me.DOMAIN is not other's sibling. None here leaves the
        // value anchored as a relative path, which is what happened before.
        assert_eq!(
            named_windows_user_home("~other\\llama.cpp", Path::new("C:\\Users\\me.DOMAIN"), Some("me")),
            None,
        );
        assert_eq!(named_windows_user_home("~other\\llama.cpp", home, None), None);
        // A bare tilde is the current profile and is handled before this is reached.
        assert_eq!(named_windows_user_home("~", home, Some("me")), None);
        assert_eq!(named_windows_user_home("~\\llama.cpp", home, Some("me")), None);
    }

    #[test]
    fn a_schema_three_cache_file_on_disk_misses_and_is_rewritten_as_schema_four() {
        // Byte for byte what the previous release wrote. Everything but the schema
        // and the new key matches, so the miss can only come from the version bump.
        // It must not panic, must not be served, and must be replaced.
        let home = CapabilityCacheHome::new("schema-three");
        let root = scratch_dir("runtime-schema-three");
        install_fake_runtime(&root);
        let fingerprint = fingerprint_for_runtime(&root);

        let path = home.cache_file();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let previous_release = format!(
            r#"{{
  "schema": 3,
  "bin_path": "{bin_path}",
  "bin_size": {bin_size},
  "bin_mtime_ms": {bin_mtime_ms},
  "studio_root_id": null,
  "marker_path": null,
  "marker_size": null,
  "marker_mtime_ms": null,
  "desktop_protocol_version": {protocol},
  "desktop_manageability_version": {manageability},
  "capability": {capability}
}}"#,
            bin_path = fingerprint.bin_path,
            bin_size = fingerprint.bin_size,
            bin_mtime_ms = fingerprint.bin_mtime_ms,
            protocol = DESKTOP_PROTOCOL_VERSION,
            manageability = DESKTOP_MANAGEABILITY_VERSION,
            capability = pre_pr_capability_json(),
        );
        fs::write(&path, previous_release).unwrap();
        assert!(
            read_cached_capability(&fingerprint).is_none(),
            "a schema 3 entry reached Ready without ever looking at the runtime"
        );

        write_cached_capability(&fingerprint, &healthy_capability());
        let rewritten: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).expect("the rewrite must be JSON");
        assert_eq!(
            rewritten["schema"].as_u64(),
            Some(u64::from(MANAGED_CAPABILITY_CACHE_SCHEMA))
        );
        assert_eq!(
            rewritten["llama_runtime"].as_str(),
            fingerprint.llama_runtime.as_deref(),
            "the rewritten entry must carry the runtime the old one had no room for"
        );
        assert!(
            read_cached_capability(&fingerprint).is_some(),
            "the replacement entry must then hit, or the upgrade never stops re-probing"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn files_below_the_binary_directory_are_not_walked() {
        // The walk is one level deep because it runs on the launch path. A nested
        // directory must contribute nothing, or a model cache parked under build/bin
        // would make the fingerprint cost grow without bound.
        let root = scratch_dir("runtime-nested");
        let bin = install_fake_runtime(&root);
        let flat = llama_runtime_fingerprint_at(&root).unwrap();

        let nested = bin.join("vendor").join("deep");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("payload.bin"), vec![0u8; 8192]).unwrap();
        assert_eq!(
            flat,
            llama_runtime_fingerprint_at(&root).unwrap(),
            "a subdirectory and its contents are not part of the runtime fingerprint"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_replacement_of_the_same_size_is_a_collision_this_fingerprint_does_not_catch() {
        // The fingerprint is a count and a byte total, no mtime and no hash, so a
        // file rewritten in place at its old length is invisible. Accepted because
        // quarantine and partial extraction always change one of the two.
        let root = scratch_dir("runtime-collision");
        let bin = install_fake_runtime(&root);
        let original = llama_runtime_fingerprint_at(&root).unwrap();

        fs::write(bin.join("llama-server"), vec![0xABu8; 4096]).unwrap();
        assert_eq!(
            original,
            llama_runtime_fingerprint_at(&root).unwrap(),
            "same count and same bytes is a collision, and this asserts it honestly"
        );

        fs::write(bin.join("llama-server"), vec![0u8; 4097]).unwrap();
        assert_ne!(
            original,
            llama_runtime_fingerprint_at(&root).unwrap(),
            "one byte of size difference must still invalidate"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_root_entrypoint_is_part_of_the_fingerprint() {
        // Codex 3971674487, P2. _find_llama_server_binary reaches <root>/llama-server
        // before build/bin, and create_exec_entrypoint writes a real wrapper there when
        // it cannot make a symlink, so it rots on its own. installed_runtime_health
        // grades it; the walk above only reads build/bin, so without this the cached
        // Ready outlived the damage and the CLI was never asked.
        let root = scratch_dir("runtime-root-entrypoint");
        install_fake_runtime(&root);
        let without = llama_runtime_fingerprint_at(&root).unwrap();

        let wrapper = root.join(if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        });
        fs::write(&wrapper, b"#!/bin/sh\n").unwrap();
        let with_wrapper = llama_runtime_fingerprint_at(&root).unwrap();
        assert_ne!(without, with_wrapper, "a root entrypoint appearing must invalidate");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut mode = fs::metadata(&wrapper).unwrap().permissions();
            mode.set_mode(0o755);
            fs::set_permissions(&wrapper, mode).unwrap();
            let executable = llama_runtime_fingerprint_at(&root).unwrap();
            assert_ne!(with_wrapper, executable, "the wrapper's mode is graded, so watch it");

            mode = fs::metadata(&wrapper).unwrap().permissions();
            mode.set_mode(0o644);
            fs::set_permissions(&wrapper, mode).unwrap();
            assert_eq!(
                with_wrapper,
                llama_runtime_fingerprint_at(&root).unwrap(),
                "and the same tree must fingerprint the same both times"
            );
        }

        fs::remove_file(&wrapper).unwrap();
        assert_eq!(
            without,
            llama_runtime_fingerprint_at(&root).unwrap(),
            "an absent root entrypoint is the state a plain build/bin install is in"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn renaming_a_runtime_file_in_place_moves_the_fingerprint() {
        // Codex 3962938547, P2. Quarantine does not always delete: some products
        // rename the file beside itself. The count, the byte total, the link count
        // and the mode sum are all aggregates and none of them moves for that, so
        // the cached Ready kept matching a tree installed_runtime_health rejects
        // and preflight answered Ready without running the capability probe. The
        // launch then failed with no repair offered.
        let root = scratch_dir("runtime-renamed-in-place");
        let bin = install_fake_runtime(&root);

        let intact = llama_runtime_fingerprint_at(&root).unwrap();
        fs::rename(
            bin.join("libggml-base.so"),
            bin.join("libggml-base.so.quarantine"),
        )
        .unwrap();
        let renamed = llama_runtime_fingerprint_at(&root).unwrap();
        assert_ne!(
            intact, renamed,
            "a required library renamed in place must not fingerprint as the healthy tree"
        );
        // And the counters really are blind to it, which is why the names are here.
        // The root segment is dropped first: it carries its own colons.
        let counters = |value: &str| {
            value.split_once('|').unwrap().0.rsplit_once(':').unwrap().0.to_string()
        };
        assert_eq!(counters(&intact), counters(&renamed));

        // Renaming it back restores the original exactly, so the sum is a property
        // of the tree rather than of the order the names arrived in.
        fs::rename(
            bin.join("libggml-base.so.quarantine"),
            bin.join("libggml-base.so"),
        )
        .unwrap();
        assert_eq!(intact, llama_runtime_fingerprint_at(&root).unwrap());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_runtime_root_that_is_a_file_fingerprints_as_no_runtime() {
        // UNSLOTH_LLAMA_CPP_PATH can point at anything a user typed. Joining
        // build/bin onto a regular file must degrade rather than panic.
        let parent = scratch_dir("runtime-is-a-file");
        let root = parent.join("llama.cpp");
        fs::write(&root, "not a directory").unwrap();
        assert_eq!(llama_runtime_fingerprint_at(&root), None);
        let _ = fs::remove_dir_all(&parent);
    }

    #[test]
    fn a_runtime_path_with_spaces_and_non_ascii_fingerprints_normally() {
        // Home directories are named after people, and a path this walk could not
        // handle would silently drop the runtime out for those users only.
        let parent = scratch_dir("runtime-unicode");
        let root = parent.join("Мой каталог ünïcode llama.cpp");
        install_fake_runtime(&root);
        assert_eq!(
            llama_runtime_fingerprint_at(&root).as_deref(),
            Some(
                format!(
                    "bin:2:6144:0:{}:{}|root:-,-",
                    if cfg!(unix) { 913 } else { 0 },
                    names_sum(&["llama-server", "libggml-base.so"])
                )
                .as_str()
            )
        );
        let _ = fs::remove_dir_all(&parent);
    }

    #[test]
    fn the_runtime_fingerprint_does_not_depend_on_directory_order() {
        // read_dir order is unspecified, and the venv half sorts for that reason.
        // Two trees with the same files in opposite orders must agree, or an install
        // misses its own cache on some launches and not others.
        let parent = scratch_dir("runtime-order");
        let names = ["llama-server", "libggml.so", "llama-cli", "libmtmd.so"];

        let forwards = parent.join("forwards");
        let forwards_bin = runtime_bin_dir(&forwards);
        fs::create_dir_all(&forwards_bin).unwrap();
        for (index, name) in names.iter().enumerate() {
            fs::write(forwards_bin.join(name), vec![0u8; 100 + index]).unwrap();
        }

        let backwards = parent.join("backwards");
        let backwards_bin = runtime_bin_dir(&backwards);
        fs::create_dir_all(&backwards_bin).unwrap();
        for (index, name) in names.iter().enumerate().rev() {
            fs::write(backwards_bin.join(name), vec![0u8; 100 + index]).unwrap();
        }

        assert_eq!(
            llama_runtime_fingerprint_at(&forwards),
            llama_runtime_fingerprint_at(&backwards)
        );
        let _ = fs::remove_dir_all(&parent);
    }

    #[test]
    fn a_large_runtime_directory_fingerprints_fast_enough_for_the_launch_path() {
        // This runs on every launch ahead of the probes it saves, so a build
        // directory's worth of files must cost milliseconds. The bound is loose
        // enough not to flake and tight enough to catch a walk that reads contents
        // or recurses.
        let root = scratch_dir("runtime-large");
        let bin = runtime_bin_dir(&root);
        fs::create_dir_all(&bin).unwrap();
        for index in 0..5000 {
            let artifact = bin.join(format!("artifact-{index:05}.o"));
            fs::write(&artifact, [0u8; 1]).unwrap();
            // Same reason as install_fake_runtime: the mode is fingerprinted, so
            // it is set here rather than left to the runner's umask.
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut mode = fs::metadata(&artifact).unwrap().permissions();
                mode.set_mode(0o644);
                fs::set_permissions(&artifact, mode).unwrap();
            }
        }

        let started = Instant::now();
        let fingerprint = llama_runtime_fingerprint_at(&root);
        let elapsed = started.elapsed();
        assert_eq!(
            fingerprint.as_deref(),
            Some(
                format!(
                    "bin:5000:5000:0:{}:{}|root:-,-",
                    if cfg!(unix) { 2100000 } else { 0 },
                    names_sum(
                        &(0..5000)
                            .map(|index| format!("artifact-{index:05}.o"))
                            .collect::<Vec<_>>()
                            .iter()
                            .map(String::as_str)
                            .collect::<Vec<_>>()
                    )
                )
                .as_str()
            )
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "5000 files took {elapsed:?}, which is too much to spend before the window opens"
        );

        let _ = fs::remove_dir_all(&root);
    }

    // Symlinks are what a developer checkout and a hand-placed
    // UNSLOTH_LLAMA_CPP_PATH look like, and the one input that can make this walk
    // answer about a tree other than the one it was given.
    #[cfg(unix)]
    #[test]
    fn a_symlinked_runtime_root_fingerprints_the_tree_it_points_at() {
        let parent = scratch_dir("runtime-symlinked-root");
        let real = parent.join("real");
        let bin = install_fake_runtime(&real);
        let linked = parent.join("linked");
        std::os::unix::fs::symlink(&real, &linked).unwrap();

        assert_eq!(
            llama_runtime_fingerprint_at(&real),
            llama_runtime_fingerprint_at(&linked),
            "a symlinked root must see the same tree, or an override written that way is not covered"
        );
        // And it must keep tracking it: a quarantine through the link counts.
        fs::remove_file(bin.join("libggml-base.so")).unwrap();
        assert_eq!(
            llama_runtime_fingerprint_at(&linked).as_deref(),
            Some(
                format!(
                    "bin:1:4096:0:{}:{}|root:-,-",
                    if cfg!(unix) { 493 } else { 0 },
                    names_sum(&["llama-server"])
                )
                .as_str()
            )
        );

        let _ = fs::remove_dir_all(&parent);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_file_inside_the_runtime_is_counted_and_its_loss_is_seen() {
        // Codex 3960069958, P1. DirEntry::metadata does not follow a link, so a link
        // was neither a file nor a byte and no count could move when one appeared or
        // went. That is not a bound worth keeping: on macOS the install name
        // llama-server loads is the middle link of libggml.dylib -> libggml.0.dylib
        // -> libggml.0.23.0.dylib, and on Linux the SONAME can be a link too, so
        // losing exactly the entry installed_runtime_health calls fatal left a
        // cached Ready matching.
        let parent = scratch_dir("runtime-symlinked-file");
        let root = parent.join("llama.cpp");
        let bin = install_fake_runtime(&root);
        let before = llama_runtime_fingerprint_at(&root).unwrap();

        std::os::unix::fs::symlink(bin.join("llama-server"), bin.join("llama-cli")).unwrap();
        let with_link = llama_runtime_fingerprint_at(&root).unwrap();
        assert_ne!(
            before, with_link,
            "a link beside the binaries has to move the fingerprint"
        );
        // And losing it again is seen, which is the case that matters.
        fs::remove_file(bin.join("llama-cli")).unwrap();
        assert_eq!(before, llama_runtime_fingerprint_at(&root).unwrap());

        let _ = fs::remove_dir_all(&parent);
    }

    #[test]
    fn an_installer_managed_venv_names_its_own_runtime() {
        // Codex 3960069972, P1. The desktop strips UNSLOTH_STUDIO_HOME before
        // spawning the CLI, but the CLI puts it back: _resolve_studio_home re-infers
        // the root from sys.prefix and _ensure_studio_env_exported points
        // UNSLOTH_LLAMA_CPP_PATH at that root's llama.cpp. Reading only the ambient
        // environment fingerprinted the legacy tree, so quarantine inside the custom
        // runtime never moved it and a cached Ready outlived the damage.
        let root = scratch_dir("inferred-studio-home");
        let bin = root.join("unsloth_studio").join("bin").join("unsloth");
        fs::create_dir_all(bin.parent().unwrap()).unwrap();
        fs::write(&bin, b"launcher").unwrap();

        // No sentinel yet, so this is a developer venv that happens to be named
        // unsloth_studio and the legacy tree is still the answer.
        assert_eq!(inferred_studio_llama_root(&bin), None);

        fs::create_dir_all(root.join("share")).unwrap();
        fs::write(root.join("share").join("studio.conf"), b"managed").unwrap();
        assert_eq!(
            inferred_studio_llama_root(&bin),
            Some(root.join("llama.cpp")),
            "an installer-managed root names the runtime beside it"
        );

        // A venv that is not the one the installer builds says nothing either.
        let loose = root.join("some_venv").join("bin").join("unsloth");
        fs::create_dir_all(loose.parent().unwrap()).unwrap();
        fs::write(&loose, b"launcher").unwrap();
        assert_eq!(inferred_studio_llama_root(&loose), None);

        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(not(windows))]
    #[test]
    fn the_unix_sentinel_is_the_launcher_the_installer_writes() {
        let root = scratch_dir("studio-sentinel");
        fs::create_dir_all(root.join("bin")).unwrap();
        assert!(!looks_like_installer_managed_studio_home(&root));
        fs::write(root.join("bin").join("unsloth"), b"launcher").unwrap();
        assert!(looks_like_installer_managed_studio_home(&root));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_cmd_shim_has_to_be_the_one_the_installer_generated() {
        // The directory is on PATH, so any file of that name would otherwise be
        // enough to point a custom root at itself. Same markers Python reads.
        let root = scratch_dir("cmd-shim");
        let shim = root.join("unsloth.cmd");
        fs::create_dir_all(&root).unwrap();
        assert!(!is_managed_cmd_shim(&shim));

        fs::write(&shim, b"@echo off\r\npython -m unsloth_cli %*\r\n").unwrap();
        assert!(
            !is_managed_cmd_shim(&shim),
            "a hand rolled wrapper that calls the CLI is not this shim"
        );

        fs::write(
            &shim,
            b"@rem unsloth-studio-managed-launcher\r\n@python -c \"from unsloth_cli import app; app()\" %*\r\n",
        )
        .unwrap();
        assert!(is_managed_cmd_shim(&shim));

        // Too big to be the generated shim, whatever it contains.
        let mut oversized = b"unsloth-studio-managed-launcher from unsloth_cli import app".to_vec();
        oversized.resize((CMD_SHIM_MAX_BYTES + 1) as usize, b' ');
        fs::write(&shim, &oversized).unwrap();
        assert!(!is_managed_cmd_shim(&shim));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_override_with_no_server_in_it_does_not_stop_the_search() {
        // Codex 3960069962, P2. _scan_pinned finds no candidate under an empty or
        // missing UNSLOTH_LLAMA_CPP_PATH and walks on to the tree behind it, so
        // treating the override as final fingerprinted a folder nobody loads while
        // the runtime the backend really opens went unwatched.
        let root = scratch_dir("override-empty");
        fs::create_dir_all(&root).unwrap();
        assert!(!holds_a_llama_server(&root));
        assert!(!holds_a_llama_server(&root.join("not-there")));

        let bin = install_fake_runtime(&root);
        // The name the finder looks for, which is not the name that fixture writes:
        // it exists to be counted, and the counters do not care what a file is
        // called. On Windows the entrypoint carries .exe, so writing it here is what
        // makes this a tree the backend would actually stop at.
        fs::write(bin.join(server_file_name()), b"binary").unwrap();
        assert!(
            holds_a_llama_server(&root),
            "build/bin/llama-server is one of the layouts the backend accepts"
        );

        // The flat layout counts too, and so does a server without the execute
        // bit: _scan_pinned calls that one non_executable and the caller turns it
        // into _unavailable, which ends the search here rather than walking on.
        let flat = scratch_dir("override-flat");
        fs::create_dir_all(&flat).unwrap();
        let server = flat.join(server_file_name());
        fs::write(&server, b"binary").unwrap();
        assert!(holds_a_llama_server(&flat));

        // A directory of that name is not a server, and neither is an empty tree.
        let decoy = scratch_dir("override-decoy");
        fs::create_dir_all(decoy.join(server_file_name())).unwrap();
        assert!(!holds_a_llama_server(&decoy));
        let _ = fs::remove_dir_all(&decoy);

        let _ = bin;
        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(&flat);
    }

    #[cfg(unix)]
    #[test]
    fn a_named_user_resolves_through_the_reentrant_lookup() {
        // Codex 3960069965, P2. getpwnam hands back process-global storage and this
        // runs on a Tauri worker thread, so the lookup is the _r form now. root is
        // the one account every unix box has, which makes this checkable anywhere.
        let home = Path::new("/home/whoever");
        let resolved = named_user_home("~root/llama.cpp", Some(home));
        assert!(
            resolved.is_some(),
            "root must resolve, or the reentrant lookup is not answering at all"
        );
        assert!(resolved.unwrap().ends_with("llama.cpp"));
        assert_eq!(
            named_user_home("~no-such-account-anywhere/x", Some(home)),
            None,
            "an unknown name leaves the value to be anchored like any relative path"
        );
    }

    #[cfg(unix)]
    #[test]
    fn clearing_the_execute_bit_moves_the_fingerprint() {
        // The other half of the same catch: taking the execute bit off llama-server
        // changes neither the count nor the byte total, and it is the state
        // installed_runtime_health answers llama_runtime_binaries_missing for, so a
        // cached Ready went on matching a tree that could no longer launch.
        use std::os::unix::fs::PermissionsExt;

        let parent = scratch_dir("runtime-exec-bit");
        let root = parent.join("llama.cpp");
        let bin = install_fake_runtime(&root);
        let before = llama_runtime_fingerprint_at(&root).unwrap();

        let server = bin.join("llama-server");
        let mut mode = fs::metadata(&server).unwrap().permissions();
        mode.set_mode(0o644);
        fs::set_permissions(&server, mode).unwrap();
        assert_ne!(
            before,
            llama_runtime_fingerprint_at(&root).unwrap(),
            "a binary that can no longer be executed is a different tree"
        );

        let mut restored = fs::metadata(&server).unwrap().permissions();
        restored.set_mode(0o755);
        fs::set_permissions(&server, restored).unwrap();
        assert_eq!(before, llama_runtime_fingerprint_at(&root).unwrap());

        let _ = fs::remove_dir_all(&parent);
    }

    #[cfg(unix)]
    #[test]
    fn losing_only_the_owners_execute_bit_moves_the_fingerprint() {
        // Codex 3960401504, P2. os.access(X_OK) asks whether THIS user may run the
        // file, and the owner's bits answer that on their own: 0755 -> 0655 leaves
        // group and other executable, so a counter of "has some execute bit" never
        // moved while installed_runtime_health flipped to
        // llama_runtime_binaries_missing. The permission bits themselves are in the
        // fingerprint now, so the cached Ready cannot outlive the change.
        use std::os::unix::fs::PermissionsExt;

        let parent = scratch_dir("runtime-owner-bit");
        let root = parent.join("llama.cpp");
        let bin = install_fake_runtime(&root);
        let server = bin.join("llama-server");
        let mut executable = fs::metadata(&server).unwrap().permissions();
        executable.set_mode(0o755);
        fs::set_permissions(&server, executable).unwrap();
        let before = llama_runtime_fingerprint_at(&root).unwrap();

        let mut owner_only = fs::metadata(&server).unwrap().permissions();
        owner_only.set_mode(0o655);
        fs::set_permissions(&server, owner_only).unwrap();
        // Still executable to somebody, which is what the old counter measured.
        assert_eq!(fs::metadata(&server).unwrap().permissions().mode() & 0o111, 0o011);
        assert_ne!(
            before,
            llama_runtime_fingerprint_at(&root).unwrap(),
            "a binary this user can no longer run is a different tree"
        );

        let _ = fs::remove_dir_all(&parent);
    }

    #[cfg(unix)]
    #[test]
    fn a_dangling_symlink_does_not_break_the_walk() {
        // What a quarantine that removed a link target leaves behind. The walk must
        // finish and answer about the real files, not report the runtime missing.
        let parent = scratch_dir("runtime-dangling");
        let root = parent.join("llama.cpp");
        let bin = install_fake_runtime(&root);
        let before = llama_runtime_fingerprint_at(&root).unwrap();

        std::os::unix::fs::symlink(bin.join("gone.so"), bin.join("libggml.so")).unwrap();
        let dangling = llama_runtime_fingerprint_at(&root);
        assert!(
            dangling.is_some(),
            "a broken link must not make the walk fail or report the runtime missing"
        );
        // It counts as a link rather than as a file, so the real files behind it are
        // still reported unchanged and only the link counter moves.
        assert_ne!(before, dangling.clone().unwrap());
        fs::remove_file(bin.join("libggml.so")).unwrap();
        assert_eq!(before, llama_runtime_fingerprint_at(&root).unwrap());

        let _ = fs::remove_dir_all(&parent);
    }

    #[cfg(unix)]
    #[test]
    fn a_runtime_directory_that_cannot_be_read_is_a_broken_install_not_an_absent_one() {
        // A tightened-down or half-owned install directory. read_dir fails, and the
        // answer must be neither a panic nor None: the tree is there, so this is the
        // broken-install state, and sharing None with a machine that never installed
        // a runtime let a cache written before the install keep matching after it
        // broke.
        use std::os::unix::fs::PermissionsExt;

        let root = scratch_dir("runtime-denied");
        let bin = install_fake_runtime(&root);
        let mut denied = fs::metadata(&bin).unwrap().permissions();
        denied.set_mode(0o000);
        fs::set_permissions(&bin, denied).unwrap();

        // Mode bits do not apply to root, and some CI images run as root, so assert
        // only where the denial is real.
        if fs::read_dir(&bin).is_err() {
            let denied_fingerprint = llama_runtime_fingerprint_at(&root);
            assert!(
                denied_fingerprint.is_some(),
                "a root that is present but unreadable is a broken install, not an absent one"
            );
            assert_ne!(denied_fingerprint, None);
        }

        let mut restored = fs::metadata(&bin).unwrap().permissions();
        restored.set_mode(0o755);
        fs::set_permissions(&bin, restored).unwrap();
        let _ = fs::remove_dir_all(&root);
    }
}
