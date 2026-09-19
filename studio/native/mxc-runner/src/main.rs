// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
use std::os::windows::io::AsRawHandle;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use mxc_sdk::policy::{FilesystemSection, NetworkSection};
use mxc_sdk::{build_request, spawn_sandbox, SandboxPolicy, SandboxRequest, WaitOutcome};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::Storage::FileSystem::{
    GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION, FILE_FLAG_BACKUP_SEMANTICS,
    FILE_FLAG_OPEN_REPARSE_POINT, FILE_READ_ATTRIBUTES, FILE_SHARE_READ, FILE_SHARE_WRITE,
};
use windows_sys::Win32::System::Pipes::PeekNamedPipe;

const PROTOCOL: u32 = 1;
const PROFILE_VERSION: u32 = 1;
const POLICY_IDENTITY_VERSION: u32 = 1;
const MAX_REQUEST: usize = 262_144;
const MAX_ITEMS: usize = 512;
const MXC_REVISION: &str = "ca7ea12ac6bd9f5420d6adecb37e32a8158da476";
const MXC_SCHEMA_VERSION: &str = "0.8.0-alpha";
const PROFILE_ID: &str = "unsloth-mxc-windows-basecontainer-v1";

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunnerIdentity<'a> {
    protocol_version: u32,
    profile_id: &'a str,
    profile_version: u32,
    schema_version: &'a str,
    mxc_revision: &'a str,
    mxc_patch_sha256: &'a str,
    mxc_patched_tree: &'a str,
    runner_source_identity: &'a str,
    admission_api: bool,
    architecture: &'a str,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ObjectIdentity {
    volume_serial_number: u32,
    file_id: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct LaunchRequest {
    protocol: u32,
    run_id: String,
    token: String,
    control_pipe: String,
    profile_id: String,
    profile_version: u32,
    policy_hash: String,
    schema_version: String,
    runtime_revision: String,
    container_id: String,
    argv: Vec<String>,
    execution_kind: String,
    runtime_path: String,
    runtime_identity: ObjectIdentity,
    cwd: String,
    workdir_identity: ObjectIdentity,
    environment: BTreeMap<String, String>,
    environment_policy: String,
    command_line_policy: String,
    readwrite_paths: Vec<String>,
    readonly_paths: Vec<String>,
    denied_paths: Vec<String>,
    clear_policy_on_exit: bool,
    network_profile: String,
    allow_outbound: bool,
    allow_local_network: bool,
    ui_policy: Option<Value>,
    timeout_ms: Option<u32>,
    allow_dacl_mutation: bool,
    admission: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Event<'a> {
    v: u32,
    event: &'a str,
    run_id: &'a str,
    token: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    stage: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exit_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timed_out: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cleanup: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    backend_tier: Option<&'a str>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ControlRequest {
    v: u32,
    event: String,
    run_id: String,
    token: String,
}

fn runner_identity() -> RunnerIdentity<'static> {
    RunnerIdentity {
        protocol_version: PROTOCOL,
        profile_id: PROFILE_ID,
        profile_version: PROFILE_VERSION,
        schema_version: MXC_SCHEMA_VERSION,
        mxc_revision: MXC_REVISION,
        mxc_patch_sha256: option_env!("UNSLOTH_MXC_PATCH_SHA256").unwrap_or("unverified"),
        mxc_patched_tree: option_env!("UNSLOTH_MXC_PATCHED_TREE").unwrap_or("unverified"),
        runner_source_identity: option_env!("UNSLOTH_MXC_RUNNER_SOURCE").unwrap_or("unverified"),
        admission_api: cfg!(feature = "mxc-no-dacl-api"),
        architecture: std::env::consts::ARCH,
    }
}

fn emit_identity() -> Result<(), String> {
    serde_json::to_writer(io::stdout().lock(), &runner_identity()).map_err(|e| e.to_string())?;
    println!();
    Ok(())
}

#[cfg(feature = "mxc-test-failure-injection")]
fn inject_failure(stage: &str) {
    if std::env::var("UNSLOTH_MXC_TEST_FAILURE").as_deref() == Ok(stage) {
        if matches!(stage, "after_spawn_before_started" | "after_started") {
            thread::sleep(Duration::from_millis(500));
        }
        std::process::exit(86);
    }
}

#[cfg(not(feature = "mxc-test-failure-injection"))]
fn inject_failure(_stage: &str) {}

fn policy_material(request: &LaunchRequest) -> Value {
    json!({
        "identityVersion": POLICY_IDENTITY_VERSION,
        "protocolVersion": request.protocol,
        "profileId": request.profile_id,
        "profileVersion": request.profile_version,
        "schemaVersion": request.schema_version,
        "runtimeRevision": request.runtime_revision,
        "runId": request.run_id,
        "containerId": request.container_id,
        "argv": request.argv,
        "executionKind": request.execution_kind,
        "runtimePath": request.runtime_path,
        "runtimeIdentity": {
            "volumeSerialNumber": request.runtime_identity.volume_serial_number,
            "fileId": request.runtime_identity.file_id,
        },
        "cwd": request.cwd,
        "workdirIdentity": {
            "volumeSerialNumber": request.workdir_identity.volume_serial_number,
            "fileId": request.workdir_identity.file_id,
        },
        "environment": request.environment,
        "environmentPolicy": request.environment_policy,
        "commandLinePolicy": request.command_line_policy,
        "filesystem": {
            "readwritePaths": request.readwrite_paths,
            "readonlyPaths": request.readonly_paths,
            "deniedPaths": request.denied_paths,
            "clearPolicyOnExit": request.clear_policy_on_exit,
        },
        "network": {
            "profile": request.network_profile,
            "allowOutbound": request.allow_outbound,
            "allowLocalNetwork": request.allow_local_network,
        },
        "uiPolicy": request.ui_policy,
        "timeoutMs": request.timeout_ms,
        "fallback": {"allowDaclMutation": request.allow_dacl_mutation},
        "admission": request.admission,
    })
}

fn recompute_policy_hash(request: &LaunchRequest) -> Result<String, String> {
    let encoded = serde_json::to_vec(&policy_material(request)).map_err(|e| e.to_string())?;
    Ok(format!("sha256:{:x}", Sha256::digest(encoded)))
}

fn verify_policy_hash(request: &LaunchRequest) -> Result<(), String> {
    let recomputed = recompute_policy_hash(request)?;
    if !constant_time_eq(&request.policy_hash, &recomputed) {
        return Err(format!(
            "policy hash mismatch: received {}, recomputed {}",
            request.policy_hash, recomputed
        ));
    }
    Ok(())
}

fn send(stream: &mut File, request: &LaunchRequest, event: &str) -> io::Result<()> {
    send_full(
        stream, request, event, None, None, None, None, None, None, None,
    )
}

#[allow(clippy::too_many_arguments)]
fn send_full(
    stream: &mut File,
    request: &LaunchRequest,
    event: &str,
    stage: Option<&str>,
    code: Option<&str>,
    message: Option<String>,
    exit_code: Option<i32>,
    timed_out: Option<bool>,
    cleanup: Option<&str>,
    backend_tier: Option<&str>,
) -> io::Result<()> {
    let frame = Event {
        v: PROTOCOL,
        event,
        run_id: &request.run_id,
        token: &request.token,
        stage,
        code,
        message,
        exit_code,
        timed_out,
        cleanup,
        backend_tier,
    };
    serde_json::to_writer(&mut *stream, &frame)?;
    stream.write_all(b"\n")?;
    stream.flush()
}

fn fail(stream: &mut File, request: &LaunchRequest, stage: &str, code: &str, message: String) -> ! {
    let _ = send_full(
        stream,
        request,
        "ERROR",
        Some(stage),
        Some(code),
        Some(message),
        None,
        None,
        None,
        None,
    );
    std::process::exit(70)
}

fn read_request() -> Result<LaunchRequest, String> {
    let stdin = io::stdin();
    let mut input = Vec::new();
    stdin
        .lock()
        .take((MAX_REQUEST + 1) as u64)
        .read_to_end(&mut input)
        .map_err(|e| e.to_string())?;
    if input.len() > MAX_REQUEST {
        return Err("request exceeds the protocol bound".into());
    }
    let request: LaunchRequest =
        serde_json::from_slice(&input).map_err(|e| format!("invalid request: {e}"))?;
    if request.protocol != PROTOCOL || request.run_id.len() < 16 || request.token.len() != 64 {
        return Err("invalid protocol identity".into());
    }
    if request.schema_version != MXC_SCHEMA_VERSION
        || request.runtime_revision != MXC_REVISION
        || request.profile_id != PROFILE_ID
        || request.profile_version != PROFILE_VERSION
        || !request.control_pipe.starts_with(r"\\.\pipe\unsloth-mxc-")
        || request.control_pipe.len() > 128
    {
        return Err("unsupported schema or profile".into());
    }
    if !matches!(request.execution_kind.as_str(), "python" | "terminal")
        || request.allow_dacl_mutation
        || request.argv.is_empty()
        || request.argv.len() > MAX_ITEMS
        || request.environment.len() > MAX_ITEMS
        || request.readonly_paths.len() > MAX_ITEMS
        || request.readwrite_paths.is_empty()
        || request.readwrite_paths.len() > MAX_ITEMS
        || request.denied_paths.len() > MAX_ITEMS
        || request.container_id != format!("unsloth-{}", request.run_id)
        || !request.clear_policy_on_exit
        || request.network_profile != "compatibility"
        || !request.allow_outbound
        || !request.allow_local_network
        || request.ui_policy.is_some()
        || request.environment_policy != "sanitized-explicit-v1"
        || request.command_line_policy != "windows-createprocess-argv-v1"
        || request.admission != "atomic-no-dacl-fallback"
        || request.policy_hash.len() != 71
        || !request.policy_hash.starts_with("sha256:")
        || !request.policy_hash[7..]
            .chars()
            .all(|ch| ch.is_ascii_hexdigit())
    {
        return Err("invalid bounded launch request".into());
    }
    Ok(request)
}

fn connect_control(name: &str) -> Result<File, String> {
    let deadline = Instant::now() + Duration::from_secs(45);
    loop {
        match OpenOptions::new().read(true).write(true).open(name) {
            Ok(file) => return Ok(file),
            Err(error)
                if Instant::now() < deadline
                    && (error.kind() == io::ErrorKind::NotFound
                        || error.kind() == io::ErrorKind::WouldBlock
                        || error.raw_os_error() == Some(231)) =>
            {
                thread::sleep(Duration::from_millis(25));
            }
            Err(error) => return Err(error.to_string()),
        }
    }
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    let mut different = left.len() ^ right.len();
    let length = left.len().max(right.len());
    for index in 0..length {
        different |= usize::from(
            left.get(index).copied().unwrap_or(0) ^ right.get(index).copied().unwrap_or(0),
        );
    }
    different == 0
}

fn normalized_windows_path(path: &Path) -> String {
    let value = path.to_string_lossy();
    value
        .strip_prefix(r"\\?\")
        .unwrap_or(&value)
        .replace('/', "\\")
        .to_lowercase()
}

fn validate_no_reparse(path: &Path) -> Result<(), String> {
    for candidate in std::iter::once(path).chain(path.ancestors().skip(1)) {
        let metadata = std::fs::symlink_metadata(candidate).map_err(|error| {
            format!(
                "path validation failed for {}: {error}",
                candidate.display()
            )
        })?;
        if metadata.file_attributes() & 0x400 != 0 {
            return Err(format!(
                "MXC policy path contains a reparse point: {}",
                candidate.display()
            ));
        }
    }
    Ok(())
}

struct DirectoryIdentityGuard {
    _handle: File,
    volume_serial_number: u32,
    file_index: u64,
}

impl DirectoryIdentityGuard {
    fn matches(&self, expected: &ObjectIdentity) -> bool {
        self.volume_serial_number == expected.volume_serial_number
            && self.file_index == expected.file_id
    }
}

fn open_directory_identity(path: &Path) -> Result<DirectoryIdentityGuard, String> {
    let handle = OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
        .map_err(|error| {
            format!(
                "could not hold workdir identity {}: {error}",
                path.display()
            )
        })?;
    let metadata = handle.metadata().map_err(|error| {
        format!(
            "could not inspect workdir handle {}: {error}",
            path.display()
        )
    })?;
    if metadata.file_attributes() & 0x400 != 0 {
        return Err(format!(
            "the MXC workdir root is a reparse point: {}",
            path.display()
        ));
    }
    let mut information = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: `handle` remains live for the call and `information` is a valid
    // writable output structure.
    if unsafe { GetFileInformationByHandle(handle.as_raw_handle() as HANDLE, &mut information) }
        == 0
    {
        return Err(format!(
            "workdir object identity is unavailable for {}: {}",
            path.display(),
            io::Error::last_os_error()
        ));
    }
    let volume_serial_number = information.dwVolumeSerialNumber;
    let file_index =
        (u64::from(information.nFileIndexHigh) << 32) | u64::from(information.nFileIndexLow);
    Ok(DirectoryIdentityGuard {
        _handle: handle,
        volume_serial_number,
        file_index,
    })
}

fn open_file_identity(path: &Path) -> Result<DirectoryIdentityGuard, String> {
    let handle = OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
        .map_err(|error| {
            format!(
                "could not hold runtime identity {}: {error}",
                path.display()
            )
        })?;
    let metadata = handle.metadata().map_err(|error| {
        format!(
            "could not inspect runtime handle {}: {error}",
            path.display()
        )
    })?;
    if metadata.file_attributes() & 0x400 != 0 {
        return Err(format!(
            "the selected runtime is a reparse point: {}",
            path.display()
        ));
    }
    let mut information = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: `handle` remains live and `information` is a valid output value.
    if unsafe { GetFileInformationByHandle(handle.as_raw_handle() as HANDLE, &mut information) }
        == 0
    {
        return Err(format!(
            "runtime object identity is unavailable for {}: {}",
            path.display(),
            io::Error::last_os_error()
        ));
    }
    Ok(DirectoryIdentityGuard {
        _handle: handle,
        volume_serial_number: information.dwVolumeSerialNumber,
        file_index: (u64::from(information.nFileIndexHigh) << 32)
            | u64::from(information.nFileIndexLow),
    })
}

fn verify_directory_identity(path: &Path, held: &DirectoryIdentityGuard) -> Result<(), String> {
    let current = open_directory_identity(path)?;
    if current.volume_serial_number != held.volume_serial_number
        || current.file_index != held.file_index
    {
        return Err(format!(
            "the MXC workdir object changed before native dispatch: {}",
            path.display()
        ));
    }
    Ok(())
}

fn verify_file_identity(path: &Path, held: &DirectoryIdentityGuard) -> Result<(), String> {
    let current = open_file_identity(path)?;
    if current.volume_serial_number != held.volume_serial_number
        || current.file_index != held.file_index
    {
        return Err(format!(
            "the selected runtime object changed before native dispatch: {}",
            path.display()
        ));
    }
    Ok(())
}

fn file_link_count(path: &Path) -> Result<u32, String> {
    let handle = OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .open(path)
        .map_err(|error| {
            format!(
                "could not inspect file identity {}: {error}",
                path.display()
            )
        })?;
    let mut information = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: `handle` remains live and `information` is a valid output value.
    if unsafe { GetFileInformationByHandle(handle.as_raw_handle() as HANDLE, &mut information) }
        == 0
    {
        return Err(format!(
            "file identity is unavailable for {}: {}",
            path.display(),
            io::Error::last_os_error()
        ));
    }
    Ok(information.nNumberOfLinks)
}

fn validate_policy_paths(request: &LaunchRequest) -> Result<(), String> {
    let mut paths = request.readwrite_paths.clone();
    paths.extend(request.readonly_paths.iter().cloned());
    paths.push(request.cwd.clone());
    paths.push(request.runtime_path.clone());
    for value in &paths {
        if value.starts_with(r"\\") || value.starts_with(r"\\?\") || value.starts_with(r"\\.\") {
            return Err(format!("UNC and device paths are not supported: {value}"));
        }
        let path = PathBuf::from(value);
        if !path.is_absolute()
            || value
                .char_indices()
                .any(|(index, character)| character == ':' && index != 1)
        {
            return Err(format!(
                "relative and alternate-data-stream paths are not supported: {value}"
            ));
        }
        validate_no_reparse(&path)?;
        let canonical = path
            .canonicalize()
            .map_err(|error| format!("path canonicalization failed for {value}: {error}"))?;
        if normalized_windows_path(&canonical) != normalized_windows_path(&path) {
            return Err(format!(
                "MXC policy path identity changed before launch: {value}"
            ));
        }
    }
    let mut pending = vec![PathBuf::from(&request.cwd)];
    let mut seen = 0_usize;
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory)
            .map_err(|error| format!("workdir scan failed for {}: {error}", directory.display()))?
        {
            let entry = entry.map_err(|error| format!("workdir scan failed: {error}"))?;
            seen += 1;
            if seen > 50_000 {
                return Err("the MXC workdir is too large to validate reparse boundaries".into());
            }
            let metadata = std::fs::symlink_metadata(entry.path())
                .map_err(|error| format!("workdir metadata failed: {error}"))?;
            if metadata.file_attributes() & 0x400 != 0 {
                return Err(format!(
                    "the MXC workdir contains a reparse point: {}",
                    entry.path().display()
                ));
            }
            if metadata.is_dir() {
                pending.push(entry.path());
            } else if metadata.is_file() && file_link_count(&entry.path())? > 1 {
                return Err(format!(
                    "the MXC workdir contains a hard-linked file: {}",
                    entry.path().display()
                ));
            }
        }
    }
    Ok(())
}

fn read_control_frame(
    control: &mut File,
    bytes: &mut Vec<u8>,
) -> Result<Option<ControlRequest>, String> {
    let mut available = 0_u32;
    // SAFETY: `control` owns a valid named-pipe handle for the duration of the
    // call; the output pointer is valid and the unused buffer pointers are null.
    let result = unsafe {
        PeekNamedPipe(
            control.as_raw_handle() as HANDLE,
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            &mut available,
            std::ptr::null_mut(),
        )
    };
    if result == 0 {
        return Err(format!(
            "control channel unavailable: {}",
            io::Error::last_os_error()
        ));
    }
    if available == 0 {
        return Ok(None);
    }
    let mut chunk = vec![0_u8; usize::try_from(available).unwrap_or(usize::MAX).min(4096)];
    let count = control
        .read(&mut chunk)
        .map_err(|error| format!("control read failed: {error}"))?;
    if count == 0 {
        return Err("control channel disconnected".into());
    }
    bytes.extend_from_slice(&chunk[..count]);
    if bytes.len() > 65_536 {
        return Err("control request exceeds the protocol bound".into());
    }
    let Some(newline) = bytes.iter().position(|byte| *byte == b'\n') else {
        return Ok(None);
    };
    if bytes[newline + 1..]
        .iter()
        .any(|byte| !byte.is_ascii_whitespace())
    {
        return Err("duplicate control request".into());
    }
    serde_json::from_slice::<ControlRequest>(&bytes[..newline])
        .map(Some)
        .map_err(|error| format!("malformed control request: {error}"))
}

fn quote_windows_arg(value: &str) -> String {
    if !value.is_empty() && !value.chars().any(|c| c == ' ' || c == '\t' || c == '"') {
        return value.to_string();
    }
    let mut result = String::from("\"");
    let mut slashes = 0;
    for ch in value.chars() {
        if ch == '\\' {
            slashes += 1;
        } else if ch == '"' {
            result.push_str(&"\\".repeat(slashes * 2 + 1));
            result.push('"');
            slashes = 0;
        } else {
            result.push_str(&"\\".repeat(slashes));
            slashes = 0;
            result.push(ch);
        }
    }
    result.push_str(&"\\".repeat(slashes * 2));
    result.push('"');
    result
}

fn build_sandbox_request(request: &LaunchRequest) -> Result<SandboxRequest, String> {
    let mut network = NetworkSection::default();
    network.allow_outbound = request.allow_outbound;
    network.allow_local_network = request.allow_local_network;
    let policy = SandboxPolicy {
        version: request.schema_version.clone(),
        filesystem: Some(FilesystemSection {
            readwrite_paths: request.readwrite_paths.clone(),
            readonly_paths: request.readonly_paths.clone(),
            denied_paths: request.denied_paths.clone(),
            clear_policy_on_exit: Some(request.clear_policy_on_exit),
        }),
        network: Some(network),
        ui: None,
        timeout_ms: request.timeout_ms,
    };
    let command = request
        .argv
        .iter()
        .map(|arg| quote_windows_arg(arg))
        .collect::<Vec<_>>()
        .join(" ");
    let launch = build_request(&policy, &command, Some(&request.container_id))
        .map_err(|error| error.to_string())?;
    // This typed request setting is consumed by MXC's dispatcher in the same
    // operation that selects the tier and spawns the workload. A stale host
    // capability probe therefore cannot re-enable Tier 3.
    #[cfg(feature = "mxc-no-dacl-api")]
    {
        let mut launch = launch;
        launch.set_allow_dacl_mutation(false);
        if launch.allow_dacl_mutation() {
            return Err("MXC request did not retain fallback.allowDaclMutation=false".into());
        }
        launch.set_working_directory(request.cwd.clone());
        launch.set_env(request.environment.clone());
        Ok(launch)
    }
    #[cfg(not(feature = "mxc-no-dacl-api"))]
    {
        let _ = launch;
        Err(
            "the pinned MXC build lacks atomic no-DACL admission; apply the approved upstream extension"
                .into(),
        )
    }
}

#[cfg(feature = "mxc-no-dacl-api")]
fn effective_tier(sandbox: &mxc_sdk::Sandbox) -> Result<&'static str, String> {
    sandbox
        .isolation_tier()
        .ok_or_else(|| "MXC did not return structured effective-tier evidence".into())
}

#[cfg(not(feature = "mxc-no-dacl-api"))]
fn effective_tier(_sandbox: &mxc_sdk::Sandbox) -> Result<&'static str, String> {
    Err("the pinned MXC build does not expose structured effective-tier evidence".into())
}

fn main() {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments == ["--identity"] {
        if emit_identity().is_err() {
            std::process::exit(74);
        }
        return;
    }
    if !arguments.is_empty() {
        eprintln!("unsupported runner arguments");
        std::process::exit(64);
    }
    inject_failure("before_request_creation");
    let request = match read_request() {
        Ok(value) => value,
        Err(message) => {
            eprintln!("MXC request rejected: {message}");
            std::process::exit(64);
        }
    };
    let mut control = match connect_control(&request.control_pipe) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("MXC control connection failed: {error}");
            std::process::exit(69);
        }
    };
    if send(&mut control, &request, "HELLO").is_err() {
        std::process::exit(69);
    }

    if let Err(message) = verify_policy_hash(&request) {
        fail(
            &mut control,
            &request,
            "policy",
            "policy_hash_mismatch",
            message,
        );
    }

    if let Err(message) = validate_policy_paths(&request) {
        fail(
            &mut control,
            &request,
            "policy",
            "unsafe_path_boundary",
            message,
        );
    }
    let workdir = PathBuf::from(&request.cwd);
    let workdir_guard = match open_directory_identity(&workdir) {
        Ok(value) => value,
        Err(message) => fail(
            &mut control,
            &request,
            "policy",
            "workdir_identity_unavailable",
            message,
        ),
    };
    if !workdir_guard.matches(&request.workdir_identity) {
        fail(
            &mut control,
            &request,
            "policy",
            "workdir_identity_changed",
            "the workdir object does not match the hashed Python identity".into(),
        );
    }
    let runtime_path = PathBuf::from(&request.runtime_path);
    let runtime_guard = match open_file_identity(&runtime_path) {
        Ok(value) => value,
        Err(message) => fail(
            &mut control,
            &request,
            "policy",
            "runtime_identity_unavailable",
            message,
        ),
    };
    if !runtime_guard.matches(&request.runtime_identity) {
        fail(
            &mut control,
            &request,
            "policy",
            "runtime_identity_changed",
            "the selected runtime does not match the hashed Python identity".into(),
        );
    }
    inject_failure("after_policy_validation");

    let launch = match build_sandbox_request(&request) {
        Ok(value) => value,
        Err(message) => fail(&mut control, &request, "policy", "policy_rejected", message),
    };
    if send(&mut control, &request, "ACCEPTED").is_err() {
        std::process::exit(69);
    }

    inject_failure("before_spawn");
    if let Err(message) = verify_directory_identity(&workdir, &workdir_guard) {
        fail(
            &mut control,
            &request,
            "policy",
            "workdir_identity_changed",
            message,
        );
    }
    if let Err(message) = verify_file_identity(&runtime_path, &runtime_guard) {
        fail(
            &mut control,
            &request,
            "policy",
            "runtime_identity_changed",
            message,
        );
    }

    let mut sandbox = match spawn_sandbox(launch) {
        Ok(value) => value,
        Err(error) => fail(
            &mut control,
            &request,
            "launch",
            "sandbox_launch_failed",
            error.to_string(),
        ),
    };
    drop(workdir_guard);
    drop(runtime_guard);
    inject_failure("after_spawn_before_started");
    let backend_tier = match effective_tier(&sandbox) {
        Ok("base-container") => "base-container",
        Ok(other) => {
            let _ = sandbox.kill();
            let _ = sandbox.wait();
            fail(
                &mut control,
                &request,
                "admission",
                "unexpected_effective_tier",
                format!("MXC selected forbidden ProcessContainer tier: {other}"),
            )
        }
        Err(message) => {
            let _ = sandbox.kill();
            let _ = sandbox.wait();
            fail(
                &mut control,
                &request,
                "admission",
                "missing_effective_tier",
                message,
            )
        }
    };
    // Studio's one-shot Python and Terminal tools have no interactive stdin. Taking and
    // immediately dropping MXC's pipe gives the workload EOF and prevents it
    // from inheriting or blocking on the server's stdin.
    drop(sandbox.take_stdin());
    let stdout = sandbox.take_stdout();
    let stderr = sandbox.take_stderr();
    let out_thread = thread::spawn(move || {
        if let Some(mut source) = stdout {
            let mut target = io::stdout().lock();
            let _ = io::copy(&mut source, &mut target);
            let _ = target.flush();
        }
    });
    let err_thread = thread::spawn(move || {
        if let Some(mut source) = stderr {
            let mut target = io::stderr().lock();
            let _ = io::copy(&mut source, &mut target);
            let _ = target.flush();
        }
    });
    if send_full(
        &mut control,
        &request,
        "STARTED",
        None,
        None,
        None,
        None,
        None,
        None,
        Some(backend_tier),
    )
    .is_err()
    {
        let _ = sandbox.kill();
        let _ = sandbox.wait();
        std::process::exit(69);
    }
    inject_failure("after_started");
    let mut cancelled = false;
    let mut control_bytes = Vec::new();
    let started_at = Instant::now();
    let outcome = loop {
        inject_failure("while_streaming");
        match sandbox.try_wait() {
            // `try_wait` reaps the process. Calling `wait` afterwards would lose
            // the terminal receipt on Windows.
            Ok(Some(code)) => break WaitOutcome::Exited(code),
            Ok(None) => {}
            Err(error) => fail(
                &mut control,
                &request,
                "completion",
                "wait_failed",
                error.to_string(),
            ),
        }
        // The SDK's non-blocking `try_wait` does not drive its blocking-wait
        // timeout path. Enforce the same trusted request deadline here so the
        // supervisor can remain responsive to authenticated cancellation.
        if request
            .timeout_ms
            .is_some_and(|ms| started_at.elapsed() >= Duration::from_millis(ms.into()))
        {
            let _ = sandbox.kill();
            match sandbox.wait() {
                Ok(_) => break WaitOutcome::TimedOut,
                Err(error) => fail(
                    &mut control,
                    &request,
                    "completion",
                    "timeout_wait_failed",
                    error.to_string(),
                ),
            }
        }
        match read_control_frame(&mut control, &mut control_bytes) {
            Err(message) => {
                let _ = sandbox.kill();
                let _ = sandbox.wait();
                fail(&mut control, &request, "protocol", "control_lost", message)
            }
            Ok(Some(frame)) => {
                if frame.v != PROTOCOL
                    || frame.event != "CANCEL"
                    || !constant_time_eq(&frame.run_id, &request.run_id)
                    || !constant_time_eq(&frame.token, &request.token)
                {
                    let _ = sandbox.kill();
                    fail(
                        &mut control,
                        &request,
                        "protocol",
                        "unauthenticated_control",
                        "invalid cancellation request".into(),
                    );
                }
                cancelled = true;
                let _ = sandbox.kill();
                match sandbox.wait() {
                    Ok(value) => break value,
                    Err(error) => fail(
                        &mut control,
                        &request,
                        "completion",
                        "cancel_wait_failed",
                        error.to_string(),
                    ),
                }
            }
            Ok(None) => {}
        }
        thread::sleep(Duration::from_millis(50));
    };
    let _ = out_thread.join();
    let _ = err_thread.join();
    let warnings = sandbox.warnings();
    let (mut exit_code, timed_out) = match outcome {
        WaitOutcome::Exited(code) => (code, false),
        WaitOutcome::TimedOut => (124, true),
    };
    if cancelled {
        exit_code = 130;
    }
    let cleanup = if warnings.is_empty() {
        "complete"
    } else {
        "warning"
    };
    let message = if warnings.is_empty() {
        None
    } else {
        Some(warnings.join("; "))
    };
    inject_failure("before_finished");
    if send_full(
        &mut control,
        &request,
        "FINISHED",
        None,
        None,
        message,
        Some(exit_code),
        Some(timed_out),
        Some(cleanup),
        Some(backend_tier),
    )
    .is_err()
    {
        std::process::exit(69);
    }
    inject_failure("after_finished_before_cleanup");
    drop(control);
    std::process::exit(exit_code.clamp(0, 255));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> LaunchRequest {
        let runtime = std::env::current_exe().expect("current executable");
        let cwd = std::env::temp_dir();
        let runtime_guard = open_file_identity(&runtime).expect("runtime identity");
        let workdir_guard = open_directory_identity(&cwd).expect("workdir identity");
        LaunchRequest {
            protocol: PROTOCOL,
            run_id: "0123456789abcdef".into(),
            token: "0".repeat(64),
            control_pipe: r"\\.\pipe\unsloth-mxc-test".into(),
            profile_id: PROFILE_ID.into(),
            profile_version: PROFILE_VERSION,
            policy_hash: format!("sha256:{}", "0".repeat(64)),
            schema_version: MXC_SCHEMA_VERSION.into(),
            runtime_revision: MXC_REVISION.into(),
            container_id: "unsloth-0123456789abcdef".into(),
            argv: vec![runtime.to_string_lossy().into_owned(), "--version".into()],
            execution_kind: "python".into(),
            runtime_path: runtime.to_string_lossy().into_owned(),
            runtime_identity: ObjectIdentity {
                volume_serial_number: runtime_guard.volume_serial_number,
                file_id: runtime_guard.file_index,
            },
            cwd: cwd.to_string_lossy().into_owned(),
            workdir_identity: ObjectIdentity {
                volume_serial_number: workdir_guard.volume_serial_number,
                file_id: workdir_guard.file_index,
            },
            environment: BTreeMap::new(),
            environment_policy: "sanitized-explicit-v1".into(),
            command_line_policy: "windows-createprocess-argv-v1".into(),
            readwrite_paths: vec![cwd.to_string_lossy().into_owned()],
            readonly_paths: Vec::new(),
            denied_paths: Vec::new(),
            clear_policy_on_exit: true,
            network_profile: "compatibility".into(),
            allow_outbound: true,
            allow_local_network: true,
            ui_policy: None,
            timeout_ms: Some(1000),
            allow_dacl_mutation: false,
            admission: "atomic-no-dacl-fallback".into(),
        }
    }

    fn vector_request() -> LaunchRequest {
        LaunchRequest {
            protocol: PROTOCOL,
            run_id: "0123456789abcdef0123456789abcdef".into(),
            token: "0".repeat(64),
            control_pipe: r"\\.\pipe\unsloth-mxc-vector".into(),
            profile_id: PROFILE_ID.into(),
            profile_version: PROFILE_VERSION,
            policy_hash: String::new(),
            schema_version: MXC_SCHEMA_VERSION.into(),
            runtime_revision: MXC_REVISION.into(),
            container_id: "unsloth-0123456789abcdef0123456789abcdef".into(),
            argv: vec![
                r"C:\Program Files\Python\python.exe".into(),
                "-c".into(),
                "print('会話')".into(),
            ],
            execution_kind: "python".into(),
            runtime_path: r"C:\Program Files\Python\python.exe".into(),
            runtime_identity: ObjectIdentity {
                volume_serial_number: 111,
                file_id: 222,
            },
            cwd: r"D:\会話\work".into(),
            workdir_identity: ObjectIdentity {
                volume_serial_number: 333,
                file_id: 444,
            },
            environment: BTreeMap::from([
                ("Path".into(), r"C:\Windows\System32".into()),
                ("UNICODE".into(), "é".into()),
            ]),
            environment_policy: "sanitized-explicit-v1".into(),
            command_line_policy: "windows-createprocess-argv-v1".into(),
            readwrite_paths: vec![r"D:\会話\work".into()],
            readonly_paths: vec![r"C:\Program Files\Python".into(), r"C:\Windows".into()],
            denied_paths: Vec::new(),
            clear_policy_on_exit: true,
            network_profile: "compatibility".into(),
            allow_outbound: true,
            allow_local_network: true,
            ui_policy: None,
            timeout_ms: Some(30_000),
            allow_dacl_mutation: false,
            admission: "atomic-no-dacl-fallback".into(),
        }
    }

    #[cfg(feature = "mxc-no-dacl-api")]
    #[test]
    fn native_request_disables_dacl_fallback_before_spawn() {
        let launch = build_sandbox_request(&request()).expect("request should build");
        assert!(!launch.allow_dacl_mutation());
    }

    #[cfg(not(feature = "mxc-no-dacl-api"))]
    #[test]
    fn unextended_mxc_build_refuses_before_spawn() {
        let error = build_sandbox_request(&request()).expect_err("unextended MXC must fail closed");
        assert!(error.contains("atomic no-DACL admission"));
    }

    #[test]
    fn authentication_comparison_covers_length_and_content() {
        assert!(constant_time_eq("same", "same"));
        assert!(!constant_time_eq("same", "different"));
        assert!(!constant_time_eq("same", "samf"));
    }

    #[test]
    fn policy_hash_is_recomputed_from_every_security_field() {
        let mut value = request();
        value.policy_hash = recompute_policy_hash(&value).expect("hash");
        verify_policy_hash(&value).expect("matching policy identity");
        value.allow_dacl_mutation = true;
        let error = verify_policy_hash(&value).expect_err("mutation must be refused");
        assert!(error.contains("policy hash mismatch"));
    }

    #[test]
    fn rust_matches_python_policy_hash_vectors() {
        let expected: BTreeMap<String, String> =
            serde_json::from_str(include_str!("../tests/policy_hash_vectors.json"))
                .expect("shared policy hash vectors");
        let assert_vector = |name: &str, request: &LaunchRequest| {
            assert_eq!(
                recompute_policy_hash(request).expect("hash"),
                expected[name],
                "policy vector {name}"
            );
        };

        let mut value = vector_request();
        assert_vector("python_unicode", &value);
        value.argv = vec![
            r"C:\Windows\System32\cmd.exe".into(),
            "/d".into(),
            "/s".into(),
            "/c".into(),
            "echo %VAR% && (echo \"quoted\")".into(),
        ];
        value.execution_kind = "terminal".into();
        value.runtime_path = r"C:\Windows\System32\cmd.exe".into();
        value.environment.clear();
        value.timeout_ms = None;
        assert_vector("terminal_empty_optional", &value);

        value = vector_request();
        value.cwd = r"E:\alternate\work".into();
        value.readwrite_paths = vec![r"E:\alternate\work".into()];
        assert_vector("different_drive", &value);
        value = vector_request();
        value.allow_local_network = false;
        assert_vector("network_change", &value);
        value = vector_request();
        value.readonly_paths.push(r"C:\Extra".into());
        assert_vector("filesystem_change", &value);
        value = vector_request();
        value.allow_dacl_mutation = true;
        assert_vector("dacl_change", &value);
        value = vector_request();
        value.cwd = r"D:\会話\other".into();
        assert_vector("cwd_change", &value);
        value = vector_request();
        value.runtime_path = r"C:\Python\python.exe".into();
        value.argv[0] = r"C:\Python\python.exe".into();
        assert_vector("executable_change", &value);
        value = vector_request();
        value.environment.insert("EXTRA".into(), "1".into());
        assert_vector("environment_change", &value);
    }
}
