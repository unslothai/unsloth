// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::windows::fs::MetadataExt;
use std::os::windows::io::AsRawHandle;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use mxc_sdk::policy::{FilesystemSection, NetworkSection};
use mxc_sdk::{build_request, spawn_sandbox, SandboxPolicy, SandboxRequest, WaitOutcome};
use serde::{Deserialize, Serialize};
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::Pipes::PeekNamedPipe;

const PROTOCOL: u32 = 1;
const MAX_REQUEST: usize = 262_144;
const MAX_ITEMS: usize = 512;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct LaunchRequest {
    protocol: u32,
    run_id: String,
    token: String,
    control_pipe: String,
    profile_id: String,
    policy_hash: String,
    schema_version: String,
    runtime_revision: String,
    container_id: String,
    argv: Vec<String>,
    execution_kind: String,
    runtime_path: String,
    cwd: String,
    environment: BTreeMap<String, String>,
    readwrite_paths: Vec<String>,
    readonly_paths: Vec<String>,
    timeout_ms: Option<u32>,
    allow_dacl_mutation: bool,
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
    if request.schema_version != "0.8.0-alpha"
        || request.runtime_revision != "ca7ea12ac6bd9f5420d6adecb37e32a8158da476"
        || request.profile_id != "unsloth-mxc-windows-basecontainer-v1"
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
        || request.container_id != format!("unsloth-{}", request.run_id)
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
    network.allow_outbound = true;
    network.allow_local_network = true;
    let policy = SandboxPolicy {
        version: request.schema_version.clone(),
        filesystem: Some(FilesystemSection {
            readwrite_paths: request.readwrite_paths.clone(),
            readonly_paths: request.readonly_paths.clone(),
            denied_paths: Vec::new(),
            clear_policy_on_exit: Some(true),
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

    if let Err(message) = validate_policy_paths(&request) {
        fail(
            &mut control,
            &request,
            "policy",
            "unsafe_path_boundary",
            message,
        );
    }

    let launch = match build_sandbox_request(&request) {
        Ok(value) => value,
        Err(message) => fail(&mut control, &request, "policy", "policy_rejected", message),
    };
    if send(&mut control, &request, "ACCEPTED").is_err() {
        std::process::exit(69);
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
    let mut cancelled = false;
    let mut control_bytes = Vec::new();
    let started_at = Instant::now();
    let outcome = loop {
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
    drop(control);
    std::process::exit(exit_code.clamp(0, 255));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> LaunchRequest {
        let runtime = std::env::current_exe().expect("current executable");
        let cwd = std::env::temp_dir();
        LaunchRequest {
            protocol: PROTOCOL,
            run_id: "0123456789abcdef".into(),
            token: "0".repeat(64),
            control_pipe: r"\\.\pipe\unsloth-mxc-test".into(),
            profile_id: "unsloth-mxc-windows-basecontainer-v1".into(),
            policy_hash: format!("sha256:{}", "0".repeat(64)),
            schema_version: "0.8.0-alpha".into(),
            runtime_revision: "ca7ea12ac6bd9f5420d6adecb37e32a8158da476".into(),
            container_id: "unsloth-0123456789abcdef".into(),
            argv: vec![runtime.to_string_lossy().into_owned(), "--version".into()],
            execution_kind: "python".into(),
            runtime_path: runtime.to_string_lossy().into_owned(),
            cwd: cwd.to_string_lossy().into_owned(),
            environment: BTreeMap::new(),
            readwrite_paths: vec![cwd.to_string_lossy().into_owned()],
            readonly_paths: Vec::new(),
            timeout_ms: Some(1000),
            allow_dacl_mutation: false,
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
}
