#[cfg(windows)]
use log::{info, warn};
#[cfg(windows)]
use std::mem::size_of;
#[cfg(windows)]
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
#[cfg(windows)]
use std::sync::OnceLock;
#[cfg(windows)]
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
#[cfg(windows)]
use windows_sys::Win32::System::Threading::GetCurrentProcess;

#[cfg(windows)]
static APP_JOB_INITIALIZED: OnceLock<()> = OnceLock::new();

pub fn initialize() {
    #[cfg(windows)]
    {
        if APP_JOB_INITIALIZED.get().is_some() {
            return;
        }

        match unsafe { create_app_job_object() } {
            Ok(()) => {
                let _ = APP_JOB_INITIALIZED.set(());
                info!("Windows app job object initialized for crash-safe child cleanup");
            }
            Err(err) => {
                warn!(
                    "Failed to initialize Windows app job object; crash cleanup will rely on explicit stop paths: {}",
                    err
                );
            }
        }
    }
}

#[cfg(windows)]
unsafe fn create_app_job_object() -> std::io::Result<()> {
    let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
    if job.is_null() {
        return Err(std::io::Error::last_os_error());
    }

    let job = OwnedHandle::from_raw_handle(job);

    let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

    if SetInformationJobObject(
        job.as_raw_handle(),
        JobObjectExtendedLimitInformation,
        &limits as *const _ as *const _,
        size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
    ) == 0
    {
        return Err(std::io::Error::last_os_error());
    }

    if AssignProcessToJobObject(job.as_raw_handle(), GetCurrentProcess()) == 0 {
        return Err(std::io::Error::last_os_error());
    }

    // Keep the job handle alive for the full app lifetime.
    std::mem::forget(job);
    Ok(())
}
