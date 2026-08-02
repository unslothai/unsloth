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
    QueryInformationJobObject, SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
#[cfg(windows)]
use windows_sys::Win32::System::Threading::GetCurrentProcess;

#[cfg(windows)]
static APP_JOB: OnceLock<Option<OwnedHandle>> = OnceLock::new();

pub fn initialize() {
    #[cfg(windows)]
    {
        APP_JOB.get_or_init(|| match unsafe { create_app_job_object() } {
            Ok(job) => {
                info!("Windows app job object initialized for crash-safe child cleanup");
                Some(job)
            }
            Err(err) => {
                warn!(
                    "Failed to initialize Windows app job object; crash cleanup will rely on explicit stop paths: {}",
                    err
                );
                None
            }
        });
    }
}

#[cfg(windows)]
unsafe fn create_app_job_object() -> std::io::Result<OwnedHandle> {
    let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
    if job.is_null() {
        return Err(std::io::Error::last_os_error());
    }

    let job = OwnedHandle::from_raw_handle(job);

    set_kill_on_close(&job, true)?;

    if AssignProcessToJobObject(job.as_raw_handle(), GetCurrentProcess()) == 0 {
        return Err(std::io::Error::last_os_error());
    }

    Ok(job)
}

#[cfg(windows)]
unsafe fn query_job_limits(
    job: &OwnedHandle,
) -> std::io::Result<JOBOBJECT_EXTENDED_LIMIT_INFORMATION> {
    let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
    if QueryInformationJobObject(
        job.as_raw_handle(),
        JobObjectExtendedLimitInformation,
        &mut limits as *mut _ as *mut _,
        size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        std::ptr::null_mut(),
    ) == 0
    {
        return Err(std::io::Error::last_os_error());
    }
    Ok(limits)
}

#[cfg(windows)]
unsafe fn set_kill_on_close(job: &OwnedHandle, enabled: bool) -> std::io::Result<()> {
    let mut limits = query_job_limits(job)?;

    if enabled {
        limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    } else {
        limits.BasicLimitInformation.LimitFlags &= !JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    }

    if SetInformationJobObject(
        job.as_raw_handle(),
        JobObjectExtendedLimitInformation,
        &limits as *const _ as *const _,
        size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
    ) == 0
    {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Keep the updater-launched installer alive after Tauri exits this process.
///
/// The updater calls this from its native pre-exit hook after extraction and
/// immediately before it launches the installer.
#[cfg(windows)]
pub fn suspend_for_update_installer() -> std::io::Result<()> {
    let Some(job) = APP_JOB.get().and_then(Option::as_ref) else {
        return Ok(());
    };

    unsafe { set_kill_on_close(job, false) }?;
    info!("Windows job cleanup suspended for updater installer launch");
    Ok(())
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use std::process::{Child, Command, Stdio};
    use std::thread;
    use std::time::Duration;
    use windows_sys::Win32::Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT};
    use windows_sys::Win32::System::JobObjects::JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
    use windows_sys::Win32::System::Threading::WaitForSingleObject;

    const CHILD_PROCESS_ENV: &str = "UNSLOTH_WINDOWS_JOB_OBJECT_TEST_CHILD";

    unsafe fn limit_flags(job: &OwnedHandle) -> std::io::Result<u32> {
        Ok(query_job_limits(job)?.BasicLimitInformation.LimitFlags)
    }

    unsafe fn empty_job() -> std::io::Result<OwnedHandle> {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        Ok(OwnedHandle::from_raw_handle(job))
    }

    unsafe fn spawn_assigned_test_child(job: &OwnedHandle) -> std::io::Result<Child> {
        let mut child = Command::new(std::env::current_exe()?)
            .arg("job_object_survival_child")
            .env(CHILD_PROCESS_ENV, "1")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;

        if AssignProcessToJobObject(job.as_raw_handle(), child.as_raw_handle()) == 0 {
            let error = std::io::Error::last_os_error();
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }

        thread::sleep(Duration::from_millis(50));
        if let Some(status) = child.try_wait()? {
            return Err(std::io::Error::other(format!(
                "job test child exited before verification: {status}"
            )));
        }
        Ok(child)
    }

    #[test]
    fn job_object_survival_child() {
        if std::env::var_os(CHILD_PROCESS_ENV).is_some() {
            thread::sleep(Duration::from_secs(30));
        }
    }

    #[test]
    fn kill_on_close_can_be_suspended_and_restored() -> std::io::Result<()> {
        unsafe {
            let job = empty_job()?;

            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
            assert_ne!(
                SetInformationJobObject(
                    job.as_raw_handle(),
                    JobObjectExtendedLimitInformation,
                    &limits as *const _ as *const _,
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                ),
                0
            );

            set_kill_on_close(&job, true)?;
            let flags = limit_flags(&job)?;
            assert_ne!(flags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, 0);
            assert_ne!(flags & JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION, 0);

            set_kill_on_close(&job, false)?;
            let flags = limit_flags(&job)?;
            assert_eq!(flags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, 0);
            assert_ne!(flags & JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION, 0);

            set_kill_on_close(&job, true)?;
            assert_ne!(limit_flags(&job)? & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, 0);
        }
        Ok(())
    }

    #[test]
    fn suspended_kill_on_close_keeps_the_installer_process_alive() -> std::io::Result<()> {
        unsafe {
            let killing_job = empty_job()?;
            set_kill_on_close(&killing_job, true)?;
            let mut killed_child = spawn_assigned_test_child(&killing_job)?;
            drop(killing_job);

            let killed_wait = WaitForSingleObject(killed_child.as_raw_handle(), 5_000);
            if killed_wait != WAIT_OBJECT_0 {
                let _ = killed_child.kill();
                let _ = killed_child.wait();
                return Err(std::io::Error::other(format!(
                    "kill-on-close wait returned {killed_wait}, expected {WAIT_OBJECT_0}"
                )));
            }
            killed_child.wait()?;

            let suspended_job = empty_job()?;
            set_kill_on_close(&suspended_job, true)?;
            set_kill_on_close(&suspended_job, false)?;
            let mut surviving_child = spawn_assigned_test_child(&suspended_job)?;
            drop(suspended_job);

            let surviving_wait = WaitForSingleObject(surviving_child.as_raw_handle(), 250);
            let _ = surviving_child.kill();
            let _ = surviving_child.wait();
            if surviving_wait != WAIT_TIMEOUT {
                return Err(std::io::Error::other(format!(
                    "suspended kill-on-close wait returned {surviving_wait}, expected {WAIT_TIMEOUT}"
                )));
            }
        }
        Ok(())
    }
}
