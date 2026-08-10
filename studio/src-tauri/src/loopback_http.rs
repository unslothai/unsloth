use std::time::Duration;

pub(crate) fn client(timeout: Duration) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(timeout)
        .build()
}

/// A client for streaming a body down. `read_timeout`, not `timeout`: a whole gallery clip
/// outlasts any sane total deadline, while a per-read one bounds the headers and then each
/// chunk, so a backend that accepts and goes quiet cannot hang the save. Redirects are refused
/// so a loopback URL cannot be bounced off-host after the check.
pub(crate) fn streaming_client(
    connect_timeout: Duration,
    read_timeout: Duration,
) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(connect_timeout)
        .read_timeout(read_timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::process::Command;
    use std::time::Duration;

    const CHILD_ENV: &str = "UNSLOTH_TEST_LOOPBACK_HTTP_CHILD";

    #[test]
    fn client_ignores_system_proxy() {
        if std::env::var_os(CHILD_ENV).is_none() {
            let current_thread = std::thread::current();
            let test_name = current_thread.name().unwrap();
            let status = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", test_name, "--nocapture"])
                .env(CHILD_ENV, "1")
                .env("HTTP_PROXY", "http://127.0.0.1:1")
                .env("http_proxy", "http://127.0.0.1:1")
                .env_remove("HTTPS_PROXY")
                .env_remove("https_proxy")
                .env_remove("ALL_PROXY")
                .env_remove("all_proxy")
                .env_remove("NO_PROXY")
                .env_remove("no_proxy")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 1024];
            let _ = stream.read(&mut request).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                .unwrap();
        });

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let response = runtime.block_on(async {
            super::client(Duration::from_secs(2))
                .unwrap()
                .get(format!("http://127.0.0.1:{port}/health"))
                .send()
                .await
                .unwrap()
        });
        assert!(response.status().is_success());
        server.join().unwrap();
    }
}
