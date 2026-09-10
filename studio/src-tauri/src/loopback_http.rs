use std::time::Duration;

/// Redirects are refused for the same reason `streaming_client` refuses them, and it matters
/// more here: this is the client that posts `.desktop_secret` to `/api/auth/desktop-login`.
/// reqwest follows up to 10 redirects by default, and its cross-host protection strips headers,
/// not bodies, so a responder answering 307 would carry the secret off-host after the loopback
/// URL had already been checked.
pub(crate) fn client(timeout: Duration) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()
}

/// A client for streaming a body down. An optional per-read deadline normally bounds headers
/// and every chunk. The log archive path disables it so the caller can separately allow its
/// expensive first body chunk while still timing out headers and later reads. Redirects are
/// refused so a loopback URL cannot be bounced off-host after the check.
pub(crate) fn streaming_client(
    connect_timeout: Duration,
    read_timeout: Option<Duration>,
) -> Result<reqwest::Client, reqwest::Error> {
    let mut builder = reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(connect_timeout)
        .redirect(reqwest::redirect::Policy::none());
    if let Some(read_timeout) = read_timeout {
        builder = builder.read_timeout(read_timeout);
    }
    builder.build()
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

    /// The desktop secret rides this client to /api/auth/desktop-login. A 307
    /// preserves the method and body, so a followed redirect would hand the
    /// secret to whatever the Location header names. Refuse instead: the caller
    /// sees the 307 itself and treats it as a failed login.
    #[test]
    fn a_redirect_is_returned_not_followed() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut discard = [0_u8; 2048];
            let _ = stream.read(&mut discard);
            let _ = stream.write_all(
                b"HTTP/1.1 307 Temporary Redirect\r\nLocation: http://evil.test/collect\r\n\
                  Content-Length: 0\r\nConnection: close\r\n\r\n",
            );
        });

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let response = runtime.block_on(async {
            super::client(Duration::from_secs(2))
                .unwrap()
                .post(format!("http://127.0.0.1:{port}/api/auth/desktop-login"))
                .json(&serde_json::json!({ "secret": "desktop-not-a-real-secret" }))
                .send()
                .await
                .unwrap()
        });

        assert_eq!(response.status().as_u16(), 307);
        // Still the port we dialled: nothing was re-sent anywhere else.
        assert_eq!(response.url().port(), Some(port));
        server.join().unwrap();
    }
}
