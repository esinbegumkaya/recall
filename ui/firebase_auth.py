from __future__ import annotations

import base64
import http.server
import json
import socketserver
import threading
from typing import Any, Dict, Optional

import streamlit as st
import streamlit.components.v1 as components


def get_firebase_config() -> Optional[Dict[str, str]]:
    try:
        config = st.secrets.get("firebase")
    except Exception:
        config = None

    if not config:
        return None

    required = ("apiKey", "authDomain", "projectId", "appId")
    if not all(config.get(key) for key in required):
        return None

    return dict(config)


def decode_id_token(token: str) -> Dict[str, Any]:
    try:
        payload_segment = token.split(".")[1]
        padded = payload_segment + "=" * (-len(payload_segment) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("utf-8"))
        return json.loads(raw)
    except Exception:
        return {}


def consume_auth_redirect() -> Optional[Dict[str, Any]]:
    params = st.query_params

    token = params.get("fb_token")
    if not token:
        return None

    claims = decode_id_token(token)

    profile = {
        "id_token": token,
        "uid": claims.get("user_id") or claims.get("sub") or "",
        "email": params.get("fb_email") or claims.get("email") or "",
        "name": params.get("fb_name") or claims.get("name") or "",
        "provider": params.get("fb_provider") or "password",
    }

    st.query_params.clear()

    return profile


_AUTH_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Recall — Sign In</title>
<style>
  :root {
    --fb-bg: #F8FAFF; --fb-text: #17203A; --fb-muted: #565D80;
    --fb-border: rgba(23,32,58,0.18); --fb-input-bg: #FFFFFF;
    --fb-primary: #5962D7; --fb-primary-hover: #454CB8;
    --fb-google-bg: #FFFFFF; --fb-google-border: rgba(23,32,58,0.22);
    --fb-error: #D64545;
  }
  body.theme-dark {
    --fb-bg: #0B1120; --fb-text: #E8EDF8; --fb-muted: #A9B0D6;
    --fb-border: rgba(255,255,255,0.16); --fb-input-bg: rgba(255,255,255,0.08);
    --fb-google-bg: rgba(255,255,255,0.06); --fb-google-border: rgba(255,255,255,0.20);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 1.6rem;
    font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--fb-bg); color: var(--fb-text);
  }
  h1 { font-size: 1.15rem; margin: 0 0 0.15rem 0; }
  p.sub { color: var(--fb-muted); font-size: 0.82rem; margin: 0 0 1.1rem 0; }
  input {
    width: 100%; padding: 0.72rem 0.85rem; margin-bottom: 0.6rem;
    border-radius: 10px; border: 1px solid var(--fb-border);
    background: var(--fb-input-bg); color: var(--fb-text); font-size: 0.92rem;
    outline: none;
  }
  input:focus { border-color: var(--fb-primary); }
  input:-webkit-autofill, input:-webkit-autofill:hover, input:-webkit-autofill:focus {
    -webkit-text-fill-color: var(--fb-text) !important;
    -webkit-box-shadow: 0 0 0px 1000px var(--fb-input-bg) inset !important;
    transition: background-color 9999s ease-in-out 0s;
  }
  .btn-primary {
    width: 100%; padding: 0.72rem 0.85rem; border-radius: 10px; border: none;
    background: var(--fb-primary); color: #fff; font-weight: 650; font-size: 0.92rem;
    cursor: pointer; margin-bottom: 0.55rem;
  }
  .btn-primary:hover { background: var(--fb-primary-hover); }
  .btn-primary:disabled { opacity: 0.6; cursor: default; }
  .btn-secondary {
    flex: 1; padding: 0.72rem 0.85rem; border-radius: 10px;
    border: 1px solid var(--fb-border); background: transparent;
    color: var(--fb-text); font-weight: 650; font-size: 0.92rem; cursor: pointer;
  }
  .btn-secondary:hover { background: var(--fb-input-bg); }
  .btn-secondary:disabled { opacity: 0.6; cursor: default; }
  .btn-google {
    width: 100%; padding: 0.68rem 0.85rem; border-radius: 10px;
    border: 1px solid var(--fb-google-border); background: var(--fb-google-bg);
    color: var(--fb-text); font-weight: 600; font-size: 0.88rem; cursor: pointer;
    display: flex; align-items: center; justify-content: center; gap: 0.5rem;
    margin-bottom: 0.6rem;
  }
  .divider {
    display: flex; align-items: center; gap: 0.6rem; color: var(--fb-muted);
    font-size: 0.74rem; margin: 0.7rem 0;
  }
  .divider::before, .divider::after { content: ""; flex: 1; height: 1px; background: var(--fb-border); }
  .switch { text-align: center; font-size: 0.82rem; color: var(--fb-muted); margin-top: 0.3rem; }
  .switch a, .forgot { color: var(--fb-primary); cursor: pointer; font-weight: 600; text-decoration: none; }
  .forgot { display: block; text-align: right; font-size: 0.78rem; margin: -0.25rem 0 0.75rem 0; }
  #status { min-height: 1.1rem; font-size: 0.82rem; color: var(--fb-error); margin-bottom: 0.4rem; }
  #status.ok { color: #20B486; }
  #continue-link {
    display: block; margin: -0.1rem 0 0.75rem 0; font-size: 0.88rem;
    font-weight: 700; color: var(--fb-primary); text-decoration: none;
  }
  #continue-link:hover { text-decoration: underline; }
</style>
</head>
<body>
  <h1>Sign in to Recall</h1>
  <p class="sub">Authentication is handled by Firebase.</p>

  <div id="status"></div>

  <input id="email" type="email" placeholder="name@example.com" autocomplete="email" />
  <input id="password" type="password" placeholder="Password" autocomplete="current-password" />
  <a class="forgot" onclick="forgotPassword()">Forgot password</a>

  <div style="display:flex; gap:0.5rem; margin-bottom: 0.6rem;">
    <button class="btn-primary" id="signin-btn" style="margin-bottom:0; width:auto; flex:1;" onclick="submitForm('signin')">Sign in</button>
    <button class="btn-secondary" id="signup-btn" onclick="submitForm('signup')">Create account</button>
  </div>

  <div class="divider">or</div>

  <button class="btn-google" onclick="googleSignIn()">
    <svg width="16" height="16" viewBox="0 0 48 48">
      <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3c-1.6 4.6-6 8-11.3 8-6.6 0-12-5.4-12-12s5.4-12 12-12c3 0 5.8 1.1 7.9 2.9l5.7-5.7C34.5 5.9 29.5 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.3-.4-3.5z"/>
      <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.6 16 18.9 13 24 13c3 0 5.8 1.1 7.9 2.9l5.7-5.7C34.5 5.9 29.5 4 24 4c-7.6 0-14.1 4.3-17.3 10.6z"/>
      <path fill="#4CAF50" d="M24 44c5.4 0 10.3-1.8 14-4.9l-6.5-5.5c-2 1.4-4.6 2.3-7.5 2.3-5.3 0-9.7-3.4-11.3-8.1l-6.6 5.1C9.8 39.6 16.3 44 24 44z"/>
      <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.3-2.2 4.3-4 5.7l6.5 5.5C40.8 36.7 44 31 44 24c0-1.3-.1-2.3-.4-3.5z"/>
    </svg>
    Continue with Google
  </button>

  <script src="https://www.gstatic.com/firebasejs/10.13.0/firebase-app-compat.js"></script>
  <script src="https://www.gstatic.com/firebasejs/10.13.0/firebase-auth-compat.js"></script>
  <script>
    const params = new URLSearchParams(window.location.search);
    if (params.get("theme") === "dark") {
      document.body.classList.add("theme-dark");
    }

    const firebaseConfig = __FIREBASE_CONFIG__;
    firebase.initializeApp(firebaseConfig);
    const auth = firebase.auth();

    function setStatus(msg, ok) {
      const el = document.getElementById("status");
      el.textContent = msg || "";
      el.className = ok ? "ok" : "";
    }

    function setBusy(busy) {
      document.getElementById("signin-btn").disabled = busy;
      document.getElementById("signup-btn").disabled = busy;
    }

    function friendlyError(err) {
      const map = {
        "auth/invalid-email": "Invalid email address.",
        "auth/user-not-found": "No account found with this email. Try 'Create account' first.",
        "auth/wrong-password": "Incorrect password.",
        "auth/invalid-credential": "Incorrect email or password.",
        "auth/email-already-in-use": "This email is already registered. Use 'Sign in' instead.",
        "auth/weak-password": "Password must be at least 6 characters.",
        "auth/popup-closed-by-user": "Window was closed.",
        "auth/popup-blocked": "The browser blocked the pop-up. Please allow pop-ups and try again.",
        "auth/unauthorized-domain": "This domain is not authorized in Firebase."
      };
      return map[err.code] || "An error occurred (" + (err.code || "unknown error") + ").";
    }

    function clearContinueLink() {
      const old = document.getElementById("continue-link");
      if (old) old.remove();
    }

    function afterAuth(user, provider) {
      user.getIdToken().then(function (token) {
        clearContinueLink();

        if (window.opener) {
          window.opener.postMessage({
            type: "firebase-auth-result",
            ok: true,
            token: token,
            email: user.email || "",
            name: user.displayName || "",
            provider: provider
          }, "*");
          setStatus("Sign in successful, closing window…", true);
          setTimeout(function () { window.close(); }, 400);
        } else {
          setStatus("Sign in successful! Return to the main tab and refresh the page.", true);
        }
      });
    }

    function reportFailure(message) {
      setBusy(false);
      setStatus(message);
      clearContinueLink();
      if (window.opener) {
        window.opener.postMessage({ type: "firebase-auth-result", ok: false, error: message }, "*");
      }
    }

    function submitForm(requestedMode) {
      const email = document.getElementById("email").value.trim();
      const password = document.getElementById("password").value;
      if (!email || !password) {
        setStatus("Email and password are required.");
        return;
      }
      setBusy(true);
      setStatus(requestedMode === "signin" ? "Signing in…" : "Creating account…", true);
      const action = requestedMode === "signin"
        ? auth.signInWithEmailAndPassword(email, password)
        : auth.createUserWithEmailAndPassword(email, password);
      action
        .then(function (cred) { afterAuth(cred.user, "password"); })
        .catch(function (err) { reportFailure(friendlyError(err)); });
    }

    function googleSignIn() {
      const provider = new firebase.auth.GoogleAuthProvider();
      setBusy(true);
      setStatus("Opening Google window…", true);
      auth.signInWithPopup(provider)
        .then(function (cred) { afterAuth(cred.user, "google"); })
        .catch(function (err) { reportFailure(friendlyError(err)); });
    }

    function forgotPassword() {
      const email = document.getElementById("email").value.trim();
      if (!email) {
        setStatus("Enter your email first, then click.");
        return;
      }
      auth.sendPasswordResetEmail(email)
        .then(function () { setStatus("Password reset email sent.", true); })
        .catch(function (err) { setStatus(friendlyError(err)); });
    }

    document.getElementById("password").addEventListener("keydown", function (e) {
      if (e.key === "Enter") submitForm("signin");
    });
  </script>
</body>
</html>
"""


_server_state: Dict[str, Any] = {}


class _AuthPageHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = _server_state.get("html", "").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@st.cache_resource(show_spinner=False)
def _start_auth_server(config_json: str) -> int:
    html = _AUTH_PAGE_TEMPLATE.replace("__FIREBASE_CONFIG__", config_json)
    _server_state["html"] = html

    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _AuthPageHandler)
    httpd.daemon_threads = True
    port = httpd.server_address[1]

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    return port


def render_firebase_login(
    config: Dict[str, str],
    height: int = 90,
    theme_mode: str = "light",
) -> None:
    port = _start_auth_server(json.dumps(config))
    popup_url = f"http://localhost:{port}/?theme={theme_mode}"

    widget_html = f"""
    <div style="font-family: Inter, system-ui, sans-serif;">
      <style>
        #fb-launch-btn {{
          width: 100%; padding: 0.85rem 1rem; border-radius: 12px; border: none;
          background: #5962D7; color: #fff; font-weight: 650; font-size: 0.95rem;
          cursor: pointer;
        }}
        #fb-launch-btn:hover {{ background: #454CB8; }}
        #fb-launch-status {{
          margin-top: 0.55rem; font-size: 0.82rem; color: #D64545; min-height: 1.1rem;
        }}
        #fb-launch-status.ok {{ color: #20B486; }}
        #fb-continue-link {{
          display: inline-block; margin-top: 0.4rem; font-size: 0.86rem;
          font-weight: 700; color: #5962D7; text-decoration: none;
        }}
        #fb-continue-link:hover {{ text-decoration: underline; }}
      </style>
      <button id="fb-launch-btn" onclick="fbOpenPopup()">Sign in / Sign up</button>
      <div id="fb-launch-status"></div>
    </div>
    <script>
      function fbOpenPopup() {{
        const w = 440, h = 640;
        const left = window.screenX + (window.outerWidth - w) / 2;
        const top = window.screenY + (window.outerHeight - h) / 2;
        const popup = window.open(
          "{popup_url}",
          "firebaseAuthPopup",
          `width=${{w}},height=${{h}},left=${{left}},top=${{top}}`
        );
        if (!popup) {{
          document.getElementById("fb-launch-status").textContent =
            "The browser blocked the pop-up. Please allow pop-ups and try again.";
        }}
      }}

      window.addEventListener("message", function (event) {{
        const data = event.data || {{}};
        if (data.type !== "firebase-auth-result") return;

        const statusEl = document.getElementById("fb-launch-status");
        const oldLink = document.getElementById("fb-continue-link");
        if (oldLink) oldLink.remove();

        if (!data.ok) {{
          statusEl.textContent = data.error || "Sign in failed.";
          statusEl.className = "";
          return;
        }}

        statusEl.textContent = "Sign in successful, redirecting…";
        statusEl.className = "ok";

        const qs = new URLSearchParams();
        qs.set("fb_token", data.token);
        qs.set("fb_email", data.email || "");
        qs.set("fb_name", data.name || "");
        qs.set("fb_provider", data.provider || "password");
        const search = qs.toString();
        const targetUrl = "?" + search;

        try {{
          const topDoc = window.top.document;
          const script = topDoc.createElement("script");
          script.textContent = "window.location.search = " + JSON.stringify(search) + ";";
          (topDoc.body || topDoc.documentElement).appendChild(script);
          script.remove();
          return;
        }} catch (err) {{
          console.error("Top-document script injection failed:", err);
        }}

        statusEl.textContent = "Sign in successful!";
        const link = document.createElement("a");
        link.id = "fb-continue-link";
        link.href = targetUrl;
        link.target = "_top";
        link.textContent = "Click to continue →";
        statusEl.insertAdjacentElement("afterend", link);
      }});
    </script>
    """

    components.html(widget_html, height=height, scrolling=False)