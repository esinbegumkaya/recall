"""
firebase_auth.py
=================

Firebase Authentication integration for Streamlit, via a popup window.

Why a popup instead of an embedded iframe
------------------------------------------
Streamlit's `st.components.v1.html` renders content in an iframe using
`srcdoc`, which browsers treat as having protocol "about:" rather than
"http:"/"https:". Firebase Auth explicitly rejects that environment with:

    auth/operation-not-supported-in-this-environment
    "location.protocol" must be http, https or chrome-extension and
    web storage must be enabled.

To work around this, the actual Firebase sign-in form runs on a tiny local
HTTP server (bound to 127.0.0.1) and opens in a real popup window — a
genuine top-level browsing context with a proper http: origin, where
Firebase Auth is fully supported. When sign-in succeeds, the popup posts
the result back to the Streamlit tab via `window.postMessage` and closes
itself; the Streamlit tab then redirects itself with the ID token attached
as a query parameter, exactly as before.

Getting the result back to the Streamlit tab
----------------------------------------------
This is the fiddly part, because two different browser restrictions stack:

1. The popup's real `window.opener` is Streamlit's `components.html`
   iframe (that's what called `window.open()`), NOT the browser tab
   itself. Browsers only let a window navigate a frame it is directly the
   opener/parent of, or is same-origin with — so the popup trying to reach
   "the tab" via `window.opener.top` is refused outright:
   "Unsafe attempt to initiate navigation ... neither same-origin ... nor
   is it the target's parent or opener." No JS trick gets around this from
   the popup's side; it can only safely postMessage its direct opener (the
   iframe) and close itself.

2. That iframe is sandboxed without "allow-top-navigation", so
   `window.top.location = ...` from inside it also always throws — sandbox
   restrictions apply regardless of user activation, click-triggered or
   not.

Both restrictions are specifically about *navigation*, though — not about
ordinary same-origin scripting. `window.top` itself is directly, same-
origin accessible from inside the iframe (no cross-origin error reading or
touching it — only setting its `.location` is specially blocked by the
sandbox). So instead of navigating anything, the iframe injects a real
`<script>` element directly into `window.top.document`, via
`createElement` + `appendChild` (elements added this way DO execute,
unlike HTML inserted through `innerHTML`). That injected script then runs
in the *top page's own* context, completely unsandboxed, so its own
`window.location` assignment is just an ordinary same-window navigation —
no restriction applies to it at all.

Security note
-------------
`decode_id_token()` only decodes the JWT payload — it does NOT verify the
cryptographic signature. Fine for a local/prototype tool where the token
never leaves your own machine's browser. For a public deployment, verify
server-side instead with the `firebase-admin` SDK:

    import firebase_admin
    from firebase_admin import auth as fb_admin_auth, credentials

    cred = credentials.Certificate("service-account.json")
    firebase_admin.initialize_app(cred)
    decoded = fb_admin_auth.verify_id_token(id_token)
"""

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
    """Read the Firebase web config from .streamlit/secrets.toml."""
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
    """Decode (without verifying) the payload of a Firebase ID token."""
    try:
        payload_segment = token.split(".")[1]
        padded = payload_segment + "=" * (-len(payload_segment) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("utf-8"))
        return json.loads(raw)
    except Exception:
        return {}


def consume_auth_redirect() -> Optional[Dict[str, Any]]:
    """Check the URL query params for a Firebase redirect and consume it."""
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


# =========================================================
# Standalone popup page (served over real http:// so Firebase Auth is
# fully supported) — built once, then served by a tiny local HTTP server.
# =========================================================

_AUTH_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8" />
<title>Recall — Giriş</title>
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
  <h1>Recall'a giriş yap</h1>
  <p class="sub">Kimlik doğrulama Firebase tarafından yapılır.</p>

  <div id="status"></div>

  <input id="email" type="email" placeholder="name@example.com" autocomplete="email" />
  <input id="password" type="password" placeholder="Şifre" autocomplete="current-password" />
  <a class="forgot" onclick="forgotPassword()">Şifremi unuttum</a>

  <div style="display:flex; gap:0.5rem; margin-bottom: 0.6rem;">
    <button class="btn-primary" id="signin-btn" style="margin-bottom:0; width:auto; flex:1;" onclick="submitForm('signin')">Giriş yap</button>
    <button class="btn-secondary" id="signup-btn" onclick="submitForm('signup')">Hesap oluştur</button>
  </div>

  <div class="divider">veya</div>

  <button class="btn-google" onclick="googleSignIn()">
    <svg width="16" height="16" viewBox="0 0 48 48">
      <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3c-1.6 4.6-6 8-11.3 8-6.6 0-12-5.4-12-12s5.4-12 12-12c3 0 5.8 1.1 7.9 2.9l5.7-5.7C34.5 5.9 29.5 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.3-.4-3.5z"/>
      <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.6 16 18.9 13 24 13c3 0 5.8 1.1 7.9 2.9l5.7-5.7C34.5 5.9 29.5 4 24 4c-7.6 0-14.1 4.3-17.3 10.6z"/>
      <path fill="#4CAF50" d="M24 44c5.4 0 10.3-1.8 14-4.9l-6.5-5.5c-2 1.4-4.6 2.3-7.5 2.3-5.3 0-9.7-3.4-11.3-8.1l-6.6 5.1C9.8 39.6 16.3 44 24 44z"/>
      <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.3-2.2 4.3-4 5.7l6.5 5.5C40.8 36.7 44 31 44 24c0-1.3-.1-2.3-.4-3.5z"/>
    </svg>
    Google ile devam et
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
        "auth/invalid-email": "Geçersiz e-posta adresi.",
        "auth/user-not-found": "Bu e-posta ile kayıtlı bir hesap yok. Önce 'Hesap oluştur'a bas.",
        "auth/wrong-password": "Şifre hatalı.",
        "auth/invalid-credential": "E-posta veya şifre hatalı.",
        "auth/email-already-in-use": "Bu e-posta zaten kayıtlı. 'Giriş yap'ı kullan.",
        "auth/weak-password": "Şifre en az 6 karakter olmalı.",
        "auth/popup-closed-by-user": "Pencere kapatıldı.",
        "auth/popup-blocked": "Tarayıcı pop-up'ı engelledi. Pop-up izni verip tekrar deneyin.",
        "auth/unauthorized-domain": "Bu adres Firebase'de yetkili domain olarak tanımlı değil."
      };
      return map[err.code] || "Bir hata oluştu (" + (err.code || "bilinmeyen hata") + ").";
    }

    // Removes any leftover "Devam et" link from a previous attempt before
    // rendering a new status, so retries don't stack duplicate links.
    function clearContinueLink() {
      const old = document.getElementById("continue-link");
      if (old) old.remove();
    }

    function afterAuth(user, provider) {
      user.getIdToken().then(function (token) {
        clearContinueLink();

        // We deliberately do NOT try to navigate window.opener.top (or
        // opener itself) from here. Browsers only let a window navigate a
        // frame it is directly the opener/parent of, or is same-origin
        // with. This popup's real "opener" is the Streamlit tab's sandboxed
        // component iframe — NOT the tab itself — so any attempt to reach
        // "the tab" (window.opener.top) is refused outright with:
        //   "Unsafe attempt to initiate navigation ... neither same-origin
        //   ... nor is it the target's parent or opener."
        // That's a hard browser security rule; no JS trick gets around it
        // from this side. Instead we just hand the token to our direct
        // opener via postMessage and let IT figure out how to get the
        // Streamlit tab to the right URL (see render_firebase_login's
        // message listener, which relays this to a bridge function running
        // in the tab's own unsandboxed top-level page).
        if (window.opener) {
          window.opener.postMessage({
            type: "firebase-auth-result",
            ok: true,
            token: token,
            email: user.email || "",
            name: user.displayName || "",
            provider: provider
          }, "*");
          setStatus("Giriş başarılı, pencere kapatılıyor…", true);
          setTimeout(function () { window.close(); }, 400);
        } else {
          // No opener at all (popup launched some other way) — nothing to
          // hand off to. Last resort: let the person copy the token by hand.
          setStatus("Giriş başarılı! Ana sekmeye dönüp sayfayı yenileyin.", true);
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
        setStatus("E-posta ve şifre gerekli.");
        return;
      }
      setBusy(true);
      setStatus(requestedMode === "signin" ? "Giriş yapılıyor…" : "Hesap oluşturuluyor…", true);
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
      setStatus("Google penceresi açılıyor…", true);
      auth.signInWithPopup(provider)
        .then(function (cred) { afterAuth(cred.user, "google"); })
        .catch(function (err) { reportFailure(friendlyError(err)); });
    }

    function forgotPassword() {
      const email = document.getElementById("email").value.trim();
      if (!email) {
        setStatus("Önce e-posta adresini yaz, sonra tıkla.");
        return;
      }
      auth.sendPasswordResetEmail(email)
        .then(function () { setStatus("Şifre sıfırlama e-postası gönderildi.", true); })
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
    def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
        body = _server_state.get("html", "").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # keep Streamlit's terminal output clean


@st.cache_resource(show_spinner=False)
def _start_auth_server(config_json: str) -> int:
    """Start (once per process) a local HTTP server serving the auth popup.

    Bound to 127.0.0.1 only — never reachable from outside this machine.
    """
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
    """Render a "Sign in" launcher that opens Firebase Auth in a popup.

    The popup runs on a local http:// server (see module docstring for why
    that's necessary), and reports back to this page via postMessage, which
    is relayed to the browser tab by injecting a real <script> element
    directly into the top document (see the message listener below for why).
    """
    port = _start_auth_server(json.dumps(config))
    # "localhost" (not "127.0.0.1") because Firebase's default Authorized
    # domains list includes "localhost" out of the box — using the raw IP
    # address here is exactly what was causing Google sign-in to fail with
    # auth/unauthorized-domain.
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
      <button id="fb-launch-btn" onclick="fbOpenPopup()">Giriş yap / Kayıt ol</button>
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
            "Tarayıcı pop-up'ı engelledi. Pop-up izni verip tekrar deneyin.";
        }}
      }}

      window.addEventListener("message", function (event) {{
        const data = event.data || {{}};
        if (data.type !== "firebase-auth-result") return;

        const statusEl = document.getElementById("fb-launch-status");
        const oldLink = document.getElementById("fb-continue-link");
        if (oldLink) oldLink.remove();

        if (!data.ok) {{
          statusEl.textContent = data.error || "Giriş başarısız oldu.";
          statusEl.className = "";
          return;
        }}

        statusEl.textContent = "Giriş başarılı, yönlendiriliyor…";
        statusEl.className = "ok";

        const qs = new URLSearchParams();
        qs.set("fb_token", data.token);
        qs.set("fb_email", data.email || "");
        qs.set("fb_name", data.name || "");
        qs.set("fb_provider", data.provider || "password");
        const search = qs.toString();
        const targetUrl = "?" + search;

        // This iframe (Streamlit's components.html) is sandboxed WITHOUT
        // "allow-top-navigation", so window.top.location = ... always
        // throws here — confirmed, that's the exact error we hit before.
        // BUT: window.top itself is directly, same-origin accessible (no
        // cross-origin error when reading/calling into it) — the sandbox
        // specifically blocks *navigation*, not ordinary same-origin DOM
        // access. So instead of setting window.top.location ourselves
        // (blocked) or calling a pre-defined function there (which turned
        // out not to survive Streamlit's HTML sanitizer when injected via
        // st.markdown), we inject a real <script> element directly into the
        // top document via createElement + appendChild. Elements added this
        // way DO execute (unlike innerHTML-inserted <script> tags), and the
        // injected script runs in the top page's own, completely
        // unsandboxed context — so its own window.location assignment is
        // just an ordinary same-window navigation with no restriction at
        // all.
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

        // Fallback, only reached if the injection above is somehow blocked
        // (e.g. allow-same-origin missing after all). A genuine click still
        // carries its own fresh user activation, so a plain target="_top"
        // link has a decent chance of working even where scripted attempts
        // didn't.
        statusEl.textContent = "Giriş başarılı!";
        const link = document.createElement("a");
        link.id = "fb-continue-link";
        link.href = targetUrl;
        link.target = "_top";
        link.textContent = "Devam etmek için tıkla →";
        statusEl.insertAdjacentElement("afterend", link);
      }});
    </script>
    """

    components.html(widget_html, height=height, scrolling=False)