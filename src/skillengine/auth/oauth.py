"""
OAuth 2.0 + PKCE flow for Claude.ai setup-token authentication.

Implements the same flow as ``claude setup-token`` from Claude Code,
generating a long-lived (1-year) OAuth access token for inference.

Uses only Python stdlib — no external HTTP libraries required.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import subprocess
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

# ---------------------------------------------------------------------------
# Production OAuth constants (must match Claude Code exactly)
# ---------------------------------------------------------------------------

CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://claude.com/cai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
MANUAL_REDIRECT_URL = "https://platform.claude.com/oauth/code/callback"
SUCCESS_URL = "https://platform.claude.com/oauth/code/success?app=claude-code"
SCOPE = "user:inference"
EXPIRES_IN = 365 * 24 * 60 * 60  # 1 year in seconds


# ---------------------------------------------------------------------------
# PKCE helpers (RFC 7636)
# ---------------------------------------------------------------------------

def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_code_verifier() -> str:
    return _base64url(os.urandom(32))


def generate_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return _base64url(digest)


def generate_state() -> str:
    return _base64url(os.urandom(32))


# ---------------------------------------------------------------------------
# Localhost callback server
# ---------------------------------------------------------------------------

class _CallbackHandler(BaseHTTPRequestHandler):
    server: _CallbackServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/callback":
            self.send_error(404)
            return

        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]

        if not code:
            self.send_error(400, "Missing authorization code")
            return

        if state != self.server.expected_state:
            self.send_error(400, "Invalid state parameter")
            return

        self.server.auth_code = code

        self.send_response(302)
        self.send_header("Location", SUCCESS_URL)
        self.end_headers()

        self.server.code_received.set()

    def log_message(self, format: str, *args: object) -> None:
        pass  # suppress request logs


class _CallbackServer(HTTPServer):
    address_family = socket.AF_INET6

    def __init__(self) -> None:
        super().__init__(("::1", 0), _CallbackHandler)
        self.expected_state = ""
        self.auth_code: str | None = None
        self.code_received = threading.Event()


# ---------------------------------------------------------------------------
# Auth URL builder
# ---------------------------------------------------------------------------

def build_auth_url(
    code_challenge: str,
    state: str,
    port: int = 0,
    manual: bool = False,
) -> str:
    redirect_uri = (
        MANUAL_REDIRECT_URL if manual else f"http://localhost:{port}/callback"
    )
    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


# ---------------------------------------------------------------------------
# Token exchange
# ---------------------------------------------------------------------------

def exchange_code_for_token(
    code: str,
    code_verifier: str,
    state: str,
    port: int = 0,
    manual: bool = False,
) -> dict:
    redirect_uri = (
        MANUAL_REDIRECT_URL if manual else f"http://localhost:{port}/callback"
    )
    body = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": CLIENT_ID,
        "code_verifier": code_verifier,
        "state": state,
        "expires_in": EXPIRES_IN,
    }
    result = subprocess.run(
        [
            "curl", "-s", "-X", "POST", TOKEN_URL,
            "-H", "Content-Type: application/json",
            "-d", json.dumps(body),
            "--max-time", "15",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Token exchange request failed: {result.stderr}")
    resp = json.loads(result.stdout)
    if "error" in resp:
        raise RuntimeError(f"Token exchange failed: {resp}")
    return resp


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run_setup_token_flow(manual: bool = False) -> str:
    """Run the OAuth setup-token flow and return the access token.

    Args:
        manual: If True, use the manual copy-paste flow instead of browser redirect.
    """
    verifier = generate_code_verifier()
    challenge = generate_code_challenge(verifier)
    state = generate_state()

    if manual:
        url = build_auth_url(challenge, state, manual=True)
        print("\nOpen this URL in your browser to authorize:\n")
        print(f"  {url}\n")
        print("After authorizing, paste the authorization code below:\n")
        code = input("Authorization code: ").strip()
        token_data = exchange_code_for_token(code, verifier, state, manual=True)
    else:
        server = _CallbackServer()
        server.expected_state = state
        port = server.server_address[1]

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        url = build_auth_url(challenge, state, port=port)
        print("\nOpening browser for authorization...\n")
        print(f"If the browser doesn't open, visit this URL:\n\n  {url}\n")
        webbrowser.open(url)

        print("Waiting for authorization...")
        if not server.code_received.wait(timeout=120):
            server.shutdown()
            raise TimeoutError("Authorization timed out after 120 seconds.")

        code = server.auth_code
        server.shutdown()

        if not code:
            raise RuntimeError("No authorization code received.")

        token_data = exchange_code_for_token(code, verifier, state, port=port)

    access_token = token_data["access_token"]

    print("\n\033[32m✓ Long-lived authentication token created successfully!\033[0m\n")
    print(f"Your OAuth token (valid for 1 year):\n\n  \033[33m{access_token}\033[0m\n")
    print("Store this token securely. You won't be able to see it again.\n")
    print("Usage:\n")
    print(f"  export CLAUDE_CODE_OAUTH_TOKEN={access_token}\n")

    return access_token
