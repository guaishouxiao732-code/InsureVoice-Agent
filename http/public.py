#!/usr/bin/env python3
"""Start the assistant website and a password-protected Cloudflare Quick Tunnel."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


BASE_DIR = Path(__file__).resolve().parent
CLOUDFLARED = BASE_DIR / ".tools" / "cloudflared"
PASSWORD_FILE = BASE_DIR / ".web.env"


def saved_password() -> str:
    """Load the local fixed password without embedding it in source code."""
    explicit = os.environ.get("WEB_PASSWORD", "").strip()
    if explicit:
        return explicit
    if PASSWORD_FILE.is_file():
        for raw_line in PASSWORD_FILE.read_text("utf-8").splitlines():
            key, separator, value = raw_line.partition("=")
            if separator and key.strip() == "WEB_PASSWORD" and value.strip():
                return value.strip()
    return secrets.token_urlsafe(18)


def find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_web(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("网页服务启动失败，请查看上方日志")
        try:
            with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                if json.loads(response.read()).get("status") == "ok":
                    return
        except (OSError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(.25)
    raise TimeoutError("网页服务启动超时")


def wait_for_public_url(metrics_port: int, process: subprocess.Popen[bytes]) -> str:
    metrics_url = f"http://127.0.0.1:{metrics_port}/metrics"
    pattern = re.compile(r'userHostname="(https://[^" ]+\.trycloudflare\.com)"')
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Cloudflare Tunnel 已退出")
        try:
            with urlopen(metrics_url, timeout=1) as response:
                match = pattern.search(response.read().decode("utf-8"))
            if match:
                return match.group(1)
        except (OSError, TimeoutError):
            pass
        time.sleep(.5)
    raise TimeoutError("未能读取临时公网地址")


def stop(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> None:
    if not CLOUDFLARED.is_file():
        raise SystemExit("未找到 http/.tools/cloudflared")
    port = int(os.environ.get("PORT", "8000"))
    metrics_port = find_free_port()
    password = saved_password()
    env = os.environ.copy()
    env.update({"HOST": "127.0.0.1", "PORT": str(port), "WEB_PASSWORD": password})
    web: subprocess.Popen[bytes] | None = None
    tunnel: subprocess.Popen[bytes] | None = None
    try:
        web = subprocess.Popen([sys.executable, str(BASE_DIR / "app.py")], cwd=BASE_DIR, env=env)
        wait_for_web(port, web)
        tunnel = subprocess.Popen([
            str(CLOUDFLARED), "tunnel", "--no-autoupdate",
            "--metrics", f"127.0.0.1:{metrics_port}",
            "--url", f"http://127.0.0.1:{port}",
        ])
        public_url = wait_for_public_url(metrics_port, tunnel)
        border = "=" * 72
        print(f"\n{border}", flush=True)
        print(f"上海用户访问地址：{public_url}", flush=True)
        print(f"临时访问密码：{password}", flush=True)
        print("合肥主机需保持开机；按 Ctrl+C 停止公网访问。", flush=True)
        print(f"{border}\n", flush=True)
        tunnel.wait()
    except KeyboardInterrupt:
        print("\n正在停止网页和公网隧道……", flush=True)
    finally:
        stop(tunnel)
        stop(web)


if __name__ == "__main__":
    main()
