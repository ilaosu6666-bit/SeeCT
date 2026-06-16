"""
智影溯源 — 桌面启动器（PyInstaller 打包入口）

打包命令: pyinstaller --onefile --windowed --name 智影溯源 launcher.py
"""

import subprocess
import sys
import threading
import time
import socket
import os
import webbrowser


def find_port(start=8501):
    for p in range(start, 8599):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p)); s.close(); return p
        except OSError:
            s.close()
    return 8501


def main():
    port = find_port()
    url = f"http://127.0.0.1:{port}"

    base_dir = os.path.dirname(os.path.abspath(__file__))

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(port),
         "--server.headless", "true",
         "--server.enableCORS", "false",
         "--server.enableXsrfProtection", "false",
         "--browser.gatherUsageStats", "false"],
        cwd=base_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(4)

    try:
        import webview
        webview.create_window(
            "智影溯源 - AI肺结节教学平台",
            url, width=1280, height=800,
            min_size=(900, 600), text_select=True,
        )
        webview.start()
    except ImportError:
        webbrowser.open(url)
        try:
            proc.wait()
        except KeyboardInterrupt:
            pass
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
