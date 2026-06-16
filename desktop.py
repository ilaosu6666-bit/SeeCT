"""
智影溯源 — 桌面客户端启动器

双击运行即可以原生窗口打开完整应用，无需浏览器。
首次运行需确保依赖已安装: pip install -r requirements.txt
"""

import subprocess
import sys
import threading
import time
import socket


def find_free_port(start: int = 8501) -> int:
    """找一个可用的端口"""
    port = start
    while port < 8599:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            sock.close()
            return port
        except OSError:
            port += 1
            sock.close()
    return 8501


def main():
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    # 后台启动 Streamlit
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(port),
         "--server.headless", "true",
         "--server.enableCORS", "false",
         "--server.enableXsrfProtection", "false",
         "--browser.gatherUsageStats", "false",
         "--server.enableWebsocketCompression", "false"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 等待 Streamlit 就绪
    time.sleep(4)

    try:
        import webview
        webview.create_window(
            "智影溯源 — AI肺结节教学平台",
            url,
            width=1280,
            height=800,
            min_size=(900, 600),
            text_select=True,
        )
        webview.start()
    except ImportError:
        # 没有 pywebview → 浏览器打开
        import webbrowser
        webbrowser.open(url)
        print(f"服务运行中: {url}")
        print("按 Ctrl+C 退出")
        try:
            proc.wait()
        except KeyboardInterrupt:
            pass
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
