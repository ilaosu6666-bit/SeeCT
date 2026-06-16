"""
智影溯源 桌面客户端
双击即用，原生窗口 + Streamlit 服务
打包: pyinstaller --onedir --windowed --name 智影溯源 --add-data "app.py;." --add-data "game.py;." --add-data "part1.py;." --add-data "model.py;." --add-data "model_config.json;." --add-data "requirements.txt;." --add-data "cases;cases" --add-data "data;data" --add-data "loss_fig;loss_fig" --add-data "model_parameter;model_parameter" --add-data "cases_manifest.json;." launcher.py
"""

import subprocess
import sys
import time
import socket
import os
import json
import urllib.request


def find_port(start=8501):
    for p in range(start, 8599):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.1)
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            s.close()
    return 8501


def wait_for_server(url, timeout=30):
    """轮询等待 Streamlit 就绪"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=0.5)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    # 确定工作目录（兼容 PyInstaller _MEIPASS 和直接运行）
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    os.chdir(base_dir)

    port = find_port()
    url = f"http://127.0.0.1:{port}"

    # 启动 Streamlit（禁止自动打开浏览器）
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(port),
         "--server.headless", "true",
         "--server.enableCORS", "false",
         "--server.enableXsrfProtection", "false",
         "--browser.gatherUsageStats", "false",
         "--server.address", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=base_dir,
    )

    # 等待服务就绪
    ready = wait_for_server(f"{url}/healthz", timeout=20)

    if not ready:
        # 可能 healthz 路径不对，试试根路径
        try:
            urllib.request.urlopen(url, timeout=2)
            ready = True
        except Exception:
            pass

    if not ready:
        import tkinter.messagebox as mb
        mb.showerror("启动失败", "Streamlit 服务未能启动，请检查是否安装了依赖。\n运行 install.bat 安装。")
        proc.terminate()
        return

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
        import webbrowser
        webbrowser.open(url)
        try:
            proc.wait()
        except KeyboardInterrupt:
            pass
    finally:
        proc.terminate()
        proc.wait(timeout=3)


if __name__ == "__main__":
    main()
