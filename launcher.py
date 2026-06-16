"""
智影溯源 桌面客户端
"""

import os, sys, time, socket, subprocess, tempfile

# 强制无缓冲，避免 PyInstaller windowed 模式下的管道问题
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')


def resource_path(p):
    d = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(d, p)


def find_port(start=8501):
    for p in range(start, 8600):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.1)
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            continue
    return 8501


def main():
    # ----- 子进程模式 -----
    if "--child" in sys.argv:
        port = sys.argv[sys.argv.index("--child") + 1]
        os.chdir(resource_path("."))
        os.environ["BROWSER"] = "echo"
        import streamlit.web.cli as stcli
        sys.argv = [
            "streamlit", "run", resource_path("app.py"),
            "--server.port", port,
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none",
            "--server.runOnSave", "false",
            "--server.address", "127.0.0.1",
        ]
        try:
            stcli.main()
        except SystemExit:
            pass
        os._exit(0)
        return

    # ----- 父进程模式 -----
    lock = os.path.join(tempfile.gettempdir(), "zhiyingsuyuan.lock")
    if os.path.exists(lock):
        import tkinter.messagebox as mb
        mb.showinfo("智影溯源", "程序已在运行中。")
        return
    with open(lock, "w") as f:
        f.write("1")

    port = find_port()
    url = f"http://127.0.0.1:{port}"

    # 启动子进程—同一个 exe，带 --child 标记
    child = subprocess.Popen(
        [sys.executable, "--child", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.PIPE, close_fds=True,
    )

    # 等待 Streamlit 就绪
    import urllib.request
    ok = False
    for _ in range(60):
        try:
            urllib.request.urlopen(url, timeout=0.5)
            ok = True
            break
        except Exception:
            time.sleep(0.5)
    if not ok:
        import tkinter.messagebox as mb
        mb.showerror("启动失败", "Streamlit 服务未能启动")
        child.terminate()
        os.remove(lock)
        return

    # 弹窗
    try:
        import webview
        webview.create_window(
            "智影溯源 — AI肺结节教学平台",
            url, width=1280, height=800,
            min_size=(900, 600), text_select=True,
        )
        webview.start()
    except ImportError:
        import webbrowser
        webbrowser.open(url)
        try:
            child.wait()
        except KeyboardInterrupt:
            pass

    child.terminate()
    try:
        child.wait(timeout=5)
    except Exception:
        child.kill()
    try:
        os.remove(lock)
    except Exception:
        pass


if __name__ == "__main__":
    main()
