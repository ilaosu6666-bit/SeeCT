"""
智影溯源 桌面客户端
打包: pyinstaller 智影溯源.spec
"""

import os, sys, time, socket, subprocess, tempfile


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def find_port(start=8501):
    for p in range(start, 8600):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            continue
    return 8501


def run_streamlit_child():
    """子进程：只运行 Streamlit，不再启动新 exe"""
    port = sys.argv[sys.argv.index("--run-streamlit") + 1]
    os.chdir(sys._MEIPASS if getattr(sys, 'frozen', False)
             else os.path.dirname(os.path.abspath(__file__)))

    app_path = resource_path("app.py")
    import streamlit.web.cli as stcli
    sys.argv = [
        "streamlit", "run", app_path,
        "--server.port", port,
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
        "--server.fileWatcherType", "none",
    ]
    sys.exit(stcli.main())


def main():
    # 子进程入口
    if "--run-streamlit" in sys.argv:
        run_streamlit_child()
        return

    # ----- 防重复 -----
    lock = os.path.join(tempfile.gettempdir(), "zhiyingsuyuan.lock")
    if os.path.exists(lock):
        import tkinter.messagebox as mb
        mb.showinfo("智影溯源", "程序已在运行中。")
        return
    with open(lock, "w") as f:
        f.write(str(os.getpid()))

    try:
        _main(lock)
    finally:
        try:
            os.remove(lock)
        except Exception:
            pass


def _main(lock):
    port = find_port()
    url = f"http://127.0.0.1:{port}"

    # 子进程运行 Streamlit
    proc = subprocess.Popen(
        [sys.executable, "--run-streamlit", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 等待就绪
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen(url, timeout=0.5)
            break
        except Exception:
            time.sleep(0.8)
    else:
        import tkinter.messagebox as mb
        mb.showerror("启动失败", "服务未能启动")
        proc.terminate()
        return

    # 打开窗口
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
            proc.wait()
        except KeyboardInterrupt:
            pass
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
