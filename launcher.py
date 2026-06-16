"""
智影溯源 桌面客户端
打包: pyinstaller 智影溯源.spec
"""

import os, sys, socket, time, tempfile, subprocess


def find_port(start=8501):
    for p in range(start, 8599):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            s.close()
    return 8501


def main():
    # ----- 防重复启动 -----
    lock_path = os.path.join(tempfile.gettempdir(), "zhiyingsuyuan.lock")
    if os.path.exists(lock_path):
        import tkinter.messagebox as mb
        mb.showinfo("智影溯源", "程序已在运行中。")
        return
    with open(lock_path, "w") as f:
        f.write(str(os.getpid()))

    try:
        _run(lock_path)
    finally:
        try:
            os.remove(lock_path)
        except Exception:
            pass


def _run(lock_path):
    # ----- 工作目录 -----
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    port = find_port()
    url = f"http://127.0.0.1:{port}"

    # 后台子进程启动 Streamlit
    env = os.environ.copy()
    env["BROWSER"] = "echo"  # 阻止 Streamlit 打开浏览器

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(port),
         "--server.headless", "true",
         "--server.address", "127.0.0.1",
         "--server.enableCORS", "false",
         "--server.enableXsrfProtection", "false",
         "--browser.gatherUsageStats", "false",
         "--server.runOnSave", "false",
         "--server.fileWatcherType", "none",
         "--server.maxUploadSize", "50"],
        cwd=base_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 等待服务就绪
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen(url + "/healthz", timeout=0.5)
            break
        except Exception:
            try:
                urllib.request.urlopen(url, timeout=0.5)
                break
            except Exception:
                time.sleep(0.8)
    else:
        import tkinter.messagebox as mb
        mb.showerror("启动失败", "服务未能启动，请先运行 install.bat 安装依赖。")
        proc.terminate()
        return

    # 打开桌面窗口
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
    except Exception:
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
