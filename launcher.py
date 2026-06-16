"""
智影溯源 桌面客户端

打包: pyinstaller --onedir --windowed --name 智影溯源 智影溯源.spec
"""

import os, sys, socket, threading, time, tempfile


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


def main():
    # ---- 防重复启动 ----
    lock = os.path.join(tempfile.gettempdir(), "zhiying.lock")
    if os.path.exists(lock):
        # 锁文件存在 → 可能已有实例在跑，尝试连接现有服务
        port = _read_port(lock)
        if port:
            try:
                import urllib.request
                urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=0.5)
                # 服务正常运行，只打开窗口
                _open_window(f"http://127.0.0.1:{port}")
                return
            except Exception:
                pass
        # 锁残留，清理
        try:
            os.remove(lock)
        except Exception:
            pass

    # ---- 正常启动 ----
    try:
        _start()
    finally:
        try:
            os.remove(lock)
        except Exception:
            pass


def _read_port(lock_path):
    try:
        with open(lock_path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _start():
    base_dir = (sys._MEIPASS if getattr(sys, 'frozen', False)
                else os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    port = find_port()
    url = f"http://127.0.0.1:{port}"

    # 写锁（含端口号，供后续检测连接）
    with open(os.path.join(tempfile.gettempdir(), "zhiying.lock"), "w") as f:
        f.write(str(port))

    # 环境变量
    os.environ["BROWSER"] = "echo"
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
    os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
    os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

    # 后台线程跑 Streamlit
    def _serve():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(
            os.path.join(base_dir, "app.py"),
            is_hello=False,
            args=[],
            flag_options={},
        )

    threading.Thread(target=_serve, daemon=True).start()

    # 等就绪
    import urllib.request
    for _ in range(40):
        time.sleep(0.5)
        try:
            urllib.request.urlopen(url, timeout=0.3)
            break
        except Exception:
            pass

    _open_window(url)


def _open_window(url):
    try:
        import webview
        webview.create_window(
            "智影溯源 - AI肺结节教学平台",
            url, width=1280, height=800,
            min_size=(900, 600), text_select=True,
        )
        webview.start()
    except ImportError:
        import webbrowser
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
