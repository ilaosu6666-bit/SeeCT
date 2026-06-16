"""
智影溯源 桌面客户端
"""

import os, sys, socket, threading, time


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
    # ----- 确定工作目录 -----
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    port = find_port()
    url = f"http://127.0.0.1:{port}"

    # ----- 后台线程启动 Streamlit -----
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
    os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
    os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
    # 关键：阻止 Streamlit 自动打开浏览器
    os.environ["BROWSER"] = "echo"

    def run_streamlit():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(
            os.path.join(base_dir, "app.py"),
            is_hello=False,
            args=[],
            flag_options={},
        )

    t = threading.Thread(target=run_streamlit, daemon=True)
    t.start()

    # ----- 等就绪 -----
    import urllib.request
    for _ in range(40):
        try:
            urllib.request.urlopen(url, timeout=0.5)
            break
        except Exception:
            time.sleep(0.5)
    else:
        import tkinter.messagebox as mb
        mb.showerror("启动失败", "服务未能启动")
        return

    # ----- 单一窗口 -----
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
            while t.is_alive():
                t.join(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
