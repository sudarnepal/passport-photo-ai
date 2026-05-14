import os
import sys
import socket
import threading
import webbrowser
import time

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def open_browser(port):
    time.sleep(3)
    webbrowser.open(f"http://localhost:{port}")

def get_app_path():
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "app.py")

if __name__ == "__main__":
    port = find_free_port()
    app_path = get_app_path()
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    from streamlit.web import cli as stcli
    sys.argv = [
        "streamlit", "run", app_path,
        "--server.port", str(port),
        "--server.headless", "true",
        "--global.developmentMode", "false",
        "--browser.gatherUsageStats", "false",
    ]
    sys.exit(stcli.main())
