# scripts/browser.py
"""
Custom browser window using pywebview with WebView2 (Chromium) backend.
Provides modern JS support for Gradio on Windows 8.1+ (with WebView2 runtime).
Falls back to system default if WebView2 unavailable.
"""

import sys
import time
import threading
import webview

def launch_custom_browser(gradio_url="http://localhost:7860",
                         frameless=True, width=1400, height=900,
                         title="Chat-Gradio-Gguf",
                         maximized=False):
    """
    Launch Gradio app in pywebview window or system browser based on OS/Python.
    """
    import scripts.temporary as tmp
    print(f"[BROWSER] Launching at {gradio_url}")
    
    try:
        import webview
    except ImportError as e:
        print(f"[BROWSER] pywebview not available: {e}")
        import webbrowser
        webbrowser.open(gradio_url)
        return
    
    if tmp.PLATFORM == 'windows' and tmp.OS_VERSION in ['6.1', '6.2']:
        print(f"[BROWSER] {tmp.OS_VERSION} detected - using system browser (WebView2 unsupported/partial).")
        import webbrowser
        webbrowser.open(gradio_url)
        return
    
    gui_backend = None
    if tmp.PLATFORM == 'windows':
        gui_backend = 'edgechromium'  # WebView2
    elif tmp.PLATFORM == 'linux':
        gui_backend = 'gtk' if tmp.OS_VERSION < '24.04' else None  # gtk for older Ubuntu
    
    try:
        window = webview.create_window(
            title=title,
            url=gradio_url,
            width=width,
            height=height,
            resizable=True,
            frameless=frameless,
            easy_drag=frameless,
            maximized=maximized
        )
        
        print(f"[BROWSER] Created with backend {gui_backend}")
        webview.start(gui=gui_backend)
    except Exception as e:
        print(f"[BROWSER] Custom failed ({gui_backend}): {e} - Fallback to system browser")
        import webbrowser
        webbrowser.open(gradio_url)

def wait_for_gradio(url="http://localhost:7860", timeout=30):
    """Wait for Gradio server to be ready."""
    import requests
    
    start_time = time.time()
    print(f"[BROWSER] Waiting for Gradio server at {url}...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                print("[BROWSER] ✓ Gradio server is ready")
                return True
        except:
            pass
        time.sleep(0.5)
    
    print("[BROWSER] ✗ Timeout waiting for Gradio server")
    return False