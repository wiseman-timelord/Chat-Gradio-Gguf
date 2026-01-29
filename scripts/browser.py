# scripts/browser.py
"""
Custom browser window using Qt WebEngine (Chromium-based).
Provides modern JS support for Gradio on Windows 7-11 and Ubuntu 22-25.
- Windows 7/8/8.1: Uses Qt5 WebEngine (PyQt5)
- Windows 10/11: Uses Qt6 WebEngine (PyQt6)
- Ubuntu 22-25: Uses Qt6 WebEngine (PyQt6)
Falls back to system default browser if Qt WebEngine unavailable.
"""

import sys
import time
import threading
import scripts.configuration as cfg
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings

# Global reference to Qt application and signal handler for shutdown
_qt_app = None
_qt_browser = None
_signal_handler = None

def close_browser():
    """Close the Qt browser window from any thread (thread-safe)."""
    global _signal_handler
    
    if _signal_handler is not None:
        try:
            print("[BROWSER] Requesting close via signal...")
            _signal_handler.request_close()
        except Exception as e:
            print(f"[BROWSER] Signal close failed: {e}")
            # Fallback to direct quit attempt
            if _qt_app is not None:
                try:
                    _qt_app.quit()
                except:
                    pass
    else:
        print("[BROWSER] No signal handler - attempting direct close")
        if _qt_app is not None:
            try:
                _qt_app.quit()
            except:
                pass
    
    print("[BROWSER] Close requested")

def launch_custom_browser(gradio_url="http://localhost:7860",
                         frameless=False, width=1400, height=900,
                         title="Chat-Gradio-Gguf",
                         maximized=False):
    """
    Launch Gradio app in Qt WebEngine window or system browser.
    """
    import scripts.configuration as cfg
    print(f"[BROWSER] Launching at {gradio_url}")
    print(f"[BROWSER] Platform: {cfg.PLATFORM}, Windows Version: {cfg.WINDOWS_VERSION}")
    
    # Determine which Qt version to use based on platform/OS
    qt_version = None
    
    if cfg.PLATFORM == 'windows':
        win_ver = cfg.WINDOWS_VERSION
        if win_ver in ['7', '8', '8.1']:
            qt_version = 5
            print(f"[BROWSER] Windows {win_ver} detected - using Qt5 WebEngine")
        elif win_ver in ['10', '11']:
            qt_version = 6
            print(f"[BROWSER] Windows {win_ver} detected - using Qt6 WebEngine")
        else:
            # Unknown Windows version, try Qt6 first
            qt_version = 6
            print(f"[BROWSER] Windows version '{win_ver}' - defaulting to Qt6 WebEngine")
    elif cfg.PLATFORM == 'linux':
        qt_version = 6
        print(f"[BROWSER] Linux detected - using Qt6 WebEngine")
    else:
        qt_version = 6
        print(f"[BROWSER] Unknown platform '{cfg.PLATFORM}' - defaulting to Qt6 WebEngine")
    
    # Try to launch Qt WebEngine browser
    try:
        if qt_version == 5:
            _launch_qt5_browser(gradio_url, title, width, height, frameless, maximized)
        else:
            _launch_qt6_browser(gradio_url, title, width, height, frameless, maximized)
    except ImportError as e:
        print(f"[BROWSER] Qt WebEngine not available: {e}")
        print("[BROWSER] Falling back to system browser")
        import webbrowser
        webbrowser.open(gradio_url)
    except Exception as e:
        print(f"[BROWSER] Qt WebEngine failed: {e}")
        import traceback
        traceback.print_exc()
        print("[BROWSER] Falling back to system browser")
        import webbrowser
        webbrowser.open(gradio_url)


def _launch_qt5_browser(url, title, width, height, frameless, maximized):
    """Launch browser using PyQt5 + Qt5 WebEngine (Windows 7/8/8.1)"""
    global _qt_app, _qt_browser, _signal_handler
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage, QWebEngineSettings
    from PyQt5.QtCore import QUrl, Qt, pyqtSignal, QObject

    # Signal handler for thread-safe closing
    class CloseSignalHandler(QObject):
        close_signal = pyqtSignal()
        
        def __init__(self, app):
            super().__init__()
            self._app = app
            self.close_signal.connect(self._do_close)
        
        def _do_close(self):
            print("[BROWSER] Close signal received in main thread")
            if self._app:
                self._app.quit()
        
        def request_close(self):
            """Call this from any thread - emits signal to main thread"""
            self.close_signal.emit()

    class CustomWebPage(QWebEnginePage):
        def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
            # Only print errors to reduce noise
            if level == 2:
                print(f"[JS ERROR]: {message}")

    # Must create QApplication before any Qt widgets
    _qt_app = QApplication(sys.argv)

    # Create signal handler for thread-safe closing
    _signal_handler = CloseSignalHandler(_qt_app) 

    # Create the web view
    _qt_browser = QWebEngineView()
    _qt_browser.setWindowTitle(title)

    if frameless:
        _qt_browser.setWindowFlags(Qt.FramelessWindowHint)

    # Configure web engine settings
    settings = _qt_browser.settings()
    settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
    settings.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
    settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
    settings.setAttribute(QWebEngineSettings.PluginsEnabled, True)

    # Set custom page for console logging
    page = CustomWebPage(_qt_browser)
    _qt_browser.setPage(page)

    _qt_browser.resize(width, height)

    # Load URL
    print(f"[BROWSER] Loading URL: {url}")
    _qt_browser.setUrl(QUrl(url))

    if maximized:
        _qt_browser.showMaximized()
    else:
        _qt_browser.show()

    print(f"[BROWSER] Qt5 WebEngine window created")
    _qt_app.exec_()
    print("[BROWSER] Qt5 event loop exited")


def _launch_qt6_browser(url, title, width, height, frameless, maximized):
    """Launch browser using PyQt6 + Qt6 WebEngine (Windows 10/11, Ubuntu 22-25)"""
    global _qt_app, _qt_browser, _signal_handler
    import os
    import scripts.configuration as cfg

    # Linux: Set Chromium flags for sandbox issues (especially when running as root)
    if cfg.PLATFORM == 'linux':
        if os.geteuid() == 0:
            print("[BROWSER] Running as root - disabling Chromium sandbox")
            os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--no-sandbox --disable-gpu-sandbox"
        else:
            # Even non-root may need this on some Ubuntu systems
            os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu-sandbox"

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEnginePage
    from PyQt6.QtCore import QUrl, Qt, QTimer, pyqtSignal, QObject

    # Signal handler for thread-safe closing
    class CloseSignalHandler(QObject):
        close_signal = pyqtSignal()
        
        def __init__(self, app):
            super().__init__()
            self._app = app
            self.close_signal.connect(self._do_close)
        
        def _do_close(self):
            print("[BROWSER] Close signal received in main thread")
            if self._app:
                self._app.quit()
        
        def request_close(self):
            """Call this from any thread - emits signal to main thread"""
            self.close_signal.emit()

    # Must create QApplication before any Qt widgets
    _qt_app = QApplication(sys.argv)

    # Create signal handler for thread-safe closing
    _signal_handler = CloseSignalHandler(_qt_app)

    # Create the web view
    _qt_browser = QWebEngineView()
    _qt_browser.setWindowTitle(title)

    if frameless:
        _qt_browser.setWindowFlags(Qt.WindowType.FramelessWindowHint)

    # Configure web engine settings
    settings = _qt_browser.settings()
    settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, True)

    _qt_browser.resize(width, height)

    if maximized:
        _qt_browser.showMaximized()
    else:
        _qt_browser.show()

    # Load URL after window is shown (gives time for initialization)
    def load_url():
        print(f"[BROWSER] Loading URL: {url}")
        _qt_browser.setUrl(QUrl(url))

    QTimer.singleShot(100, load_url)

    print(f"[BROWSER] Qt6 WebEngine window created")
    _qt_app.exec()
    print("[BROWSER] Qt6 event loop exited")


def wait_for_gradio(url="http://localhost:7860", timeout=30):
    """Wait for Gradio server to be fully ready."""
    import requests
    
    start_time = time.time()
    print(f"[BROWSER] Waiting for Gradio server at {url}...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                # Check if it's actually Gradio content (not just a 200 response)
                content = response.text.lower()
                if 'gradio' in content or 'svelte' in content or '<script' in content:
                    # Give Gradio a moment to fully initialize its components
                    time.sleep(1.5)
                    print("[BROWSER] Gradio server is ready")
                    return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    
    print("[BROWSER] Timeout waiting for Gradio server")
    return False
