"""Optional browser integration test. Uses an isolated, disposable index."""
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="grounded-ui-") as folder:
        env = {**os.environ, "RAG_PROVIDER": "demo", "RAG_DATA_DIR": folder, "RAG_API_TOKEN": ""}
        subprocess.run([sys.executable, "-m", "grounded", "ingest", "samples"], cwd=ROOT, env=env, check=True)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        process = subprocess.Popen([sys.executable, "-m", "grounded", "serve", "--port", str(port)], cwd=ROOT, env=env)
        try:
            for _ in range(100):
                try:
                    with urllib.request.urlopen(base + "/health", timeout=1):
                        break
                except OSError:
                    if process.poll() is not None:
                        raise RuntimeError("Server exited before becoming ready")
                    time.sleep(0.1)
            else:
                raise RuntimeError("Server did not become ready")
            with sync_playwright() as p:
                launch = {"headless": True}
                if os.environ.get("CHROMIUM_PATH"):
                    launch["executable_path"] = os.environ["CHROMIUM_PATH"]
                browser = p.chromium.launch(**launch)
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(base)
                page.wait_for_function("document.querySelector('#doc-count').textContent === '6'")
                page.get_by_role("button", name="Backup retention").click()
                page.locator("#answer-panel").wait_for(state="visible")
                assert "thirty days" in page.locator("#answer-content").inner_text()
                assert page.locator(".evidence-card").count() > 0
                page.get_by_role("button", name="View source S1", exact=True).first.click()
                assert page.locator("#evidence-S1").evaluate("node => node === document.activeElement")
                with page.expect_download() as download:
                    page.get_by_role("button", name="Export Markdown").click()
                assert download.value.suggested_filename == "grounded-answer.md"
                page.locator("#file-input").set_input_files({"name": "browser-test.md", "mimeType": "text/markdown",
                    "buffer": b"The browser test mascot is a purple otter. <script>alert('xss')</script>"})
                page.wait_for_function("document.querySelector('#doc-count').textContent === '7'")
                page.get_by_label("browser-test.md", exact=False).check()
                page.locator("#question").fill("What is the browser test mascot?")
                page.get_by_role("button", name="Ask Grounded").click()
                page.wait_for_function("document.querySelector('#answer-content').textContent.includes('purple otter')")
                assert page.locator(".evidence-card").count() == 1
                assert page.locator("#answer-content script").count() == 0
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                assert not errors, errors
                browser.close()
            print("Browser smoke test passed: answer, citations, export, upload, filters, XSS-safe rendering, mobile width.")
        finally:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
