import socket
import unittest

from PIL import Image

from gui.app import create_app
from webui import _websocket_origins


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestWebGuiBrowserRender(unittest.TestCase):
    def test_dashboard_renders_in_browser_without_websocket_errors(self):
        try:
            import panel as pn
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as err:
            self.skipTest(f"Browser render dependencies are unavailable: {err}")

        port = _free_port()
        server = pn.serve(
            {"/dashboard": create_app},
            address="127.0.0.1",
            port=port,
            websocket_origin=_websocket_origins(port),
            show=False,
            threaded=True,
            verbose=False,
            title="pycopter browser test",
        )

        try:
            with sync_playwright() as playwright:
                try:
                    browser = playwright.chromium.launch(headless=True)
                except PlaywrightError as err:
                    self.skipTest(f"Playwright Chromium is unavailable: {err}")

                page = browser.new_page(viewport={"width": 1366, "height": 900})
                messages = []
                page.on(
                    "console",
                    lambda msg: messages.append(f"{msg.type}: {msg.text}"),
                )
                page.goto(
                    f"http://127.0.0.1:{port}/dashboard",
                    wait_until="networkidle",
                    timeout=30000,
                )
                page.wait_for_timeout(1500)
                button_count = page.locator("button").count()
                input_count = page.locator("input").count()
                screenshot = page.screenshot(full_page=True)
                browser.close()
        finally:
            stop = getattr(server, "stop", None)
            if stop is not None:
                stop()

        self.assertFalse(
            any("websocket" in message.lower() and "failed" in message.lower() for message in messages),
            "\n".join(messages),
        )
        self.assertGreater(button_count, 5)
        self.assertGreater(input_count, 10)
        image = Image.open(__import__("io").BytesIO(screenshot)).convert("RGB")
        colors = image.getcolors(maxcolors=10_000_000)
        non_white_pixels = sum(count for count, color in colors if color != (255, 255, 255))
        self.assertGreater(non_white_pixels, 50_000)


if __name__ == "__main__":
    unittest.main()
