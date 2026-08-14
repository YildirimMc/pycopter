import socket
import unittest

from PIL import Image

from gui.app import create_app
from webui import _websocket_origins


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# Panel renders every component into its own shadow root, so the dashboard
# regions are invisible to a plain document.querySelector.
DEEP_MEASURE_JS = """
() => {
    function deepAll(root, acc) {
        root.querySelectorAll('*').forEach(el => {
            acc.push(el);
            if (el.shadowRoot) { deepAll(el.shadowRoot, acc); }
        });
        return acc;
    }
    const all = deepAll(document, []);
    const pick = (selector) => all.filter(el => el.matches && el.matches(selector));
    const size = (el) => {
        const box = el.getBoundingClientRect();
        return {width: Math.round(box.width), height: Math.round(box.height),
                internalOverflow: el.scrollHeight - el.clientHeight};
    };
    const root = document.documentElement;
    const frames = pick('.pycopter-plot-frame');
    return {
        pageOverflowX: root.scrollWidth - root.clientWidth,
        pageOverflowY: root.scrollHeight - root.clientHeight,
        viewportHeight: window.innerHeight,
        shell: pick('.pycopter-shell').map(size),
        columns: pick('.pycopter-column').map(size),
        plotFrame: frames.map(size),
    };
}
"""


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

    def test_dashboard_fits_common_monitor_sizes_without_clipping(self):
        """The dashboard must fit the viewport instead of overflowing the page.

        Regression guard for the fixed 1570 px wide, ~1155 px tall layout that
        clipped on 1920x1080 monitors. Every region now scrolls internally, so
        expanding the collapsible sections must not change the page geometry.
        """
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
            title="pycopter layout test",
        )

        # 1920x937 and 2560x1329 are maximized browser viewports on 1080p and
        # 1440p monitors; 1366x728 is a small laptop.
        viewports = [(1920, 937), (2560, 1329), (1366, 728)]
        measurements = {}
        try:
            with sync_playwright() as playwright:
                try:
                    browser = playwright.chromium.launch(headless=True)
                except PlaywrightError as err:
                    self.skipTest(f"Playwright Chromium is unavailable: {err}")

                for width, height in viewports:
                    page = browser.new_page(viewport={"width": width, "height": height})
                    page.goto(
                        f"http://127.0.0.1:{port}/dashboard",
                        wait_until="networkidle",
                        timeout=30000,
                    )
                    page.wait_for_timeout(1500)
                    collapsed = page.evaluate(DEEP_MEASURE_JS)

                    page.evaluate(
                        """
                        () => {
                            function deepAll(root, acc) {
                                root.querySelectorAll('*').forEach(el => {
                                    acc.push(el);
                                    if (el.shadowRoot) { deepAll(el.shadowRoot, acc); }
                                });
                                return acc;
                            }
                            deepAll(document, [])
                                .filter(el => el.tagName === 'H3' && /Solver Settings|XFOIL Polar|Coaxial Settings/
                                    .test((el.textContent || '').trim()))
                                .forEach(header => header.click());
                        }
                        """
                    )
                    page.wait_for_timeout(800)
                    expanded = page.evaluate(DEEP_MEASURE_JS)
                    measurements[(width, height)] = (collapsed, expanded)
                    page.close()

                browser.close()
        finally:
            stop = getattr(server, "stop", None)
            if stop is not None:
                stop()

        for viewport, (collapsed, expanded) in measurements.items():
            for state, measured in (("collapsed", collapsed), ("expanded", expanded)):
                label = f"{viewport[0]}x{viewport[1]} {state}"
                self.assertEqual(measured["pageOverflowY"], 0, f"{label} scrolls the page vertically")
                self.assertEqual(measured["pageOverflowX"], 0, f"{label} scrolls the page horizontally")
                self.assertEqual(len(measured["plotFrame"]), 1, f"{label} has no plot frame")
                frame = measured["plotFrame"][0]
                self.assertGreater(frame["width"], 400, f"{label} plot frame is too narrow")
                self.assertGreater(frame["height"], 400, f"{label} plot frame is too short")

            # Opening the collapsible sections must not resize the dashboard.
            self.assertEqual(
                collapsed["plotFrame"][0],
                expanded["plotFrame"][0],
                f"{viewport} plot frame changed size when sections were expanded",
            )
            self.assertEqual(
                [column["height"] for column in collapsed["columns"]],
                [column["height"] for column in expanded["columns"]],
                f"{viewport} input columns changed height when sections were expanded",
            )

        # The plot area must actually use the extra room on a larger monitor.
        small_frame = measurements[(1920, 937)][0]["plotFrame"][0]
        large_frame = measurements[(2560, 1329)][0]["plotFrame"][0]
        self.assertGreater(large_frame["width"], small_frame["width"])
        self.assertGreater(large_frame["height"], small_frame["height"])


if __name__ == "__main__":
    unittest.main()
