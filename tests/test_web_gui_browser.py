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
DEEP_ALL_HELPER = """
    function deepAll(root, acc) {
        root.querySelectorAll('*').forEach(el => {
            acc.push(el);
            if (el.shadowRoot) { deepAll(el.shadowRoot, acc); }
        });
        return acc;
    }
"""

DEEP_MEASURE_JS = (
    "() => {"
    + DEEP_ALL_HELPER
    + """
    const all = deepAll(document, []);
    const pick = (selector) => all.filter(el => el.matches && el.matches(selector));
    const size = (el) => {
        const box = el.getBoundingClientRect();
        return {width: Math.round(box.width), height: Math.round(box.height),
                internalOverflow: el.scrollHeight - el.clientHeight};
    };
    const root = document.documentElement;
    const frames = pick('.pycopter-plot-frame');
    const terminals = all.filter(el => String(el.className) === 'terminal xterm');
    return {
        pageOverflowX: root.scrollWidth - root.clientWidth,
        pageOverflowY: root.scrollHeight - root.clientHeight,
        viewportHeight: window.innerHeight,
        shell: pick('.pycopter-shell').map(size),
        columns: pick('.pycopter-column').map(size),
        rail: pick('.pycopter-rail').map(size),
        inspector: pick('.pycopter-inspector').map(size),
        runRail: pick('.pycopter-runrail').map(size),
        plotFrame: frames.map(size),
        logRegion: pick('.pycopter-log-region').map(size),
        metrics: pick('.pycopter-metrics').map(size),
        terminalHeight: terminals.length ? terminals[0].offsetHeight : 0,
    };
}
"""
)

# Centre of the log resize grip, in page coordinates.
GRIP_CENTRE_JS = (
    "() => {"
    + DEEP_ALL_HELPER
    + """
    const grip = deepAll(document, []).find(
        el => el.classList && el.classList.contains('pycopter-log-grip'));
    if (!grip) { return null; }
    const box = grip.getBoundingClientRect();
    return {x: box.left + box.width / 2, y: box.top + box.height / 2};
}
"""
)

# Switch the tab-gated inspector by clicking an icon-rail button.
SELECT_TAB_JS = (
    "(label) => {"
    + DEEP_ALL_HELPER
    + """
    const button = deepAll(document, []).find(
        el => el.tagName === 'BUTTON' && (el.textContent || '').trim() === label);
    if (!button) { return 'missing'; }
    button.click();
    return 'ok';
}
"""
)


def _serve(title):
    import panel as pn

    port = _free_port()
    server = pn.serve(
        {"/dashboard": create_app},
        address="127.0.0.1",
        port=port,
        websocket_origin=_websocket_origins(port),
        show=False,
        threaded=True,
        verbose=False,
        title=title,
    )
    return server, port


class TestWebGuiBrowserRender(unittest.TestCase):
    def test_dashboard_renders_in_browser_without_websocket_errors(self):
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as err:
            self.skipTest(f"Browser render dependencies are unavailable: {err}")

        server, port = _serve("pycopter browser test")
        try:
            with sync_playwright() as playwright:
                try:
                    browser = playwright.chromium.launch(headless=True)
                except PlaywrightError as err:
                    self.skipTest(f"Playwright Chromium is unavailable: {err}")

                page = browser.new_page(viewport={"width": 1600, "height": 900})
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
        self.assertFalse(
            [message for message in messages if message.startswith("error:")],
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

        Regression guard for the fixed-size layout that clipped on 1920x1080
        monitors. Every region scrolls internally, so switching inspector tabs
        must not change the page geometry or any other region's size.
        """
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as err:
            self.skipTest(f"Browser render dependencies are unavailable: {err}")

        server, port = _serve("pycopter layout test")

        # 1920x937 and 2560x1329 are maximized browser viewports on 1080p and
        # 1440p monitors; 1600x728 is a small laptop above the 1420 px point
        # where the body starts to scroll horizontally.
        viewports = [(1920, 937), (2560, 1329), (1600, 728)]
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
                    rotor_tab = page.evaluate(DEEP_MEASURE_JS)

                    for tab in ("BLADE", "OPER", "SOLVER", "XFOIL", "PROP"):
                        self.assertEqual("ok", page.evaluate(SELECT_TAB_JS, tab))
                        page.wait_for_timeout(350)
                    xfoil_tab = page.evaluate(DEEP_MEASURE_JS)
                    measurements[(width, height)] = (rotor_tab, xfoil_tab)
                    page.close()

                browser.close()
        finally:
            stop = getattr(server, "stop", None)
            if stop is not None:
                stop()

        for viewport, (first, last) in measurements.items():
            for state, measured in (("rotor tab", first), ("prop tab", last)):
                label = f"{viewport[0]}x{viewport[1]} {state}"
                self.assertEqual(measured["pageOverflowY"], 0, f"{label} scrolls the page vertically")
                self.assertEqual(measured["pageOverflowX"], 0, f"{label} scrolls the page horizontally")
                self.assertEqual(len(measured["plotFrame"]), 1, f"{label} has no plot frame")
                frame = measured["plotFrame"][0]
                self.assertGreater(frame["width"], 400, f"{label} plot frame is too narrow")
                self.assertGreater(frame["height"], 400, f"{label} plot frame is too short")
                # The fixed-width regions of the layout contract.
                self.assertEqual(60, measured["rail"][0]["width"], f"{label} icon rail width")
                self.assertEqual(320, measured["inspector"][0]["width"], f"{label} inspector width")
                self.assertEqual(384, measured["runRail"][0]["width"], f"{label} run rail width")
                self.assertEqual(30, measured["logRegion"][0]["height"], f"{label} log start height")

            # Changing tabs must not resize any other region.
            self.assertEqual(
                first["plotFrame"][0],
                last["plotFrame"][0],
                f"{viewport} plot frame changed size when the inspector tab changed",
            )
            self.assertEqual(
                [column["height"] for column in first["columns"]],
                [column["height"] for column in last["columns"]],
                f"{viewport} scrolling columns changed height when the inspector tab changed",
            )

        # The canvas must actually use the extra room on a larger monitor.
        small_frame = measurements[(1920, 937)][0]["plotFrame"][0]
        large_frame = measurements[(2560, 1329)][0]["plotFrame"][0]
        self.assertGreater(large_frame["width"], small_frame["width"])
        self.assertGreater(large_frame["height"], small_frame["height"])

    def test_collapsing_the_inspector_gives_its_width_to_the_canvas(self):
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as err:
            self.skipTest(f"Browser render dependencies are unavailable: {err}")

        server, port = _serve("pycopter collapse test")
        try:
            with sync_playwright() as playwright:
                try:
                    browser = playwright.chromium.launch(headless=True)
                except PlaywrightError as err:
                    self.skipTest(f"Playwright Chromium is unavailable: {err}")

                page = browser.new_page(viewport={"width": 1920, "height": 937})
                page.goto(
                    f"http://127.0.0.1:{port}/dashboard",
                    wait_until="networkidle",
                    timeout=30000,
                )
                page.wait_for_timeout(1500)
                before = page.evaluate(DEEP_MEASURE_JS)
                self.assertEqual("ok", page.evaluate(SELECT_TAB_JS, "‹"))
                page.wait_for_timeout(800)
                after = page.evaluate(DEEP_MEASURE_JS)
                browser.close()
        finally:
            stop = getattr(server, "stop", None)
            if stop is not None:
                stop()

        self.assertEqual(320, before["inspector"][0]["width"])
        self.assertEqual(0, after["inspector"][0]["width"])
        self.assertEqual(
            before["plotFrame"][0]["width"] + 320,
            after["plotFrame"][0]["width"],
        )
        self.assertEqual(after["pageOverflowX"], 0)
        self.assertEqual(after["pageOverflowY"], 0)

    def test_output_log_drags_upwards_and_compresses_the_plot(self):
        """The log grows upward on drag, taking height from the result area.

        Its starting height is the minimum, so dragging down cannot shrink it
        further, and the terminal must re-flow rather than leave a tall empty
        box with a short xterm inside it.
        """
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as err:
            self.skipTest(f"Browser render dependencies are unavailable: {err}")

        server, port = _serve("pycopter log splitter test")
        try:
            with sync_playwright() as playwright:
                try:
                    browser = playwright.chromium.launch(headless=True)
                except PlaywrightError as err:
                    self.skipTest(f"Playwright Chromium is unavailable: {err}")

                page = browser.new_page(viewport={"width": 1920, "height": 937})
                page.goto(
                    f"http://127.0.0.1:{port}/dashboard",
                    wait_until="networkidle",
                    timeout=30000,
                )
                page.wait_for_timeout(1500)

                def drag(offset_y):
                    grip = page.evaluate(GRIP_CENTRE_JS)
                    self.assertIsNotNone(grip, "log resize grip is missing")
                    page.mouse.move(grip["x"], grip["y"])
                    page.mouse.down()
                    page.mouse.move(grip["x"], grip["y"] + offset_y, steps=8)
                    page.mouse.up()
                    page.wait_for_timeout(600)

                start = page.evaluate(DEEP_MEASURE_JS)
                drag(-260)
                expanded = page.evaluate(DEEP_MEASURE_JS)
                drag(400)
                clamped = page.evaluate(DEEP_MEASURE_JS)
                browser.close()
        finally:
            stop = getattr(server, "stop", None)
            if stop is not None:
                stop()

        start_log = start["logRegion"][0]["height"]
        expanded_log = expanded["logRegion"][0]["height"]
        clamped_log = clamped["logRegion"][0]["height"]

        # Dragging up grows the log and takes that height from the plot.
        self.assertGreater(expanded_log, start_log + 200)
        self.assertLess(
            expanded["plotFrame"][0]["height"],
            start["plotFrame"][0]["height"] - 200,
        )
        # The terminal itself must grow, not just the panel around it.
        self.assertGreater(expanded["terminalHeight"], start["terminalHeight"] + 200)

        # The starting size is the minimum, so dragging down cannot go below it.
        self.assertEqual(clamped_log, start_log)
        self.assertEqual(clamped["plotFrame"][0]["height"], start["plotFrame"][0]["height"])

        for label, measured in (("start", start), ("expanded", expanded), ("clamped", clamped)):
            self.assertEqual(measured["pageOverflowY"], 0, f"{label} scrolls the page vertically")
            self.assertEqual(measured["pageOverflowX"], 0, f"{label} scrolls the page horizontally")


if __name__ == "__main__":
    unittest.main()
