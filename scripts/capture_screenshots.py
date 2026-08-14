"""Capture PyCopter Web UI screenshots for the README.

Runs the dashboard headlessly, drives a full hover calculation, and saves one
PNG per view under ``docs/images``. Regenerate after UI changes with:

    python scripts/capture_screenshots.py

Requires the development extras (``playwright`` plus ``playwright install
chromium``). XFOIL polar generation runs on the first pass, so the initial
capture takes a few minutes; later runs reuse the cached polars.
"""

from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import panel as pn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from gui.app import create_app  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "docs" / "images"
VIEWPORT = {"width": 1920, "height": 937}
HOVER_TIMEOUT_S = 900

# Panel renders each component into its own shadow root, so every browser-side
# helper needs this walker rather than a plain document.querySelector.
DEEP_ALL_HELPER = """
    function deepAll(root, acc) {
        root.querySelectorAll('*').forEach(el => {
            acc.push(el);
            if (el.shadowRoot) { deepAll(el.shadowRoot, acc); }
        });
        return acc;
    }
"""


def _js(params: str, body: str) -> str:
    """Build a browser function whose body can use deepAll()."""
    return f"({params}) => {{\n{DEEP_ALL_HELPER}\n{body}\n}}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _click_button(page, label: str) -> None:
    clicked = page.evaluate(
        _js(
            "label",
            """
            const button = deepAll(document, []).find(
                el => el.tagName === 'BUTTON' && (el.textContent || '').includes(label));
            if (!button || button.disabled) { return false; }
            button.click();
            return true;
            """,
        ),
        label,
    )
    if not clicked:
        raise RuntimeError(f"Button {label!r} was missing or disabled.")


def _select_plot(page, plot_name: str) -> None:
    page.evaluate(
        _js(
            "plotName",
            """
            const select = deepAll(document, []).find(
                el => el.tagName === 'SELECT'
                    && [...el.options].some(option => option.value === plotName));
            if (!select) { throw new Error('plot select not found: ' + plotName); }
            select.value = plotName;
            select.dispatchEvent(new Event('change', {bubbles: true}));
            """,
        ),
        plot_name,
    )
    page.wait_for_timeout(400)


def _activate_tab(page, tab_label: str) -> None:
    page.evaluate(
        _js(
            "label",
            """
            const tab = deepAll(document, []).find(
                el => el.classList && el.classList.contains('bk-tab')
                    && (el.textContent || '').trim() === label);
            if (!tab) { throw new Error('tab not found: ' + label); }
            tab.click();
            """,
        ),
        tab_label,
    )
    page.wait_for_timeout(700)


def _plot_is_ready(page) -> bool:
    return bool(
        page.evaluate(
            _js(
                "",
                """
                const frame = deepAll(document, []).find(
                    el => el.classList && el.classList.contains('pycopter-plot-frame'));
                if (!frame) { return false; }
                const img = (frame.shadowRoot || frame).querySelector('img');
                // The placeholder figure is a few kB; a real plot is much larger.
                return !!img && img.src.length > 40000;
                """,
            )
        )
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    server = pn.serve(
        {"/dashboard": create_app},
        address="127.0.0.1",
        port=port,
        websocket_origin=[f"127.0.0.1:{port}", f"localhost:{port}"],
        show=False,
        threaded=True,
        verbose=False,
        title="pycopter Web UI",
    )

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            # Scale factor 1 keeps the committed PNGs small; 1920 px wide is
            # already more than GitHub renders a README image at.
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{port}/dashboard", wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(2000)

            page.screenshot(path=OUTPUT_DIR / "dashboard-empty.png")
            print(f"saved {OUTPUT_DIR / 'dashboard-empty.png'}")

            _click_button(page, "Initialize Rotor")
            page.wait_for_timeout(1500)

            print("running hover calculation (first run generates XFOIL polars)...")
            _click_button(page, "Calculate Hover")
            deadline = time.monotonic() + HOVER_TIMEOUT_S
            while time.monotonic() < deadline:
                if _plot_is_ready(page):
                    break
                page.wait_for_timeout(2000)
            else:
                raise RuntimeError("Hover calculation did not finish before the timeout.")
            page.wait_for_timeout(1500)

            page.screenshot(path=OUTPUT_DIR / "dashboard-hover.png")
            print(f"saved {OUTPUT_DIR / 'dashboard-hover.png'}")

            for plot_name, filename in (
                ("Radial Loads", "plot-radial-loads.png"),
                ("Alpha, Re, Mach vs Radius", "plot-alpha-re-mach.png"),
                ("Disk Loading Sensitivity", "plot-disk-loading.png"),
            ):
                _select_plot(page, plot_name)
                _click_button(page, "Generate Plot")
                deadline = time.monotonic() + HOVER_TIMEOUT_S
                while time.monotonic() < deadline:
                    if _plot_is_ready(page):
                        break
                    page.wait_for_timeout(2000)
                page.wait_for_timeout(1200)
                page.screenshot(path=OUTPUT_DIR / filename)
                print(f"saved {OUTPUT_DIR / filename}")

            _activate_tab(page, "Summary")
            page.screenshot(path=OUTPUT_DIR / "dashboard-summary.png")
            print(f"saved {OUTPUT_DIR / 'dashboard-summary.png'}")

            _activate_tab(page, "Blade Element Loads")
            page.screenshot(path=OUTPUT_DIR / "dashboard-blade-loads.png")
            print(f"saved {OUTPUT_DIR / 'dashboard-blade-loads.png'}")

            browser.close()
    finally:
        stop = getattr(server, "stop", None)
        if stop is not None:
            stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
