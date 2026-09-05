"""Headless UI smoke test for the Streamlit app."""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path("/opt/cursor/artifacts/screenshots")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
TEST_IMAGE = ROOT / "data/demo/train/pneumonia_005.png"


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1200})
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        page.wait_for_selector("text=Chest X-Ray Pneumonia Detection", timeout=60000)

        page.locator('input[type="file"]').set_input_files(str(TEST_IMAGE))
        page.wait_for_selector("text=Prediction", timeout=60000)
        page.wait_for_timeout(3000)

        screenshot_path = ARTIFACTS / "streamlit_pneumonia_prediction.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"Screenshot saved to {screenshot_path}")
        browser.close()


if __name__ == "__main__":
    main()
