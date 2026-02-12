from playwright.async_api import async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from typing import Tuple, Union
import re


class BaseScrapper:
    def __init__(self, url_base: str, headless: bool = True):
        self.url_base = url_base
        self.headless = headless

        self.playwright = None
        self.browser = None
        self.page = None
        self.detail_page = None

    # ──────────────────────────────
    # Lifecycle
    # ──────────────────────────────

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless
        )
        self.page = await self.browser.new_page()
        self.detail_page = await self.browser.new_page()

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    # ──────────────────────────────
    # Helpers (pure python)
    # ──────────────────────────────

    def safe_int(self, txt):
        if not txt:
            return None
        m = re.search(r"\d+", txt)
        return int(m.group()) if m else None

    def normalize_label(self, text: str) -> str:
        return (
            text.lower()
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
        )

    async def check_url_change(
        self,
        url: str,
        timeout: int = 10_000
    ) -> Tuple[bool, Union[int, str]]:
        """
        Verifica si una URL es accesible.

        Returns
        -------
        (ok, info)
            ok   : bool
            info : status code o mensaje de error
        """
        page = await self.browser.new_page()

        try:
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=timeout
            )

            if response is None:
                return False, "no response"

            status = response.status

            if status >= 400:
                return False, status

            return True, status

        except PlaywrightTimeoutError as e:
            return False, "timeout"

        except Exception as e:
            return False, str(e)

        finally:
            await page.close()
