from playwright.async_api import async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

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
    async def check_url_change(self, url: str, timeout: int = 10000) -> bool:
        """
        Verifica si una URL es accesible.
        Devuelve:
            True  -> la URL responde correctamente
            False -> la URL cambió, fue removida o da error (401, 403, 404, etc.)
        """
        page = await self.browser.new_page()

        try:
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=timeout
            )

            if response is None:
                return False

            status = response.status

            if status >= 400:
                return False

            return True

        except PlaywrightTimeoutError:
            return False

        except Exception as e:

            exc_info=True
            return False

        finally:
            await page.close()

