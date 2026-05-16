import asyncio
import os
import re
from pathlib import Path
from typing import Tuple, Union

from patchright.async_api import async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError


class BaseScraper:
    def __init__(self, url_base: str, headless: bool = True):
        self.url_base = url_base
        self.headless = headless
        self.browser_channel = os.getenv("BROWSER_CHANNEL")

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.detail_page = None

        storage_dir = Path(
            os.getenv("SCRAPER_STORAGE_DIR", "storage")
        )
        user_data_dir = Path(
            os.getenv(
                "PLAYWRIGHT_USER_DATA_DIR",
                f"playwright_user_data_{self.__class__.__name__}",
            )
        )

        storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        user_data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.storage_state_path = str(
            storage_dir / f"playwright_state_{self.__class__.__name__}.json"
        )
        self.user_data_dir = str(user_data_dir)

    async def start(self):
        self.playwright = await async_playwright().start()
        launch_options = {
            "user_data_dir": self.user_data_dir,
            "headless": self.headless,
            "no_viewport": True,
        }
        if self.browser_channel:
            launch_options["channel"] = self.browser_channel

        self.browser = await self.playwright.chromium.launch_persistent_context(
            **launch_options,
        )
        self.page = await self.browser.new_page()
        self.detail_page = await self.browser.new_page()
        self.page.on("popup", lambda p: asyncio.create_task(self._close_popup(p)))
        self.detail_page.on("popup", lambda p: asyncio.create_task(self._close_popup(p)))

    async def close(self):
        if self.context:
            try:
                await self.context.storage_state(path=self.storage_state_path)
            except Exception:
                pass
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    def safe_int(self, txt):
        if not txt:
            return None
        match = re.search(r"\d+", txt)
        return int(match.group()) if match else None

    def normalize_label(self, text: str) -> str:
        return (
            text.lower()
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
        )

    async def check_url_change(self, url: str, timeout: int = 10_000) -> Tuple[bool, Union[int, str]]:
        page = (
            self.page
            if self.page
            else (await self.context.new_page() if self.context else await self.browser.new_page())
        )

        try:
            for _ in range(3):
                response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout)

                if response is None:
                    return False, "no response"

                if await self._looks_like_human_check(page):
                    await self._pause_for_human_check(page, url)
                    continue

                status = response.status
                if status >= 400:
                    return False, status

                return True, status

            return False, "human_check"
        except PlaywrightTimeoutError:
            return False, "timeout"
        except Exception as exc:
            return False, str(exc)
        finally:
            if page is not self.page:
                await page.close()

    async def _looks_like_human_check(self, page) -> bool:
        try:
            locator = page.locator(
                "text=Confirme que es humano, text=Compruebe que es humano, text=Comprobá que sos humano, text=Human Verification, text=Check you are human"
            )
            return await locator.count() > 0
        except Exception:
            return False

    async def _pause_for_human_check(self, page, url: str) -> None:
        print(f"[Scraper] Se detectó un 'compruebe que es humano' en {url}.")
        await asyncio.to_thread(
            input,
            "[Scraper] Resolvé el challenge en la ventana del browser y presioná Enter para continuar... ",
        )

    async def _close_popup(self, popup_page) -> None:
        try:
            await popup_page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass
        try:
            await popup_page.close()
        except Exception:
            pass


BaseScrapper = BaseScraper
