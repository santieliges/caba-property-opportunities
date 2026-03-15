from patchright.async_api import async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from typing import Tuple, Union
import asyncio
import os
import re


class BaseScrapper:
    def __init__(self, url_base: str, headless: bool = True):
        self.url_base = url_base
        self.headless = headless

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.detail_page = None

        # Persistir cookies/sesión para no tener que resolver el challenge cada corrida.
        os.makedirs("storage", exist_ok=True)
        self.storage_state_path = os.path.join("storage", f"playwright_state_{self.__class__.__name__}.json")

    # ──────────────────────────────
    # Lifecycle
    # ──────────────────────────────

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=f"playwright_user_data_{self.__class__.__name__}",
            channel="chrome",
            headless=self.headless,
            no_viewport=True
        )
        # # Usar un BrowserContext compartido para que cookies/sesión se re-utilicen entre páginas.
        # if os.path.exists(self.storage_state_path):
        #     self.context = await self.browser.new_context(storage_state=self.storage_state_path)
        # else:
        #     self.context = await self.browser.new_context()
        self.page = await self.browser.new_page()
        self.detail_page = await self.browser.new_page()

        # Evitar que el sitio spamee ventanas emergentes: cerrar cualquier popup/página extra.
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
        # Reusar una pestaña existente (no abrir páginas nuevas) para evitar popups y gatillar menos defensas anti-bot.
        page = self.page if self.page else (await self.context.new_page() if self.context else await self.browser.new_page())

        try:
            # Reintenta si el WAF muestra verificación humana (a veces responde 200 con HTML de challenge).
            for _ in range(3):
                response = await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=timeout
                )

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

        except PlaywrightTimeoutError as e:
            return False, "timeout"

        except Exception as e:
            return False, str(e)

        finally:
            # Si estamos reusando self.page, no lo cerramos.
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
        # No intenta evadir el challenge: solo pausa para permitir que el usuario lo resuelva en el browser visible.
        print(f"[Scrapper] Se detectó un 'compruebe que es humano' en {url}.")
        await asyncio.to_thread(
            input,
            "[Scrapper] Resolvé el challenge en la ventana del browser y presioná Enter para continuar... "
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
