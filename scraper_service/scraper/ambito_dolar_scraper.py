import re

from scraper_service.scraper.scraper_base import BaseScraper


class AmbitoDolarScraper(BaseScraper):
    def __init__(self, headless: bool = True):
        super().__init__(
            url_base="https://www.ambito.com/contenidos",
            headless=headless,
        )
        self.tipos_dolar = {
            "oficial": "dolar-oficial",
            "nacion": "dolar-nacion",
            "mep": "dolar-mep",
            "blue": "dolar-blue",
        }
        self.valores = {}

    async def safe_text(self, page, selector: str, timeout=5000):
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            return (await page.locator(selector).first.inner_text()).strip()
        except Exception:
            return None

    def parse_price(self, text: str):
        if not text:
            return None
        match = re.search(r"\d+(?:,\d+)?", text)
        return float(match.group().replace(",", ".")) if match else None

    async def run(self):
        for nombre, slug in self.tipos_dolar.items():
            url = f"{self.url_base}/{slug}.html"
            page = await self.context.new_page() if self.context else await self.browser.new_page()
            await page.goto(url, wait_until="domcontentloaded")

            compra_txt = await self.safe_text(
                page,
                "span.variation-max-min__value.data-compra",
            )
            venta_txt = await self.safe_text(
                page,
                "span.variation-max-min__value.data-venta",
            )

            self.valores[nombre] = {
                "compra": {"raw": compra_txt, "value": self.parse_price(compra_txt)},
                "venta": {"raw": venta_txt, "value": self.parse_price(venta_txt)},
                "url": url,
            }

            await page.close()

    def get_valores(self):
        return self.valores


AmbitoDolarScrapper = AmbitoDolarScraper
