from playwright.sync_api import sync_playwright

class DolarHoyScrapper:

    def __init__(self, headless=True):
        self.base_url = "https://www.ambito.com/contenidos"
        self.tipos_dolar = {
            "oficial": "dolar-oficial",
            "nacion": "dolar-nacion",
            "mep": "dolar-mep",
            "blue": "dolar-blue",
        }
        self.valores = {}

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)

    def safe_text(self, page, selector):
        try:
            page.wait_for_selector(selector, timeout=5000)
            return page.locator(selector).first.inner_text().strip()
        except:
            return None


    def run(self):
        page = self.browser.new_page()

        for nombre, slug in self.tipos_dolar.items():
            url = f"{self.base_url}/{slug}.html"
            page.goto(url)

            valor = self.safe_text(
                page,
                "span.variation-max-min__value.data-venta"
            )

            self.valores[nombre] = valor

        page.close()
        self.browser.close()
        self.playwright.stop()

        
    def get_valores(self):
        return self.valores