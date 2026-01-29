from abc import ABC, abstractmethod
from playwright.sync_api import sync_playwright
import json
import requests
import os

class Scrapper(ABC):
    def __init__(self, headless=True, url_base=None):
        if not hasattr(self, "url_base"):
            raise ValueError("La subclase debe definir self.url_base")
        self.api_calls = []
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.page = self.browser.new_page()
        self.detail_page = self.browser.new_page()
        self.data_base = []

    def run(self):
        self.page.goto(self.url_base)
        self.page.wait_for_timeout(8000)
        self.scroll_page()
        self.page.screenshot(path="debug.png", full_page=True)
        self.page.wait_for_timeout(2000)

    def close(self):
        self.browser.close()
        self.playwright.stop()

    def scroll_page(self):
        self.page.evaluate("""
            async () => {
                for (let i = 0; i < 10; i++) {
                    window.scrollBy(0, window.innerHeight);
                    await new Promise(r => setTimeout(r, 800));
                }
            }
        """)
       

    def save_api_calls(self, filename="api_calls.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.api_calls, f, ensure_ascii=False, indent=4)

    def html_snapshot(self, filename="snapshot.html"):
        html_content = self.page.content()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
            return html_content
        print(f"HTML snapshot saved to {filename}")

    def safe_text(self, element, selector):
        el = element.query_selector(selector)
        return el.inner_text().strip() if el else None

    def safe_attr(self, element, selector, attr):
        el = element.query_selector(selector)
        return el.get_attribute(attr) if el else None

    def download_image(self, url, listing_id):
        if not url:
            return None

        os.makedirs("images", exist_ok=True)

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                path = f"images/{listing_id}.jpg"
                with open(path, "wb") as f:
                    f.write(response.content)
                return path
        except Exception:
            pass

        return None

    def extract_lat_lon(self, listing_url):
        page = self.detail_page

        page.goto(listing_url)
        page.wait_for_timeout(3000)

        map_div = page.query_selector("[data-location-map]")
        if not map_div:
            return None, None

        lat_raw = map_div.get_attribute("data-latitude")
        lon_raw = map_div.get_attribute("data-longitude")

        if not lat_raw or not lon_raw:
            return None, None

        return (
            float(lat_raw.replace(",", ".")),
            float(lon_raw.replace(",", "."))
        )


    @abstractmethod
    def extract_listings_from_page(self, entry_id):
        pass

    def extract_all_pages(self, n_pages=5):
        all_listings = []

        for page in range(1, n_pages + 1):
            print(f"Scraping página {page}")

            url = f"{self.url_base}?pagina-{page}"
            self.page.goto(url)

            self.page.wait_for_selector(".listing__item", timeout=10000)
            self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            self.page.wait_for_timeout(3000)

            page_listings = self.extract_listings_from_page(self.page)
            print(f" → {len(page_listings)} listings encontrados")

            all_listings.extend(page_listings)

        missing = [l for l in all_listings if not l.get("area_m2")]
        print(f"Listings sin superficie: {len(missing)}")

        if missing:
            self.solve_missing_superficie(all_listings)

        return all_listings

    def extract_superficie(self, page=None):
        if page is None:
            page = self.page

        superficie_total = None
        superficie_descubierta = None

        items = page.query_selector_all("#section-superficie li")

        for li in items:
            text = li.inner_text().lower()
            strong = li.query_selector("strong")
            if not strong:
                continue

            value = strong.inner_text().replace("m2", "").strip()

            if "sup. total" in text:
                superficie_total = value
            elif "sup. descubierta" in text:
                superficie_descubierta = value

        return superficie_total, superficie_descubierta


    def solve_missing_superficie(self, listings):
        for listing in listings:
            if not listing.get("area_m2"):
                self.page.goto(listing["url"])
                superficie_total, _ = self.extract_superficie(self.page)
                listing["area_m2"] = superficie_total
    
    def check_url_change(self, entry_id, entry, timeout_ms=8000):
        url = entry.get("url")
        if not url:
            return False

        got_410 = False
        page = self.browser.new_page()

        def on_response(response):
            nonlocal got_410
            if response.status == 410:
                got_410 = True

        page.on("response", on_response)

        try:
            page.goto(url, timeout=timeout_ms)
            page.wait_for_timeout(3000)
        except Exception:
            pass
        finally:
            page.close()

        return got_410

    def check_if_a_estrenar(self, entry_id, entry, timeout_ms=8000):
        url = entry.get("url")
        if not url:
            return False

        is_a_estrenar = False
        page = self.browser.new_page()

        try:
            page.goto(url, timeout=timeout_ms)
            page.wait_for_timeout(3000)

            antiguedad_text = self.safe_text(page, 'li[title="Antiguedad"] p.strong')
            if antiguedad_text and "estrenar" in antiguedad_text.lower():
                is_a_estrenar = True

        except Exception:
            pass
        finally:
            page.close()

        return is_a_estrenar