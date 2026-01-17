from playwright.sync_api import sync_playwright
import json
import requests
import os
from scrapper.Scrapper import Scrapper


class AlquilerScrapper(Scrapper):
    def __init__(self, headless=True, url_base=None):
        self.url_base = url_base or "https://www.argenprop.com/departamentos/alquiler/capital-federal/"
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

    
    def extract_listings_from_page(self, page):
        listings = [] 

        items = page.query_selector_all(".listing__item")

        for item in items:
            try:
                listing_id = item.get_attribute("id")

                href = self.safe_attr(item, "a.card", "href")
                full_url = f"https://www.argenprop.com{href}" if href else None

                price = self.safe_text(item, ".card__price")
                expenses = self.safe_attr(item, ".card__expenses", "title")

                features = item.query_selector_all(".card__main-features li")
                area_m2 = None
                dormitorios = None
                antiguedad = None

                points_text = self.safe_text(item, ".card__points")
                visitas = None
                if points_text:
                    visitas = int(points_text.replace(".", "").strip())

                for f in features:
                    text = f.inner_text().lower()
                    if "m²" in text:
                        area_m2 = text
                    elif "dorm" in text:
                        dormitorios = text
                    elif "años" in text:
                        antiguedad = text

                img = item.query_selector(".card__photos img")
                image_url = None
                if img:
                    image_url = img.get_attribute("src") or img.get_attribute("data-src")

                image_path = None
                if image_url:
                    image_path = self.download_image(image_url, listing_id)
                
                lat, lon = self.extract_lat_lon(full_url) if full_url else (None, None)

                listings.append({
                    "id": listing_id,
                    "url": full_url,
                    "precio": price,
                    "expensas": expenses,
                    "area_m2": area_m2,
                    "dormitorios": dormitorios,
                    "antiguedad": antiguedad,
                    "puntaje_arg_prop": visitas,
                    "imagen_path": image_path,
                    "image_url": image_url,
                    "lat": lat,
                    "lon": lon
                })

            except Exception as e:
                print(f"Error al procesar listing {listing_id}: {e}")
                continue

        return listings


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
    
