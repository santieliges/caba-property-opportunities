
from scrapper.Scrapper import Scrapper


class AlquilerScrapper(Scrapper):

    def __init__(self, url_base, headless=True):
        self.url_base = url_base
        super().__init__(headless=headless)
    
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
                    elif "años" in text or "estrenar" in text:
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

