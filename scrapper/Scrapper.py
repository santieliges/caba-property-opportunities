from abc import ABC, abstractmethod
from playwright.sync_api import sync_playwright
import json
import requests
import os
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class InmuebleData:
    id: str
    url: str
    precio: Optional[int] = None
    moneda: Optional[str] = None
    expensas: Optional[int] = None
    tipo_unidad: Optional[str] = None

    area_m2_cubierta: Optional[float] = None
    area_m2_descubierta: Optional[float] = None
    area_m2_total: Optional[float] = None

    antiguedad: Optional[int] = None
    estado_edificio: Optional[str] = None
    ambientes: Optional[int] = None
    banos: Optional[int] = None
    estado: Optional[str] = None
    disposicion: Optional[str] = None
    orientacion: Optional[str] = None

    latitud: Optional[float] = None
    longitud: Optional[float] = None

    image_url: Optional[str] = None
    imagen_path: Optional[str] = None


class Scrapper:
    def __init__(self, url_base: str, headless: bool = True):
        self.url_base = url_base
        self.api_calls = []

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.page = self.browser.new_page()
        self.detail_page = self.browser.new_page()

    # ──────────────────────────────
    # Ciclo de vida
    # ──────────────────────────────

    def close(self):
        self.browser.close()
        self.playwright.stop()

    def scroll_page(self, n_scrolls=10, delay_ms=800):
        self.page.evaluate(
            f"""
            async () => {{
                for (let i = 0; i < {n_scrolls}; i++) {{
                    window.scrollBy(0, window.innerHeight);
                    await new Promise(r => setTimeout(r, {delay_ms}));
                }}
            }}
            """
        )

    # ──────────────────────────────
    # Helpers
    # ──────────────────────────────

    def safe_text(self, element, selector):
        el = element.query_selector(selector)
        return el.inner_text().strip() if el else None

    def safe_attr(self, element, selector, attr):
        el = element.query_selector(selector)
        return el.get_attribute(attr) if el else None

    def normalize_label(self, text: str) -> str:
        return (
            text.lower()
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
        )

    # ──────────────────────────────
    # I/O
    # ──────────────────────────────

    def save_api_calls(self, filename="api_calls.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.api_calls, f, ensure_ascii=False, indent=4)

    def html_snapshot(self, filename="snapshot.html"):
        html = self.page.content()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)
        return html

    def download_image(self, url, listing_id):
        if not url:
            return None

        os.makedirs("images", exist_ok=True)

        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                path = f"images/{listing_id}.jpg"
                with open(path, "wb") as f:
                    f.write(r.content)
                return path
        except Exception:
            pass

        return None

    # ──────────────────────────────
    # Extracción: página de resultados
    # ──────────────────────────────

    def extract_listings_from_page(self):
        listings = []
        items = self.page.query_selector_all(".listing__item")

        for item in items:
            try:
                listing_id = item.get_attribute("id")
                href = self.safe_attr(item, "a.card", "href")
                if not href:
                    continue

                url = f"https://www.argenprop.com{href}"

                img = item.query_selector(".card__photos img")
                image_url = (
                    img.get_attribute("src") or img.get_attribute("data-src")
                    if img else None
                )

                imagen_path = (
                    self.download_image(image_url, listing_id)
                    if image_url and listing_id else None
                )

                listings.append({
                    "id": listing_id,
                    "url": url,
                    "image_url": image_url,
                    "imagen_path": imagen_path,
                })

            except Exception as e:
                print(f"Error procesando listing {listing_id}: {e}")

        return listings

    # ──────────────────────────────
    # Extracción: página de detalle
    # ──────────────────────────────

    def extract_detail_data(self, url):
        self.detail_page.goto(url)
        self.detail_page.wait_for_timeout(2000)

        precio, moneda, expensas = self.extract_price_and_expenses()
        superficies = self.extract_superficies()
        features = self.extract_features_from_section("section-caracteristicas")
        edificio = self.extract_features_from_section(
            "section-caracteristicas-del-edificio"
        )
        datos_basicos = self.extract_features_from_section("section-datos-basicos")
        lat, lon = self.extract_lat_lon()

        return {
            "precio": precio,
            "moneda": moneda,
            "expensas": expensas,
            "area_m2_cubierta": superficies.get("sup_cubierta"),
            "area_m2_descubierta": superficies.get("sup_descubierta"),
            "area_m2_total": superficies.get("sup_total"),
            "antiguedad": features.get("antiguedad") or edificio.get("antiguedad"),
            "ambientes": features.get("cant. ambientes"),
            "banos": features.get("cant. baños"),
            "estado": features.get("estado"),
            "disposicion": features.get("disposicion"),
            "orientacion": features.get("orientacion"),
            "estado_edificio": edificio.get("estado edificio"),
            "tipo_unidad": datos_basicos.get("tipo de unidad"),
            "latitud": lat,
            "longitud": lon,
        }

    # ──────────────────────────────
    # Orquestador
    # ──────────────────────────────

    def extract_all_pages(self, n_pages=5):
        inmuebles = []

        for page_num in range(1, n_pages + 1):
            print(f"Scraping página {page_num}")

            self.page.goto(f"{self.url_base}?pagina-{page_num}")
            self.page.wait_for_selector(".listing__item", timeout=10000)
            self.scroll_page()

            listings = self.extract_listings_from_page()
            print(f" → {len(listings)} listings encontrados")

            for base in listings:
                try:
                    detail = self.extract_detail_data(base["url"])

                    inmuebles.append(
                        InmuebleData(
                            id=base["id"],
                            url=base["url"],
                            image_url=base["image_url"],
                            imagen_path=base["imagen_path"],
                            **detail,
                        )
                    )

                except Exception as e:
                    print(f"Error en {base['url']}: {e}")

        return inmuebles

    # ──────────────────────────────
    # Extractores específicos
    # ──────────────────────────────

    def extract_features_from_section(self, section_id):
        features = {}
        items = self.detail_page.query_selector_all(f"#{section_id} li")

        for li in items:
            p = li.query_selector("p")
            strong = li.query_selector("strong")
            if not p or not strong:
                continue

            label = self.normalize_label(
                p.inner_text()
                .replace(strong.inner_text(), "")
                .replace(":", "")
                .strip()
            )

            features[label] = strong.inner_text().strip()

        return features

    def extract_superficies(self):
        raw = self.extract_features_from_section("section-superficie")

        sup = {"sup_cubierta": None, "sup_descubierta": None, "sup_total": None}

        for k, v in raw.items():
            m = re.search(r"\d+", v)
            if not m:
                continue

            val = int(m.group())

            if " cubierta" in k:
                sup["sup_cubierta"] = val
            elif "descubierta" in k:
                sup["sup_descubierta"] = val
            elif "total" in k:
                sup["sup_total"] = val

        if sup["sup_cubierta"] is None and sup["sup_total"] is not None:
            sup["sup_cubierta"] = sup["sup_total"]

        return sup

    def extract_lat_lon(self):
        map_div = self.detail_page.query_selector("[data-location-map]")
        if not map_div:
            return None, None

        lat = map_div.get_attribute("data-latitude")
        lon = map_div.get_attribute("data-longitude")

        if not lat or not lon:
            return None, None

        return float(lat.replace(",", ".")), float(lon.replace(",", "."))

    def extract_price_and_expenses(self):
        precio = expensas = moneda = None

        price_el = self.detail_page.query_selector(".titlebar__price")
        if price_el:
            txt = price_el.inner_text().lower()
            moneda = "USD" if "u$s" in txt or "usd" in txt else "ARS"
            try:
                precio = int(
                    txt.replace("u$s", "")
                       .replace("usd", "")
                       .replace("$", "")
                       .replace(".", "")
                       .strip()
                )
            except ValueError:
                pass

        exp_el = self.detail_page.query_selector(".titlebar__expenses")
        if exp_el:
            try:
                expensas = int(
                    exp_el.inner_text()
                    .lower()
                    .replace("+", "")
                    .replace("$", "")
                    .replace("expensas", "")
                    .replace(".", "")
                    .strip()
                )
            except ValueError:
                pass

        return precio, moneda, expensas


    def update_old_data(self, listings):
        inmuebles = []


        print(f" → {len(listings)} listings encontrados")

        for base in listings:
            try:
                detail = self.extract_detail_data(base["url"])

                inmuebles.append(
                    InmuebleData(
                        id=base["id"],
                        url=base["url"],
                        image_url=base["image_url"],
                        imagen_path=base["imagen_path"],
                        **detail,
                    )
                )

            except Exception as e:
                print(f"Error en {base['url']}: {e}")

        return inmuebles
