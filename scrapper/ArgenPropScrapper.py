import os
from scrapper.Scrapper import BaseScrapper
import re
import requests
from dataclasses import dataclass, asdict
from typing import Optional
import json

# ──────────────────────────────
# Data model
# ──────────────────────────────

@dataclass
class InmuebleData:
    id: int
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
    dormitorios: Optional[int] = None
    banos: Optional[int] = None
    estado: Optional[str] = None
    disposicion: Optional[str] = None
    orientacion: Optional[str] = None
    cocheras: Optional[int] = None

    latitud: Optional[float] = None
    longitud: Optional[float] = None

    image_url: Optional[str] = None
    imagen_path: Optional[str] = None

    def __post_init__(self):
        self.precio = self._to_int(self.precio)
        self.expensas = self._to_int(self.expensas)
        self.ambientes = self._to_int(self.ambientes)
        self.dormitorios = self._to_int(self.dormitorios)
        self.banos = self._to_int(self.banos)
        self.cocheras = self._to_int(self.cocheras)

        self.area_m2_cubierta = self._to_float(self.area_m2_cubierta)
        self.area_m2_descubierta = self._to_float(self.area_m2_descubierta)
        self.area_m2_total = self._to_float(self.area_m2_total)

        self.latitud = self._to_float(self.latitud)
        self.longitud = self._to_float(self.longitud)

    @staticmethod
    def _to_int(v):
        try:
            return int(v) if v is not None else None
        except Exception:
            return None

    @staticmethod
    def _to_float(v):
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    def to_dict(self):
        return asdict(self)

    
class ArgenPropScrapper(BaseScrapper):
    def __init__(self, url_base: str, headless: bool = True):
        super().__init__(url_base=url_base, headless=headless)

    # ──────────────────────────────
    # Page interactions 
    # ──────────────────────────────
    async def scroll_page(self, n_scrolls=10, delay_ms=800):
        await self.page.evaluate(
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
    # I/O
    # ──────────────────────────────

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
    # Result page extraction
    # ──────────────────────────────

    async def extract_listings_from_page(self):
        listings = []
        items = await self.page.query_selector_all(".listing__item")

        for item in items:
            try:
                href = await item.get_attribute("href")

                card = await item.query_selector("a.card")
                href = await card.get_attribute("href") if card else None
                listing_id = self.extract_argenprop_id(href)

                if not href:
                    continue

                url = f"https://www.argenprop.com{href}"

                img = await item.query_selector(".card__photos img")
                image_url = None
                if img:
                    image_url = (
                        await img.get_attribute("src")
                        or await img.get_attribute("data-src")
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
                print(f"Error procesando listing: {e}")

        return listings

    # ──────────────────────────────
    # Detail page extraction
    # ──────────────────────────────

    async def extract_detail_data(self, url):
        await self.detail_page.goto(url)
        await self.detail_page.wait_for_timeout(2000)

        precio, moneda, expensas = await self.extract_price_and_expenses()
        superficies = await self.extract_superficies()
        features = await self.extract_features_from_section("section-caracteristicas")
        edificio = await self.extract_features_from_section(
            "section-caracteristicas-del-edificio"
        )
        datos_basicos = await self.extract_features_from_section("section-datos-basicos")
        lat, lon = await self.extract_lat_lon()

        return {
            "precio": precio,
            "moneda": moneda,
            "expensas": expensas,
            "area_m2_cubierta": superficies.get("sup_cubierta"),
            "area_m2_descubierta": superficies.get("sup_descubierta"),
            "area_m2_total": superficies.get("sup_total"),
            "antiguedad": features.get("antiguedad") or edificio.get("antiguedad"),
            "ambientes": features.get("cant. ambientes"),
            "dormitorios":features.get("cant. dormitorios"),
            "banos": features.get("cant. baños"),
            "estado": features.get("estado"),
            "disposicion": features.get("disposicion"),
            "orientacion": features.get("orientacion"),
            "cocheras": features.get("cant. cocheras"),
            "estado_edificio": edificio.get("estado edificio"),
            "tipo_unidad": datos_basicos.get("tipo de unidad"),
            "latitud": lat,
            "longitud": lon,
        }

    # ──────────────────────────────
    # Orchestrator
    # ──────────────────────────────

    async def extract_all_pages(self, n_pages=5):
        inmuebles = []

        for page_num in range(1, n_pages + 1):
            print(f"Scraping página {page_num}")

            await self.page.goto(f"{self.url_base}?pagina-{page_num}")
            await self.page.wait_for_selector(".listing__item", timeout=10000)
            await self.scroll_page()

            listings = await self.extract_listings_from_page()
            print(f" → {len(listings)} listings encontrados")

            for base in listings:
                try:
                    detail = await self.extract_detail_data(base["url"])

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
    # Specific extractors
    # ──────────────────────────────

    async def extract_features_from_section(self, section_id):
        features = {}
        items = await self.detail_page.query_selector_all(f"#{section_id} li")

        for li in items:
            p = await li.query_selector("p")
            strong = await li.query_selector("strong")
            if not p or not strong:
                continue

            p_txt = await p.inner_text()
            s_txt = await strong.inner_text()

            label = self.normalize_label(
                p_txt.replace(s_txt, "").replace(":", "").strip()
            )

            features[label] = s_txt.strip()

        return features

    async def extract_superficies(self):
        raw = await self.extract_features_from_section("section-superficie")

        sup = {"sup_cubierta": None, "sup_descubierta": None, "sup_total": None}

        for k, v in raw.items():
            m = re.search(r"\d+([.,]\d+)?", v)
            if not m:
                continue

            val = float(m.group().replace(",", "."))

            if " cubierta" in k:
                sup["sup_cubierta"] = val
            elif "descubierta" in k:
                sup["sup_descubierta"] = val
            elif "total" in k:
                sup["sup_total"] = val

        if sup["sup_cubierta"] is None and sup["sup_total"] is not None:
            sup["sup_cubierta"] = sup["sup_total"]

        return sup

    async def extract_lat_lon(self):
        map_div = await self.detail_page.query_selector("[data-location-map]")
        if not map_div:
            return None, None

        lat = await map_div.get_attribute("data-latitude")
        lon = await map_div.get_attribute("data-longitude")

        if not lat or not lon:
            return None, None

        return float(lat.replace(",", ".")), float(lon.replace(",", "."))

    async def extract_price_and_expenses(self):
        precio = expensas = moneda = None

        price_el = await self.detail_page.query_selector(".titlebar__price")
        if price_el:
            txt = (await price_el.inner_text()).lower()
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

        exp_el = await self.detail_page.query_selector(".titlebar__expenses")
        if exp_el:
            try:
                expensas = int(
                    (await exp_el.inner_text())
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
    
    def extract_argenprop_id(self, url):
        m = re.search(r'--(\d+)$', str(url))
        return int(m.group(1)) if m else None

    
