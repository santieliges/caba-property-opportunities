import asyncio
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

import requests

from scraper_service.scraper.scraper_base import BaseScraper
from scraper_service.scraper.SosivaApiClient import (
    SosivaApiClient,
    map_aviso_to_inmueble_fields,
    detect_pozo,
)


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
    informacion_adicional: Optional[str] = None
    pozo: Optional[int] = None
    image_url: Optional[str] = None
    imagen_path: Optional[str] = None

    def __post_init__(self):
        self.precio = self._to_int(self.precio)
        self.expensas = self._to_int(self.expensas)
        self.ambientes = self._to_int(self.ambientes)
        self.dormitorios = self._to_int(self.dormitorios)
        self.banos = self._to_int(self.banos)
        self.cocheras = self._to_int(self.cocheras)
        self.pozo = self._to_int(self.pozo)
        self.area_m2_cubierta = self._to_float(self.area_m2_cubierta)
        self.area_m2_descubierta = self._to_float(self.area_m2_descubierta)
        self.area_m2_total = self._to_float(self.area_m2_total)
        self.latitud = self._to_float(self.latitud)
        self.longitud = self._to_float(self.longitud)

    @staticmethod
    def _to_int(value):
        try:
            if value is None:
                return None
            if isinstance(value, str):
                digits = ''.join(ch for ch in value if ch.isdigit())
                if digits:
                    return int(digits)
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _to_float(value):
        try:
            return float(value) if value is not None else None
        except Exception:
            return None

    def to_dict(self):
        return asdict(self)


class ArgenPropScraper(BaseScraper):
    def __init__(
        self,
        url_base: str,
        headless: bool = True,
        download_images: bool = True,
        use_api_details: bool = True,
    ):
        super().__init__(url_base=url_base, headless=headless)
        self.download_images = download_images
        self.use_api_details = use_api_details
        self.sosiva_api = SosivaApiClient()

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

    def download_image(self, url, listing_id):
        if not url:
            return None

        os.makedirs("images", exist_ok=True)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                path = f"images/{listing_id}.jpg"
                with open(path, "wb") as file:
                    file.write(response.content)
                return path
        except Exception:
            pass
        return None

    async def extract_listings_from_page(self):
        listings = []
        items = await self.page.query_selector_all(".listing__item")

        for item in items:
            try:
                card = await item.query_selector("a.card")
                href = await card.get_attribute("href") if card else None
                listing_id = self.extract_argenprop_id(href)

                if not href:
                    continue

                url = f"https://www.argenprop.com{href}"
                img = await item.query_selector(".card__photos img")
                image_url = None
                if img:
                    image_url = await img.get_attribute("src") or await img.get_attribute("data-src")

                imagen_path = None
                if self.download_images and image_url and listing_id:
                    imagen_path = self.download_image(image_url, listing_id)

                listings.append(
                    {
                        "id": listing_id,
                        "url": url,
                        "image_url": image_url,
                        "imagen_path": imagen_path,
                    }
                )
            except Exception as exc:
                print(f"Error procesando listing: {exc}")

        return listings

    async def extract_detail_data(self, url):
        await self.detail_page.goto(url)
        if await self._looks_like_human_check(self.detail_page):
            await self._pause_for_human_check(self.detail_page, url)
            await self.detail_page.goto(url)
        await self.detail_page.wait_for_timeout(2000)

        precio, moneda, expensas = await self.extract_price_and_expenses()
        superficies = await self.extract_superficies()
        features = await self.extract_features_from_section("section-caracteristicas")
        edificio = await self.extract_features_from_section("section-caracteristicas-del-edificio")
        datos_basicos = await self.extract_features_from_section("section-datos-basicos")
        lat, lon = await self.extract_lat_lon()
        informacion_adicional = await self.extract_informacion_adicional()

        detail = {
            "precio": precio,
            "moneda": moneda,
            "expensas": expensas,
            "area_m2_cubierta": superficies.get("sup_cubierta"),
            "area_m2_descubierta": superficies.get("sup_descubierta"),
            "area_m2_total": superficies.get("sup_total"),
            "antiguedad": features.get("antiguedad") or edificio.get("antiguedad"),
            "ambientes": features.get("cant. ambientes"),
            "dormitorios": features.get("cant. dormitorios"),
            "banos": features.get("cant. baños"),
            "estado": features.get("estado"),
            "disposicion": features.get("disposicion"),
            "orientacion": features.get("orientacion"),
            "cocheras": features.get("cant. cocheras"),
            "estado_edificio": edificio.get("estado edificio"),
            "tipo_unidad": datos_basicos.get("tipo de unidad"),
            "latitud": lat,
            "longitud": lon,
            "informacion_adicional": informacion_adicional,
        }

        if informacion_adicional:
            detail["pozo"] = detect_pozo({"InformacionAdicional_t": informacion_adicional})

        return detail
    async def extract_informacion_adicional(self):
        # Selector para la descripción en ArgenProp (ajustar si es necesario)
        selectors = [
            ".card__description",
            ".description",
            "[data-testid='description']",
            "p.description",
            ".property-description",
        ]
        for selector in selectors:
            element = await self.detail_page.query_selector(selector)
            if element:
                text = await element.inner_text()
                return text.strip()
        return None

    async def extract_all_pages(
        self,
        n_pages=5,
        delay_s: float = 0.0,
        jitter_s: float = 0.0,
        existing_ids: set | None = None,
        max_existing_hits: int | None = None,
    ):
        inmuebles = []
        existing_ids = existing_ids or set()
        existing_hits = 0

        for page_num in range(1, n_pages + 1):
            print(f"Scraping página {page_num}")

            sep = "&" if "?" in str(self.url_base) else "?"
            list_url = f"{self.url_base}{sep}pagina-{page_num}"
            response = await self.page.goto(list_url, wait_until="domcontentloaded")

            if await self._looks_like_human_check(self.page):
                await self._pause_for_human_check(self.page, list_url)
                response = await self.page.goto(list_url, wait_until="domcontentloaded")

            if response is not None and response.status == 429:
                await asyncio.sleep(10.0)
                await self.page.goto(list_url, wait_until="domcontentloaded")

            if response is not None and response.status == 404:
                print("Página no encontrada, terminando.")
                break

            try:
                await self.page.wait_for_selector(".listing__item", timeout=30000)
            except Exception:
                if await self._looks_like_human_check(self.page):
                    await self._pause_for_human_check(self.page, list_url)
                    await self.page.goto(list_url, wait_until="domcontentloaded")
                    await self.page.wait_for_selector(".listing__item", timeout=30000)
                    await self.scroll_page()
                    continue

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("output/scrape_debug", exist_ok=True)
                try:
                    await self.page.screenshot(
                        path=f"output/scrape_debug/list_{ts}.png",
                        full_page=True,
                    )
                except Exception:
                    pass
                try:
                    html = await self.page.content()
                    with open(f"output/scrape_debug/list_{ts}.html", "w", encoding="utf-8") as file:
                        file.write(html)
                except Exception:
                    pass
                raise

            await self.scroll_page()
            listings = await self.extract_listings_from_page()
            print(f" -> {len(listings)} listings encontrados")

            if not listings:
                print("No hay listings en esta página; terminando.")
                break

            for base in listings:
                if base.get("id") is not None and base["id"] in existing_ids:
                    existing_hits += 1
                    print(f"Encontrado listing existente {base['id']} ({existing_hits}/{max_existing_hits or '∞'})")
                    if max_existing_hits and existing_hits >= max_existing_hits:
                        print("Se alcanzó el límite de departamentos ya existentes, deteniendo el scraping.")
                        break
                    continue

                try:
                    detail = None
                    if self.use_api_details and base.get("id") is not None:
                        api_res = await asyncio.to_thread(self.sosiva_api.get_aviso, int(base["id"]))
                        if api_res.status_code == 200 and api_res.json_data:
                            detail = map_aviso_to_inmueble_fields(api_res.json_data)
                            print(f"Using API for {base['id']}, antiguedad: {detail.get('antiguedad')}")
                        elif api_res.status_code in (404, 410):
                            continue

                    if detail is None:
                        detail = await self.extract_detail_data(base["url"])
                        print(f"Using page for {base['id']}, antiguedad: {detail.get('antiguedad')}")

                    # Si detail vino de página y no tiene pozo, calcularlo
                    if 'pozo' not in detail and detail.get('informacion_adicional'):
                        aviso_simulado = {"InformacionAdicional_t": detail['informacion_adicional']}
                        detail['pozo'] = detect_pozo(aviso_simulado)

                    inmuebles.append(
                        InmuebleData(
                            id=base["id"],
                            url=base["url"],
                            image_url=base["image_url"],
                            imagen_path=base["imagen_path"],
                            **detail,
                        )
                    )
                except Exception as exc:
                    print(f"Error en {base['url']}: {exc}")

                if delay_s or jitter_s:
                    await asyncio.sleep(delay_s + (random.random() * jitter_s))

            if max_existing_hits and existing_hits >= max_existing_hits:
                break

        return inmuebles

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
            label = self.normalize_label(p_txt.replace(s_txt, "").replace(":", "").strip())
            features[label] = s_txt.strip()

        return features

    async def extract_superficies(self):
        raw = await self.extract_features_from_section("section-superficie")
        sup = {"sup_cubierta": None, "sup_descubierta": None, "sup_total": None}

        for key, value in raw.items():
            match = re.search(r"\d+([.,]\d+)?", value)
            if not match:
                continue
            val = float(match.group().replace(",", "."))
            if " cubierta" in key:
                sup["sup_cubierta"] = val
            elif "descubierta" in key:
                sup["sup_descubierta"] = val
            elif "total" in key:
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
                precio = int(txt.replace("u$s", "").replace("usd", "").replace("$", "").replace(".", "").strip())
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
        match = re.search(r"--(\d+)$", str(url))
        return int(match.group(1)) if match else None


ArgenPropScrapper = ArgenPropScraper
