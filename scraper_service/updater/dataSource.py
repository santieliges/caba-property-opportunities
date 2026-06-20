import asyncio
from abc import ABC, abstractmethod

from scraper_service.scraper import BaseScraper
from scraper_service.scraper.SosivaApiClient import (
    SosivaApiClient,
    map_aviso_to_inmueble_fields,
)
from scraper_service.scraper.argenprop_scraper import InmuebleData
from scraper_service.scraper.scraper_base import ScraperHTTPError


class DataSource(ABC):
    @abstractmethod
    async def fetch(self, entry_id, entry) -> tuple[dict | None, int | None]:
        """Devuelve (entry, status); entry puede ser None para errores o bajas."""
        raise NotImplementedError


class SosivaDataSource(DataSource):
    def __init__(self, sosiva_api_client: SosivaApiClient):
        self.sosiva_api = sosiva_api_client

    async def fetch(self, entry_id, entry):
        api_res = await asyncio.to_thread(self.sosiva_api.get_aviso, int(entry_id))
        if api_res.status_code != 200 or not api_res.json_data:
            return None, api_res.status_code

        detail = map_aviso_to_inmueble_fields(api_res.json_data)
        return (
            InmuebleData(
                id=entry_id,
                url=entry.get("url"),
                image_url=entry.get("image_url"),
                imagen_path=entry.get("imagen_path"),
                **detail,
            ).to_dict(),
            api_res.status_code,
        )


class ScrappingDataSource(DataSource):
    def __init__(self, scraper: BaseScraper):
        self.scraper = scraper
        self._lock = asyncio.Lock()

    async def fetch(self, entry_id, entry):
        url = entry.get("url")
        if not url:
            return None, None

        # ArgenPropScraper reutiliza una unica detail_page.
        async with self._lock:
            try:
                detail = await self.scraper.extract_detail_data(url)
            except ScraperHTTPError as exc:
                return None, exc.status_code
        return (
            InmuebleData(
                id=entry_id,
                url=url,
                image_url=entry.get("image_url"),
                imagen_path=entry.get("imagen_path"),
                **detail,
            ).to_dict(),
            200,
        )
