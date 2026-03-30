import asyncio

from scraper_service.scrapper.ArgenPropScrapper import InmuebleData
from scraper_service.scrapper.SosivaApiClient import (
    SosivaApiClient,
    map_aviso_to_inmueble_fields,
)


class Updater:
    def __init__(self):
        self.sosiva_api = SosivaApiClient()

    async def fetch(self, entry_id, entry, argenPropScrapper):
        url = entry.get("url")
        if not url:
            return None

        api_res = await asyncio.to_thread(self.sosiva_api.get_aviso, int(entry_id))
        if api_res.status_code == 200 and api_res.json_data:
            detail = map_aviso_to_inmueble_fields(api_res.json_data)
            return InmuebleData(
                id=entry_id,
                url=url,
                image_url=entry.get("image_url"),
                imagen_path=entry.get("imagen_path"),
                **detail,
            ).to_dict()

        if api_res.status_code in (404, 410):
            return 410

        return api_res.status_code
