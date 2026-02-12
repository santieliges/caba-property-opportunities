from scrapper import ArgenPropScrapper, AmbitoDolarScrapper
from scrapper.ArgenPropScrapper import InmuebleData
class Updater:
    def __init__(self):
        pass
    async def fetch(self, entry_id, entry, argenPropScrapper):
        url = entry.get("url")
        if not url:
            return None

        ok, info = await argenPropScrapper.check_url_change(url)
        if not ok:
            return info

        detail = await argenPropScrapper.extract_detail_data(url)

        return InmuebleData(
            id=entry_id,
            url=url,
            image_url=entry.get("image_url"),
            imagen_path=entry.get("imagen_path"),
            **detail,
        ).to_dict()

    