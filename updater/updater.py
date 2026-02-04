from scrapper.Scrapper import Scrapper, InmuebleData
class Updater:
    def __init__(self, scrapper: Scrapper):
        self.scrapper = scrapper

    def fetch(self, entry_id, entry):
        url = entry.get("url")
        if not url:
            return None

        if self.scrapper.check_url_change(url):
            return None

        detail = self.scrapper.extract_detail_data(url)

        return InmuebleData(
            id=entry_id,
            url=url,
            image_url=entry.get("image_url"),
            imagen_path=entry.get("imagen_path"),
            **detail,
        ).to_dict()

    async def extract_all_pages(self, n_pages):
        await self.scrapper.start()
        inmuebles = await self.scrapper.extract_all_pages(n_pages = n_pages)
        await self.scrapper.close()
        return inmuebles
    # def update_dataset(self):
    #     data = self.storage.get_all()
    #     for entry_id in data:
    #         self.update_entry(entry_id)
        
    # def update_entry(self, entry_id):
    #     data = self.storage.get_all()
    #     entry = data.loc[entry_id]
    #     url = entry.get("url", None)
    #     if url:
    #         has_error_410 = self.scrapper.check_url_change(url)
    #         if has_error_410:
    #             self.storage.close(entry_id, valid_to = datetime.now())
    #         else:
    #             try:
    #                 detail = self.scrapper.extract_detail_data(url)

    #                 data_actualizada = InmuebleData(
    #                     id=entry_id,
    #                     url=url,
    #                     image_url=entry.get("image_url"),
    #                     imagen_path=entry.get("imagen_path"),
    #                     **detail,
    #                 )
    #                 self.storage.update(entry_id, data_actualizada.to_dict())
    #             except Exception as e:
    #                 print(f"Error en {entry['url']}: {e}")




    # def close_endend_urls(self):
    #     data = self.storage.get_all()
    #     for entry_id, entry in data.iterrows():
    #         url = entry.get("url", None)
    #         if url:
    #             has_error_410 = self.scrapper.check_url_change(entry_id, entry)
    #             if has_error_410:
    #                 self.storage.close(entry_id, valid_to = datetime.now())
    #     self.storage.save()