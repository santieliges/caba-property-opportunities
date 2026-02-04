from storage.storage import Storage
from scrapper.Scrapper import ArgenPropScrapper
from updater.updater import Updater
from sync.sync import Synchronizer


class RoutineJob:
    def __init__(
        self,
        storage: Storage,
        scrapper: ArgenPropScrapper,
        updater: Updater,
        synchronizer: Synchronizer,
    ):
        self.storage = storage
        self.scrapper = scrapper
        self.updater = updater
        self.sync = synchronizer

    async def fetch_and_sync_data(self):
        """
        Recorre todas las entradas activas del storage,
        obtiene el estado actual vía Updater y delega
        la decisión al Synchronizer.
        """
        await self.scrapper.start()
        data = self.storage.get_all()

        for entry_idx, old_entry in data.iterrows():
            print(f"[RoutineJob] Procesando {entry_idx} de {len(data)}  entradas.")
            entry_id = old_entry.get('id')
            try:
                new_entry = await self.updater.fetch(entry_id, old_entry, argenPropScrapper=self.scrapper)
                self.sync.sync_entry(entry_id, new_entry)

            except Exception as e:
                # decisión explícita: el job no cae por una URL rota
                print(f"[RoutineJob] Error procesando {entry_id}: {e}")

        # una sola escritura al final
        self.storage.save()
        await self.scrapper.close()

    async def fetch_and_sync_new_listings(self, n_pages=5):
        await self.scrapper.start()
        new_listings = await self.scrapper.extract_all_pages(n_pages=n_pages)
        await self.scrapper.close()

        for listing in new_listings:
            entry = listing.to_dict()
            self.sync.sync_entry(entry_id=entry.get("id"), entry=entry)

        self.storage.save()

        