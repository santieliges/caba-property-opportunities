from storage.storage import Storage
from scrapper import ArgenPropScrapper
from updater.updater import Updater
from sync.sync import Synchronizer
import traceback

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

    async def fetch_and_sync_data(self, batch_size: int = 1000):
        """
        Recorre todas las entradas activas del storage,
        obtiene el estado actual vía Updater y delega
        la decisión al Synchronizer.
        Guarda en disco cada `batch_size` entradas procesadas.
        """
        await self.scrapper.start()
        data = self.storage.get_all()

        processed = 0

        for i, (entry_idx, old_entry) in enumerate(data.iterrows(), start=1):
            print(f"[RoutineJob] Procesando {i} de {len(data)} entradas.")
            entry_id = old_entry.get('id')

            try:
                new_entry = await self.updater.fetch(
                    entry_id,
                    old_entry,
                    argenPropScrapper=self.scrapper
                )
                if isinstance(new_entry, dict):        
                    self.sync.sync_entry(entry_id, new_entry)
                    processed += 1
                elif new_entry == 410:
                    self.sync.sync_entry(entry_id, None)
                    processed += 1
                else:
                    print(f"No se pudo acceder a la URL, error: {new_entry}")
                    continue
                # 🔹 guardado por batch
                if processed % batch_size == 0:
                    print(f"[RoutineJob] Guardando batch ({processed} registros procesados)")
                    self.storage.save()

            except Exception:
                print(f"Error running job for entry_id={entry_id}")
                traceback.print_exc()

        # 🔹 guardado final de seguridad
        if processed % batch_size != 0:
            print(f"[RoutineJob] Guardado final ({processed} registros totales)")
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

        