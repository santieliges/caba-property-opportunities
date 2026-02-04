from storage.storage import Storage
from updater.updater import Updater
from sync.sync import Synchronizer


class RoutineJob:
    def __init__(
        self,
        storage: Storage,
        updater: Updater,
        synchronizer: Synchronizer,
    ):
        self.storage = storage
        self.updater = updater
        self.sync = synchronizer

    def fetch_and_sync_data(self):
        """
        Recorre todas las entradas activas del storage,
        obtiene el estado actual vía Updater y delega
        la decisión al Synchronizer.
        """
        data = self.storage.get_all()

        for entry_idx, old_entry in data.iterrows():
            entry_id = old_entry.get('id')
            try:
                new_entry = self.updater.fetch(entry_id, old_entry)
                self.sync.sync_entry(entry_id, new_entry)

            except Exception as e:
                # decisión explícita: el job no cae por una URL rota
                print(f"[RoutineJob] Error procesando {entry_id}: {e}")

        # una sola escritura al final
        self.storage.save()

    async def fetch_and_sync_new_listings(self, n_pages=5):
        """Scrapea nuevas páginas y sincroniza con storage"""
        # scrapear nuevas páginas
        new_listings = await self.updater.extract_all_pages(n_pages=n_pages)

        # convertir a dict y mandar al sync
        for listing in new_listings:
            entry = listing.to_dict()
            self.sync.sync_entry(entry_id=entry.get('id'),entry=entry)

        # guardar cambios en storage
        self.storage.save()
        