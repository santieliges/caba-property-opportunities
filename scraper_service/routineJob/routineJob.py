import asyncio
import logging
import random

from scraper_service.storage.storage import Storage
from scraper_service.sync.sync import Synchronizer
from scraper_service.updater.updater import Updater

logger = logging.getLogger(__name__)


class RoutineJob:
    def __init__(self, storage: Storage, scrapper, updater: Updater, synchronizer: Synchronizer):
        self.storage = storage
        self.scrapper = scrapper
        self.updater = updater
        self.sync = synchronizer

    async def fetch_and_sync_data(
        self,
        batch_size: int = 1000,
        delay_s: float = 0.0,
        jitter_s: float = 0.0,
        max_entries: int | None = None,
    ):
        await self.scrapper.start()
        data = self.storage.get_all()
        if max_entries is not None:
            data = data.head(max_entries).copy()

        processed = 0
        failed = 0

        for i, (entry_idx, old_entry) in enumerate(data.iterrows(), start=1):
            logger.info("[RoutineJob] Procesando %s de %s entradas.", i, len(data))
            entry_id = old_entry.get("id")

            try:
                new_entry = await self.updater.fetch(
                    entry_id,
                    old_entry,
                    argenPropScrapper=self.scrapper,
                )
                if isinstance(new_entry, dict):
                    self.sync.sync_entry(entry_id, new_entry)
                    processed += 1
                elif new_entry == 410:
                    self.sync.sync_entry(entry_id, None)
                    processed += 1
                else:
                    failed += 1
                    logger.warning(
                        "No se pudo acceder a la URL (entry_id=%s), error: %s",
                        entry_id,
                        new_entry,
                    )
                    continue

                if processed % batch_size == 0:
                    logger.info("[RoutineJob] Guardando batch (%s registros procesados)", processed)
                    self.storage.save()

            except Exception:
                failed += 1
                logger.exception("Error running job for entry_id=%s", entry_id)

            finally:
                if delay_s or jitter_s:
                    await asyncio.sleep(delay_s + (random.random() * jitter_s))

        if processed % batch_size != 0:
            logger.info("[RoutineJob] Guardado final (%s registros procesados)", processed)
            self.storage.save()

        await self.scrapper.close()
        logger.info(
            "[RoutineJob] Finalizado. processed=%s failed=%s total=%s",
            processed,
            failed,
            len(data),
        )
        return {"processed": processed, "failed": failed, "total": len(data)}

    async def fetch_and_sync_new_listings(
        self,
        n_pages: int = 5,
        delay_s: float = 0.0,
        jitter_s: float = 0.0,
    ):
        await self.scrapper.start()
        try:
            new_listings = await self.scrapper.extract_all_pages(
                n_pages=n_pages,
                delay_s=delay_s,
                jitter_s=jitter_s,
            )
        finally:
            await self.scrapper.close()

        for listing in new_listings:
            entry = listing.to_dict()
            self.sync.sync_entry(entry_id=entry.get("id"), entry=entry)

        self.storage.save()
