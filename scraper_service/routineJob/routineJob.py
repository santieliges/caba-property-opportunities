import asyncio
import logging
import random

from scrapper import Scrapper
from storage.storage import Storage
from sync.sync import Synchronizer
from updater.updater import Updater

logger = logging.getLogger(__name__)


class RoutineJob:
    def __init__(self, storage: Storage, scrapper: Scrapper, updater: Updater, synchronizer: Synchronizer):
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
        max_concurrency: int = 10,
    ):
        """Actualiza avisos en paralelo usando un semáforo de concurrencia.

        max_concurrency limita las peticiones simultáneas a la API para no saturarla.
        """

        async def _process_one(entry_pos: int, entry_idx: int, entry_id, row, sem: asyncio.Semaphore):
            async with sem:
                try:
                    new_entry = await self.updater.fetch(entry_id, row, argenPropScrapper=self.scrapper)
                    return entry_pos, entry_idx, entry_id, new_entry, None
                except Exception as exc:  # capturamos para no abortar gather completo
                    return entry_pos, entry_idx, entry_id, None, exc

        await self.scrapper.start()
        data = self.storage.get_all()
        if max_entries is not None:
            data = data.head(max_entries).copy()

        sem = asyncio.Semaphore(max_concurrency)

        processed = 0
        failed = 0
        closed = 0

        def chunk_iter(iterable, size):
            it = iter(iterable)
            while True:
                block = []
                for _ in range(size):
                    try:
                        block.append(next(it))
                    except StopIteration:
                        break
                if not block:
                    break
                yield block

        total = len(data)
        entries = list(enumerate(data.iterrows(), start=1))
        for block in chunk_iter(entries, batch_size):
            tasks = []
            for entry_pos, (entry_idx, row) in block:
                entry_id = row.get("id")
                logger.info(
                    "[RoutineJob] Procesando %s de %s entradas (df_index=%s, entry_id=%s).",
                    entry_pos,
                    total,
                    entry_idx,
                    entry_id,
                )
                tasks.append(_process_one(entry_pos, entry_idx, entry_id, row, sem))

            results = await asyncio.gather(*tasks)

            for entry_pos, entry_idx, entry_id, new_entry, exc in results:
                if exc:
                    failed += 1
                    logger.exception(
                        "Error running job for entry_id=%s (pos=%s, df_index=%s)",
                        entry_id,
                        entry_pos,
                        entry_idx,
                    )
                    continue

                if isinstance(new_entry, dict):
                    self.sync.sync_entry(entry_id, new_entry)
                    processed += 1
                elif new_entry == 410:
                    self.sync.sync_entry(entry_id, None)
                    closed += 1
                    processed += 1
                    logger.info("Inmueble cerrado (410) - entry_id=%s", entry_id)
                else:
                    failed += 1
                    logger.warning(
                        "No se pudo acceder a la URL (entry_id=%s), error: %s",
                        entry_id,
                        new_entry,
                    )

            logger.info("[RoutineJob] Guardando batch (%s registros procesados, %s cerrados)", processed, closed)
            self.storage.save()

            if delay_s or jitter_s:
                await asyncio.sleep(delay_s + (random.random() * jitter_s))

        await self.scrapper.close()
        logger.info(
            "[RoutineJob] Finalizado. processed=%s closed=%s failed=%s total=%s",
            processed,
            closed,
            failed,
            total,
        )
        return {"processed": processed, "closed": closed, "failed": failed, "total": total}

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
