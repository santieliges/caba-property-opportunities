import asyncio
import logging
import random
import numpy as np
from scraper_service.scraper import BaseScraper
from scraper_service.storage.storage import Storage
from scraper_service.sync.sync import Synchronizer
from scraper_service.updater.updater import Updater
from scraper_service.updater.samplers import RandomSampler, PoissonSampler, NormalSampler

logger = logging.getLogger(__name__)



class RoutineJob:
    def __init__(self, storage: Storage, scraper: BaseScraper, updater: Updater, synchronizer: Synchronizer):
        self.storage = storage
        self.scraper = scraper
        self.updater = updater
        self.sync = synchronizer

    async def fetch_and_sync_data(
        self,
        batch_size_sampler: RandomSampler = PoissonSampler(lam=300),
        batch_delay_sampler: RandomSampler = NormalSampler(mean=35.0, std=5.0),
        max_entries: int | None = None,
        max_concurrency: int = 10,
    ):
        """Actualiza avisos en paralelo usando un semáforo de concurrencia.

        max_concurrency limita las peticiones simultáneas a la API para no saturarla.
        """

        async def _process_one(entry_pos: int, entry_idx: int, entry_id, row, sem: asyncio.Semaphore):
            async with sem:
                try:
                    new_entry = await self.updater.fetch(entry_id, row, argenprop_scraper=self.scraper)
                    return entry_pos, entry_idx, entry_id, new_entry, None
                except Exception as exc:  # capturamos para no abortar gather completo
                    return entry_pos, entry_idx, entry_id, None, exc

        await self.scraper.start()
        data = self.storage.get_all()
        if max_entries is not None:
            data = data.head(max_entries).copy()

        sem = asyncio.Semaphore(max_concurrency)

        processed = 0
        failed = 0
        closed = 0

        total = len(data)
        entries = list(enumerate(data.iterrows(), start=1))
        entries_it = iter(entries)

        while True:
            # muestrea el tamaño del batch (devuelve array de longitud 1)
            bs_arr = batch_size_sampler.sample(1)
            batch_size = int(np.asarray(bs_arr).ravel()[0])
            if batch_size < 1:
                batch_size = 1

            # construir bloque dinámico
            block = []
            for _ in range(batch_size):
                try:
                    block.append(next(entries_it))
                except StopIteration:
                    break
            if not block:
                break

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

            # muestrea delay entre batches
            delay_arr = batch_delay_sampler.sample(1)
            delay_s = float(np.asarray(delay_arr).ravel()[0])
            if delay_s > 0:
                await asyncio.sleep(delay_s)

        await self.scraper.close()
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
        max_existing_hits: int | None = None,
    ):
        await self.scraper.start()
        try:
            existing_ids = {
                int(row["id"])
                for _, row in self.storage.get_all().iterrows()
                if row.get("id") is not None
            }
            new_listings = await self.scraper.extract_all_pages(
                n_pages=n_pages,
                delay_s=delay_s,
                jitter_s=jitter_s,
                existing_ids=existing_ids if max_existing_hits else None,
                max_existing_hits=max_existing_hits,
            )
        finally:
            await self.scraper.close()

        for listing in new_listings:
            entry = listing.to_dict()
            self.sync.sync_entry(entry_id=entry.get("id"), entry=entry)

        self.storage.save()
