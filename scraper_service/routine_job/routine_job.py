import asyncio
import logging

import numpy as np

from scraper_service.scraper import BaseScraper
from scraper_service.storage.storage import Storage
from scraper_service.sync.sync import Synchronizer
from scraper_service.updater.samplers import NormalSampler, PoissonSampler, RandomSampler
from scraper_service.updater.updater import Updater

logger = logging.getLogger(__name__)


class RoutineJob:
    def __init__(
        self,
        storage: Storage,
        scraper: BaseScraper,
        synchronizer: Synchronizer,
        updater: Updater | None = None,
    ):
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
        """Actualiza avisos en batches con concurrencia limitada."""
        if max_concurrency < 1:
            raise ValueError("max_concurrency debe ser mayor o igual a 1")
        if self.updater is None:
            raise ValueError("fetch_and_sync_data requiere un Updater")

        data = self.storage.get_all()
        if max_entries is not None:
            data = data.head(max_entries).copy()

        total = len(data)
        totals = {"processed": 0, "closed": 0, "failed": 0}
        entries = iter(enumerate(data.iterrows(), start=1))

        await self.scraper.start()
        try:
            while True:
                batch_size = max(
                    1,
                    int(np.asarray(batch_size_sampler.sample(1)).ravel()[0]),
                )
                batch = []
                for _ in range(batch_size):
                    try:
                        batch.append(next(entries))
                    except StopIteration:
                        break

                if not batch:
                    break

                update_entries = []
                for entry_pos, (entry_idx, row) in batch:
                    entry_id = row.get("id")
                    logger.info(
                        "[RoutineJob] Procesando %s/%s (df_index=%s, entry_id=%s)",
                        entry_pos,
                        total,
                        entry_idx,
                        entry_id,
                    )
                    update_entries.append((entry_id, row))

                batch_result = await self.updater.sync_batch(
                    update_entries,
                    max_concurrency=max_concurrency,
                )
                for metric in totals:
                    totals[metric] += batch_result[metric]

                self.storage.save()
                logger.info(
                    "[RoutineJob] Batch guardado. processed=%s closed=%s failed=%s",
                    totals["processed"],
                    totals["closed"],
                    totals["failed"],
                )

                delay_s = float(
                    np.asarray(batch_delay_sampler.sample(1)).ravel()[0]
                )
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
        finally:
            await self.scraper.close()

        result = {**totals, "total": total}
        logger.info("[RoutineJob] Finalizado: %s", result)
        return result

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
