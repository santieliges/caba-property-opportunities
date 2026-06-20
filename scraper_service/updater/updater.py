import asyncio
import logging

from scraper_service.sync.sync import Synchronizer
from scraper_service.updater.dataSource import DataSource

logger = logging.getLogger(__name__)


class Updater:
    """Actualiza y sincroniza avisos."""

    def __init__(self, synchronizer: Synchronizer, data_source: DataSource):
        self.synchronizer = synchronizer
        self.data_source = data_source

    async def sync_data(self, entry_id, entry):
        """Obtiene el estado actual de un aviso y lo sincroniza."""
        new_entry, status_code = await self.data_source.fetch(entry_id, entry)

        if status_code == 200 and new_entry is not None:
            self.synchronizer.sync_entry(entry_id, new_entry)
            return new_entry

        if status_code in (404, 410):
            self.synchronizer.sync_entry(entry_id, None)
            return 410

        logger.warning(
            "No se pudo actualizar entry_id=%s (status=%s)", entry_id, status_code
        )
        return None

    async def sync_batch(self, entries, max_concurrency: int = 10):
        """Actualiza un batch sin abortarlo cuando falla una entrada."""
        if max_concurrency < 1:
            raise ValueError("max_concurrency debe ser mayor o igual a 1")

        semaphore = asyncio.Semaphore(max_concurrency)

        async def sync_one(entry_id, entry):
            async with semaphore:
                try:
                    return entry_id, await self.sync_data(entry_id, entry), None
                except Exception as exc:
                    return entry_id, None, exc

        results = await asyncio.gather(
            *(sync_one(entry_id, entry) for entry_id, entry in entries)
        )

        summary = {"processed": 0, "closed": 0, "failed": 0}
        for entry_id, result, exc in results:
            if exc is not None:
                summary["failed"] += 1
                logger.error("Error actualizando entry_id=%s: %s", entry_id, exc)
            elif result == 410:
                summary["processed"] += 1
                summary["closed"] += 1
            elif isinstance(result, dict):
                summary["processed"] += 1
            else:
                summary["failed"] += 1

        return summary
