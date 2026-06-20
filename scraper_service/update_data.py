import asyncio
import logging
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scraper_service.routine_job.routine_job import RoutineJob
from scraper_service.scraper.SosivaApiClient import SosivaApiClient, get_aviso_field_value
from scraper_service.scraper.argenprop_scraper import ArgenPropScraper
from scraper_service.storage.storage import CSVStorage
from scraper_service.sync.sync import Synchronizer
from scraper_service.updater.updater import Updater
from scraper_service.updater.samplers import PoissonSampler, NormalSampler
from scraper_service.updater.dataSource import ScrappingDataSource

logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOG_PATH = SCRIPT_DIR / "update_data_scraper.log"


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_missing(value) -> bool:
    if value in (None, "", b""):
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


async def run_argenprop_job(
    *,
    csv_path: str,
    url_base: str,
):
    logger.info("Iniciando job ArgenProp. csv_path=%s", csv_path)
    storage = CSVStorage(csv_path)
    sync = Synchronizer(storage=storage)
    scraper = ArgenPropScraper(
        headless=env_flag("HEADLESS", False),
        url_base=url_base,
        download_images=False,
        use_api_details=False,
    )
    updater = Updater(
        synchronizer=sync,
        data_source=ScrappingDataSource(scraper),
    )

    job = RoutineJob(
        storage=storage,
        scraper=scraper,
        updater=updater,
        synchronizer=sync,
    )

    result = await job.fetch_and_sync_data(
        batch_size_sampler=PoissonSampler(lam=10),
        batch_delay_sampler=NormalSampler(mean=10.0, std=5.0),
    )
    logger.info("Job ArgenProp finalizado. csv_path=%s result=%s", csv_path, result)


async def backfill_active_field_from_aviso(
    *,
    csv_path: str,
    aviso_field: str,
    target_field: str | None = None,
    only_if_missing: bool = True,
    delay_s: float = 0.0,
    max_entries: int | None = None,
):
    target_field = target_field or aviso_field
    storage = CSVStorage(csv_path)
    sosiva_api = SosivaApiClient()
    active_rows = storage.get_all()

    if max_entries is not None:
        active_rows = active_rows.head(max_entries).copy()

    updated = 0
    skipped = 0
    failed = 0

    for _, row in active_rows.iterrows():
        entry_id = row.get("id")
        current_value = row.get(target_field)

        if only_if_missing and not is_missing(current_value):
            skipped += 1
            continue

        try:
            api_res = await asyncio.to_thread(sosiva_api.get_aviso, int(entry_id))
            if api_res.status_code != 200 or not api_res.json_data:
                failed += 1
                logger.warning(
                    "No se pudo obtener aviso para id=%s al backfillear %s (status=%s)",
                    entry_id,
                    aviso_field,
                    api_res.status_code,
                )
                continue

            value = get_aviso_field_value(api_res.json_data, aviso_field)
            if is_missing(value):
                skipped += 1
                continue

            storage.patch_active(entry_id, {target_field: value})
            updated += 1

            if delay_s:
                await asyncio.sleep(delay_s)
        except Exception:
            failed += 1
            logger.exception(
                "Error backfilleando campo %s para id=%s",
                aviso_field,
                entry_id,
            )

    storage.save()
    logger.info(
        "Backfill finalizado. field=%s target=%s updated=%s skipped=%s failed=%s",
        aviso_field,
        target_field,
        updated,
        skipped,
        failed,
    )
    return {"updated": updated, "skipped": skipped, "failed": failed}


async def main():
    await run_argenprop_job(
        csv_path=str(RAW_DATA_DIR / "arg_venta_data.csv"),
        url_base="https://www.argenprop.com",
    )


if __name__ == "__main__":
    asyncio.run(main())
