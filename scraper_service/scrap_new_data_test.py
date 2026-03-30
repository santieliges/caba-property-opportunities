import asyncio
import os
import traceback

from scraper_service.routineJob.routineJob import RoutineJob
from scraper_service.scrapper.ArgenPropScrapper import ArgenPropScrapper
from scraper_service.storage.storage import CSVStorage
from scraper_service.sync.sync import Synchronizer
from scraper_service.updater.updater import Updater


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


async def run_job(url, csv_path, n_pages=5):
    scrapper = ArgenPropScrapper(
        headless=env_flag("HEADLESS", True),
        url_base=url,
        download_images=False,
    )
    storage = CSVStorage(csv_path)
    sync_data = Synchronizer(storage=storage)
    updater = Updater()

    routine_job = RoutineJob(
        storage=storage,
        scrapper=scrapper,
        updater=updater,
        synchronizer=sync_data,
    )
    try:
        await routine_job.fetch_and_sync_new_listings(n_pages=n_pages, delay_s=3, jitter_s=10)
    except Exception:
        print(f"Error running job for {url}")
        traceback.print_exc()


async def main():
    await run_job(
        "https://www.argenprop.com/departamentos/venta/almagro-o-boedo-o-caballito?orden-masnuevos",
        "scraper_service/storage/data/arg_venta_data.csv",
        5,
    )


if __name__ == "__main__":
    asyncio.run(main())
