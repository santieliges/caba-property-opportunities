from scrapper.ArgenPropScrapper import ArgenPropScrapper
from scrapper.AmbitoDolarScrapper import AmbitoDolarScrapper
from routineJob.routineJob import RoutineJob
from updater.updater import Updater
import json
import pandas as pd
from storage.storage import CSVStorage
from sync.sync import Synchronizer
from datacleaner.datacleaner import DataCleaner
import re
import asyncio
import traceback

async def run_job(url, csv_path, n_pages = 5):
    scrapper = ArgenPropScrapper(
        headless=False,
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
        synchronizer=sync_data
    )
    try:
        await routine_job.fetch_and_sync_new_listings(n_pages=n_pages, delay_s=3, jitter_s=10)
    except Exception as e:

        print(f"Error running job for {url}")
        traceback.print_exc()



async def main():
    # await run_job(
    #     "https://www.argenprop.com/departamentos/alquiler/capital-federal?orden-masnuevos",
    #     "storage/data/arg_alquiler_data.csv",
    #     20
    # )
    await run_job(
        "https://www.argenprop.com/departamentos/venta/chacarita-o-colegiales",
        "storage/data/arg_venta_data.csv",
        20
    )

if __name__ == "__main__":
    asyncio.run(main())
