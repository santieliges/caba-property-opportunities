from scrapper.ArgenPropScrapper import ArgenPropScrapper
from scrapper.AmbitoDolarScrapper import AmbitoDolarScrapper
from routineJob.routineJob import RoutineJob
from updater.updater import Updater
import json
import pandas as pd
from storage.CSVStorage import CSVStorage
from sync.sync import Synchronizer
from datacleaner.datacleaner import DataCleaner
import re
import asyncio
import traceback

async def run_job(url, csv_path, n_pages = 5):
    scrapper = ArgenPropScrapper(
        headless=True,
        url_base=url
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
        await routine_job.fetch_and_sync_new_listings(n_pages=n_pages)
    except Exception as e:

        print(f"Error running job for {url}")
        traceback.print_exc()



async def main():
    await asyncio.gather(
        run_job(
            "https://www.argenprop.com/departamentos/alquiler/capital-federal?orden-masnuevos",
            "storage/data/arg_alquiler_data.csv",
            99
        ),
        run_job(
            "https://www.argenprop.com/departamentos/venta/capital-federal?orden-masnuevos",
            "storage/data/arg_venta_data.csv",
            99   
        )
    )

if __name__ == "__main__":
    asyncio.run(main())
