from scrapper.Scrapper import ArgenPropScrapper, AmbitoDolarScrapper
from routineJob.routineJob import RoutineJob
from updater.updater import Updater
import json
import pandas as pd
from storage.storage import CSVStorage
from sync.sync import Synchronizer
from datacleaner.datacleaner import DataCleaner
import re
import asyncio

async def run_job(url, csv_path):
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
        await routine_job.fetch_and_sync_new_listings(n_pages=2)
    except Exception as e:
        print(f"Error running job for {url}: {e}")


async def main():
    await asyncio.gather(
        run_job(
            "https://www.argenprop.com/departamentos/alquiler/capital-federal?orden-masnuevos",
            "storage/data/historic/arg_alquiler_data.csv"
        ),
        run_job(
            "https://www.argenprop.com/departamentos/venta/capital-federal?orden-masnuevos",
            "storage/data/historic/arg_venta_data.csv"
        )
    )

if __name__ == "__main__":
    asyncio.run(main())
