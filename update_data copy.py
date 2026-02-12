import asyncio
from storage.CSVStorage import CSVStorage
from sync.sync import Synchronizer
from scrapper.ArgenPropScrapper import ArgenPropScrapper
from updater.updater import Updater
from routineJob.routineJob import RoutineJob
import pandas as pd
import asyncio
import logging

async def run_argenprop_job(
    *,
    csv_path: str,
    url_base: str,
    updater: Updater
):
    storage = CSVStorage(csv_path)
    sync = Synchronizer(storage=storage)
    scrapper = ArgenPropScrapper(
        headless=True,
        url_base=url_base
    )

    job = RoutineJob(
        storage=storage,
        scrapper=scrapper,
        updater=updater,
        synchronizer=sync,
    )

    await job.fetch_and_sync_data()

async def main():

 

    # ───── Inmuebles ─────
    updater = Updater()
    await asyncio.gather(
        run_argenprop_job(
            csv_path="storage/data/historic/arg_alquiler_data_prueba.csv",
            url_base="https://www.argenprop.com/departamentos/alquiler/capital-federal?orden-masnuevos",
            updater=updater,
        )
    )

if __name__ == "__main__":
    asyncio.run(main())