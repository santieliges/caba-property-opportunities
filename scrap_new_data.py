from scrapper.Scrapper import Scrapper
from routineJob.routineJob import RoutineJob
from updater.updater import Updater
import json
import pandas as pd
from storage.storage import CSVStorage
from sync.sync import Synchronizer
from datacleaner.datacleaner import DataCleaner
import re
import asyncio

async def main():

    ######################## Datos Alquiler ##########################
    scrapper_alq = Scrapper(
        headless=True,
        url_base="https://www.argenprop.com/departamentos/alquiler/capital-federal?orden-masnuevos"
    )
    storage_alq = CSVStorage("storage/data/historic/arg_alquiler_data.csv")
    sync_data_alq = Synchronizer(storage=storage_alq)
    updater_alq = Updater(scrapper=scrapper_alq)

    routine_job_alq = RoutineJob(
        storage=storage_alq,
        updater=updater_alq,
        synchronizer=sync_data_alq
    )

    await routine_job_alq.fetch_and_sync_new_listings(n_pages=2)

    ######################## Datos Venta ##########################
    scrapper_vent = Scrapper(
        headless=True,
        url_base="https://www.argenprop.com/departamentos/venta/capital-federal?orden-masnuevos"
    )
    storage_vent = CSVStorage("storage/data/historic/arg_venta_data.csv")
    sync_data_vent = Synchronizer(storage=storage_vent)
    updater_vent = Updater(scrapper=scrapper_vent)

    routine_job_vent = RoutineJob(
        storage=storage_vent,
        updater=updater_vent,
        synchronizer=sync_data_vent
    )

    await routine_job_vent.fetch_and_sync_new_listings(n_pages=2)


if __name__ == "__main__":
    asyncio.run(main())
