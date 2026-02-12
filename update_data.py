import asyncio
from storage.CSVStorage import CSVStorage
from sync.sync import Synchronizer
from scrapper.ArgenPropScrapper import ArgenPropScrapper
from scrapper.AmbitoDolarScrapper import AmbitoDolarScrapper
from updater.updater import Updater
from routineJob.routineJob import RoutineJob
import pandas as pd
import asyncio
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("update_data_scraper.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

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

    await job.fetch_and_sync_data(batch_size=100)

async def main():

    # ───── Dólar ─────
    dolar_scrapper = AmbitoDolarScrapper(headless=True)
    await dolar_scrapper.start()
    await dolar_scrapper.run()
    valores = dolar_scrapper.get_valores()
    await dolar_scrapper.close()

    rows = [
        {
            "tipo": tipo,
            "operacion": op,
            "valor": data[op]["value"],
            "raw": data[op]["raw"],
            "url": data["url"],
        }
        for tipo, data in valores.items()
        for op in ("compra", "venta")
    ]

    pd.DataFrame(rows).to_csv("dolar_hoy.csv", index=False)

    # ───── Inmuebles ─────
    updater = Updater()
    await asyncio.gather(
        run_argenprop_job(
            csv_path="storage/data/arg_alquiler_data.csv",
            url_base="https://www.argenprop.com",
            updater=updater,
        ),
        run_argenprop_job(
            csv_path="storage/data/arg_venta_data_a_actualizar.csv",
            url_base="https://www.argenprop.com",
            updater=updater,
        )
    )

if __name__ == "__main__":
    asyncio.run(main())