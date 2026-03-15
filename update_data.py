import asyncio
from storage.storage import CSVStorage
from sync.sync import Synchronizer
from scrapper.ArgenPropScrapper import ArgenPropScrapper
from scrapper.AmbitoDolarScrapper import AmbitoDolarScrapper
from updater.updater import Updater
from routineJob.routineJob import RoutineJob
import pandas as pd
import logging

logger = logging.getLogger(__name__)

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
    updater: Updater,
    delay_s: float = 0.0,
    jitter_s: float = 0.0,
):
    logger.info("Iniciando job ArgenProp. csv_path=%s", csv_path)
    storage = CSVStorage(csv_path)
    sync = Synchronizer(storage=storage)
    scrapper = ArgenPropScrapper(
        headless=False,
        url_base=url_base,
        download_images=False,
    )

    job = RoutineJob(
        storage=storage,
        scrapper=scrapper,
        updater=updater,
        synchronizer=sync,
    )

    result = await job.fetch_and_sync_data(batch_size=100, delay_s=delay_s, jitter_s=jitter_s)
    logger.info("Job ArgenProp finalizado. csv_path=%s result=%s", csv_path, result)

async def main():

    # # ───── Dólar ─────
    # dolar_scrapper = AmbitoDolarScrapper(headless=False)
    # await dolar_scrapper.start()
    # await dolar_scrapper.run()
    # valores = dolar_scrapper.get_valores()
    # await dolar_scrapper.close()

    # rows = [
    #     {
    #         "tipo": tipo,
    #         "operacion": op,
    #         "valor": data[op]["value"],
    #         "raw": data[op]["raw"],
    #         "url": data["url"],
    #     }
    #     for tipo, data in valores.items()
    #     for op in ("compra", "venta")
    # ]

    # pd.DataFrame(rows).to_csv("dolar_hoy.csv", index=False)

    # ───── Inmuebles ─────
    updater = Updater()
    # await run_argenprop_job(
    #     csv_path="storage/data/arg_alquiler_data.csv",
    #     url_base="https://www.argenprop.com",
    #     updater=updater,
    #     delay_s=0.5,
    #     jitter_s=0.5,
    # )
    await run_argenprop_job(
        csv_path="storage/data/arg_venta_data.csv",
        url_base="https://www.argenprop.com",
        updater=updater,
        delay_s=1,
        jitter_s=3,
    )

if __name__ == "__main__":
    asyncio.run(main())
