from storage.storage import Storage, CSVStorage
from sync.sync import Synchronizer
from scrapper.Scrapper import Scrapper
from scrapper.ValorDolarScrapper import DolarHoyScrapper
from updater.updater import Updater
from routineJob.routineJob import RoutineJob
import pandas as pd

### actualizamos valor del dolar usando el dato de ambito financiero 
dolar_scraper = DolarHoyScrapper()
dolar_scraper.run()
valores_dolar = dolar_scraper.get_valores()
print(valores_dolar)

df_valores_dolar = pd.DataFrame(
    list(valores_dolar.items()),
    columns=["tipo_dolar", "valor"]
)

df_valores_dolar["valor"] = (
    df_valores_dolar["valor"]
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

df_valores_dolar.to_csv("dolar_hoy.csv", index=False)

### Alquiler
storage_alq_caba = CSVStorage("storage/data/historic/arg_alquiler_data_prueba.csv")
sync_alq_caba = Synchronizer(storage=storage_alq_caba)
scrapper_alq_caba = Scrapper(headless=True, url_base="https://www.argenprop.com/departamentos/alquiler/capital-federal?orden-masnuevos")
updater_alq_caba = Updater(scrapper=scrapper_alq_caba)
routine_job_alq_caba = RoutineJob(storage=storage_alq_caba, updater=updater_alq_caba, synchronizer= sync_alq_caba)

routine_job_alq_caba.fetch_and_sync_data()

### Venta 
storage_vent_caba = CSVStorage("storage/data/historic/arg_venta_data_prueba.csv")
sync_vent_caba = Synchronizer(storage=storage_vent_caba)
scrapper_vent_caba = Scrapper(headless=True, url_base="https://www.argenprop.com/departamentos/venta/capital-federal?orden-masnuevos")
updater_vent_caba = Updater(scrapper=scrapper_vent_caba)
routine_job_vent_caba = RoutineJob(storage=storage_vent_caba, updater=updater_vent_caba, synchronizer= sync_vent_caba)

routine_job_vent_caba.fetch_and_sync_data()