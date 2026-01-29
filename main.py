from scrapper.AlquilerScrapper import AlquilerScrapper
from scrapper.VentaScrapper import VentaScrapper
from scrapper.ValorDolarScrapper import DolarHoyScrapper
import json
import pandas as pd
from storage.storage import CSVStorage
from sync.sync import Synchronizer
from datacleaner.datacleaner import DataCleaner
import re

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
 
 ####################### Scrappeamos la data de Argenprop ##########################

######################## Datos Alquiler ##########################
#Inicializo el scrapper de alquiler y scrappeo n paginas
scraper = AlquilerScrapper(headless=True, url_base="https://www.argenprop.com/departamentos/alquiler/capital-federal?orden-masnuevos")
scraper.run()
all_data = scraper.extract_all_pages(n_pages=99)
pd.DataFrame(all_data).to_csv("storage/data/raw/arg_alquiler_data.csv", index=False)
storage_scapped_data = CSVStorage("storage/data/raw/arg_alquiler_data.csv")
# Limpio los datos scrappeados
datacleaner = DataCleaner(storage_scapped_data)
datacleaner.clean_expensas_column()
datacleaner.clean_antiguedad_column()
datacleaner.clean_precio_column()
datacleaner.clean_area_m2_column()
datacleaner.ender_cleaning()

storage_scapped_data = CSVStorage("storage/data/clean/arg_alquiler_data.csv")

# Sincronizo la data scrappeada con el storage
storage_path = "storage/data/historic/arg_alquiler_data.csv"
storage = CSVStorage(storage_path)


# Sincronizo datos scrapeados con los datos actuales en el storage
synchronizer = Synchronizer(storage, scraper)
synchronizer.sync_data(storage_scapped_data.get_all())


#Cierro los datos de urls que dieron error 410
#synchronizer.close_endend_urls()
scraper.close()

####################### Datos Venta ########################
#Inicializo el scrapper de venta y scrappeo n paginas
scraper = VentaScrapper(headless=True, url_base="https://www.argenprop.com/casas-o-departamentos-o-ph/venta/capital-federal/dolares-desde-20000?orden-masnuevos")
scraper.run()
all_data = scraper.extract_all_pages(n_pages=2)
pd.DataFrame(all_data).to_csv("storage/data/raw/arg_venta_data.csv", index=False)
storage_scapped_data = CSVStorage("storage/data/raw/arg_venta_data.csv")

# Limpio los datos scrappeados
datacleaner = DataCleaner(storage_scapped_data)
datacleaner.clean_expensas_column()
datacleaner.clean_antiguedad_column()
datacleaner.clean_precio_column()
datacleaner.clean_area_m2_column()
datacleaner.ender_cleaning()

storage_scapped_data = CSVStorage("storage/data/clean/arg_venta_data.csv")

# Sincronizo la data scrappeada con el storage
storage_path = "storage/data/historic/arg_venta_data.csv"
storage = CSVStorage(storage_path)

# Sincronizo datos scrapeados con los datos actuales en el storage
synchronizer = Synchronizer(storage, scraper)
synchronizer.sync_data(storage_scapped_data.get_all())
#Cierro los datos de urls que dieron error 410
#synchronizer.close_endend_urls()

scraper.close()





