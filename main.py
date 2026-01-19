from scrapper.AlquilerScrapper import AlquilerScrapper
from scrapper.VentaScrapper import VentaScrapper
from scrapper.ValorDolarScrapper import DolarHoyScrapper
import json
import pandas as pd
from storage.storage import CSVStorage
from sync.sync import Synchronizer

# ### actualizamos valor del dolar usando el dato de ambito financiero 
# dolar_scraper = DolarHoyScrapper()
# dolar_scraper.run()
# valores_dolar = dolar_scraper.get_valores()
# print(valores_dolar)

# df_valores_dolar = pd.DataFrame(
#     list(valores_dolar.items()),
#     columns=["tipo_dolar", "valor"]
# )

# df_valores_dolar["valor"] = (
#     df_valores_dolar["valor"]
#     .str.replace(".", "", regex=False)
#     .str.replace(",", ".", regex=False)
#     .astype(float)
# )

# df_valores_dolar.to_csv("dolar_hoy.csv", index=False)
 
#  ####################### Scrappeamos la data de Argenprop ##########################

# ######################## Datos Alquiler ##########################
# scraper = AlquilerScrapper(headless=True, url_base="https://www.argenprop.com/departamentos/alquiler/capital-federal/")
# scraper.run()
# all_data = scraper.extract_all_pages(n_pages=1)

# storage_path = "storage/arg_alquiler_data.csv"
# storage = CSVStorage(storage_path)
# synchronizer = Synchronizer(storage)
# synchronizer.sync_data(all_data)

# scraper.close()

####################### Datos Venta ##########################

scraper = VentaScrapper(headless=True, url_base="https://www.argenprop.com/casas-o-departamentos-o-ph/venta/capital-federal/dolares-desde-20000?orden-menorprecio")
scraper.run()
all_data = scraper.extract_all_pages(n_pages=99)

storage_path = "storage/arg_venta_data.csv"
storage = CSVStorage(storage_path)
synchronizer = Synchronizer(storage)
synchronizer.sync_data(all_data)

scraper.close()
