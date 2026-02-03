from scrapper.AlquilerScrapper import AlquilerScrapper, Scrapper
from scrapper.VentaScrapper import VentaScrapper
from scrapper.ValorDolarScrapper import DolarHoyScrapper
import json
import pandas as pd
from storage.storage import CSVStorage
from sync.sync import Synchronizer
from datacleaner.datacleaner import DataCleaner
import re



# ####################### Datos Venta ########################
# #Inicializo el scrapper de venta y scrappeo n paginas
scraper = Scrapper(headless=True, url_base="https://www.argenprop.com/casas-o-departamentos-o-ph/venta/capital-federal/dolares-desde-20000?orden-masnuevos")
all_data = CSVStorage("storage/data/historic/arg_venta_data.csv").get_all().reset_index().to_dict(orient="records")
all_data = scraper.update_old_data(all_data)
pd.DataFrame(all_data).to_csv("storage/data/historic/arg_venta_data.csv", index=False)
storage_scapped_data = CSVStorage("storage/data/historic/arg_venta_data.csv")

scraper.close()





