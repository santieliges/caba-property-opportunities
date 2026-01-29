from scrapper.AlquilerScrapper import AlquilerScrapper
from scrapper.VentaScrapper import VentaScrapper
from scrapper.ValorDolarScrapper import DolarHoyScrapper
import json
import pandas as pd
from storage.storage import CSVStorage
from sync.sync import Synchronizer
from datacleaner.datacleaner import DataCleaner
import re
from datetime import datetime

######################## Datos Alquiler ##########################

# Sincronizo la data scrappeada con el storage
storage_path = "storage/data/historic/arg_alquiler_data.csv"
storage = CSVStorage(storage_path)
# Limpio los datos scrappeados
datacleaner = DataCleaner(storage)
datacleaner.drop_duplicate_ids()

storage.save()

####################### Datos Venta ########################
#Inicializo el scrapper de venta y scrappeo n paginas


# Sincronizo la data scrappeada con el storage
storage_path = "storage/data/historic/arg_venta_data.csv"
storage = CSVStorage(storage_path)
# Limpio los datos scrappeados
datacleaner = DataCleaner(storage)
datacleaner.drop_duplicate_ids()

storage.save()





