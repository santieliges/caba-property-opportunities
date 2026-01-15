from scrapper import Scrapper
from valorDolarHoy import DolarHoyScrapper
import json
import pandas as pd


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

scraper = Scrapper(headless=True)
scraper.run()
all_data = scraper.extract_all_pages(n_pages=99)

df = pd.DataFrame(all_data)
df.to_csv("datos_inmuebles.csv", index=False)

print(df.head())

scraper.close()

