from scrapper import Scrapper
import json
import pandas as pd

scraper = Scrapper(headless=True)
scraper.run()
all_data = scraper.extract_all_pages(n_pages=99)

df = pd.DataFrame(all_data)
df.to_csv("datos_inmuebles.csv", index=False)

print(df.head())

scraper.close()

