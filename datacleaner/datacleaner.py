from storage.storage import Storage
import pandas as pd
import re
import numpy as np

class DataCleaner:
    def __init__(self, storage: Storage):
        self.storage = storage

    def standardize_column(self, column, func):
        self.storage.apply_to_column(column, func)
        self.storage.save()

    def fill_missing_values(self, column, fill_value):
        self.storage.fillna(column, fill_value)
        self.storage.save()

    def drop_missing_values(self, column):
        self.storage.dropna(column)
        self.storage.save()

    def parse_expensas(self, value):
        if pd.isna(value):
            return None
        
        if not isinstance(value, str):
            return None
        
        match = re.search(r'\$\s*([\d\.]+)', value)
        if not match:
            return None
        
        number = match.group(1).replace('.', '')
        
        try:
            return float(number)
        except ValueError:
            return None
        
    def clean_expensas_column(self):
        self.standardize_column(column="expensas", func=self.parse_expensas)
        self.fill_missing_values(column="expensas", fill_value=0)

    def parse_precio_raw(self, value):
        if pd.isna(value) or not isinstance(value, str):
            return (np.nan, np.nan)

        matches = re.findall(r"(?i)(usd|u\$s|\$)\s*([\d\.]+)", value)
        if not matches:
            return (np.nan, np.nan)

        moneda, numero = matches[0]
        numero = numero.replace('.', '')

        try:
            return moneda.lower(), float(numero)
        except ValueError:
            return (np.nan, np.nan)

    def clean_precio_column(self):
        self.standardize_column(column="precio", func=self.parse_precio_raw)
        self.storage.save()

    def drop_rows_with_missing_precio(self):
        self.drop_missing_values(column="precio")
        self.storage.save()
    
    def parse_antiguedad(self, value):
        if pd.isna(value):
            return None
        
        if not isinstance(value, str):
            return None
        
        value_as_str = value.strip().lower()

        if value_as_str == "a estrenar":
            return 0
        
        match = re.search(r'(\d+)', value_as_str)
        if match:
            return float(match.group(1))
    

    def clean_antiguedad_column(self):
        self.standardize_column(column="antiguedad", func=self.parse_antiguedad)
        self.storage.save()


    def parse_area_m2(self, value):
        if pd.isna(value):
            return None

        if not isinstance(value, str):
            return None

        text = value.lower().strip()

        match = re.search(r'(\d+(?:[\.,]\d+)?)', text)
        if not match:
            return None

        number = match.group(1).replace(',', '.')

        try:
            return float(number)
        except ValueError:
            return None

    def clean_area_m2_column(self):
        self.standardize_column(column="area_m2", func=self.parse_area_m2)
        self.storage.save()

    def ender_cleaning(self, path = None):
        if path is None:
            path = str(self.storage.path).replace("raw", "clean")
        self.storage.save(new_path = path)

    def drop_duplicate_ids(self):
        df = self.storage.get_all()
        df = df[~df.index.duplicated(keep="last")]
        self.storage.df = df
        self.storage.save()