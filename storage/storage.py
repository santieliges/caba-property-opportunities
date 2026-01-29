from abc import ABC, abstractmethod

class Storage(ABC):

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def get_all(self):
        pass

    @abstractmethod
    def get_by_id(self, entry_id):
        pass

    @abstractmethod
    def insert(self, entry: dict):
        pass

    @abstractmethod
    def update(self, entry_id, entry: dict):
        pass

    @abstractmethod
    def delete(self, entry_id):
        pass

    @abstractmethod
    def exists(self, entry_id) -> bool:
        pass

    @abstractmethod
    def close(self, old_entry, valid_to):
        pass
    @abstractmethod
    def apply_to_column(self, column, func):
        pass
    @abstractmethod
    def fillna(self, column, fill_value):
        pass
    @abstractmethod
    def dropna(self, column):
        pass

import pandas as pd
from pathlib import Path

class CSVStorage(Storage):
    def __init__(self, path: str):
        self.path = Path(path)
        self.df = self.load()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def load(self):
        if self.path.exists():
            return pd.read_csv(self.path).set_index("id")
        else:
            df = pd.DataFrame()
            df.index.name = "id"
            return df

    def save(self, new_path: str = None):
        if new_path:
            self.path = Path(new_path)
        self.df.reset_index().to_csv(self.path, index=False)

    def get_all(self) -> pd.DataFrame:
        return self.df.copy()

    def get_by_id(self, entry_id):
        if entry_id in self.df.index:
            return self.df.loc[entry_id]
        return None

    def exists(self, entry_id) -> bool:
        return entry_id in self.df.index

    def insert(self, entry: dict, valid_from, entry_id=None):
        entry = entry.copy()
        entry["valido_desde"] = valid_from
        entry["valido_hasta"] = pd.NaT

        # CASO INICIAL: df sin columnas
        if self.df.empty and len(self.df.columns) == 0:
            self.df = (
                pd.DataFrame([entry])
                .set_index("id")
            )
        else:
            self.df.loc[entry_id, entry.keys()] = list(entry.values())



    def close(self, entry_id, valid_to):

        if "valido_hasta" not in self.df.columns:
            self.df["valido_hasta"] = pd.NaT

        self.df.at[entry_id, "valido_hasta"] = valid_to



    def update(self, entry_id, updates: dict):
        for col, val in updates.items():
            self.df.loc[entry_id, col] = val


    def delete(self, entry_id):
        if entry_id in self.df.index:
            self.df = self.df.drop(entry_id)

    def apply_to_column(self, column, func):
        self.df[column] = self.df[column].apply(func)

    def fillna(self, column, fill_value):
        self.df[column] = self.df[column].fillna(fill_value)

    def dropna(self, column):
        self.df = self.df.dropna(subset=[column])
