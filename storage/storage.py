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

import pandas as pd
from pathlib import Path

class CSVStorage(Storage):
    def __init__(self, path: str):
        self.path = Path(path)
        self.df = self.load()


    def load(self):
        if self.path.exists():
            return pd.read_csv(self.path).set_index("id")
        else:
            df = pd.DataFrame()
            df.index.name = "id"
            return df


    def save(self):
        self.df.reset_index().to_csv(self.path, index=False)

    def get_all(self) -> pd.DataFrame:
        return self.df.copy()

    def get_by_id(self, entry_id):
        if entry_id in self.df.index:
            return self.df.loc[entry_id]
        return None

    def exists(self, entry_id) -> bool:
        return entry_id in self.df.index

    def insert(self, entry: dict, valid_from):
        entry = entry.copy()
        entry_id = entry["id"]
        entry["valido_desde"] = valid_from

        # CASO INICIAL: df sin columnas
        if self.df.empty and len(self.df.columns) == 0:
            self.df = (
                pd.DataFrame([entry])
                .set_index("id")
            )
        else:
            self.df.loc[entry_id] = entry

        self.save()


    def close(self, old_entry, valid_to):
        entry_id = old_entry.name  

        if "valido_hasta" not in self.df.columns:
            self.df["valido_hasta"] = pd.NaT

        self.df.at[entry_id, "valido_hasta"] = valid_to
        self.save()



    def update(self, entry_id, entry: dict):
        if entry_id not in self.df.index:
            raise KeyError(f"id {entry_id} no existe")
        self.df.loc[entry_id] = entry
        self.save()

    def delete(self, entry_id):
        if entry_id in self.df.index:
            self.df = self.df.drop(entry_id)
            self.save()
