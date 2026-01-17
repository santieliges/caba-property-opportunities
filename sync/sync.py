import pandas as pd
from storage.storage import Storage
from datetime import datetime
class Synchronizer:
    def __init__(self, storage: Storage):
        self.storage = storage

    def sync_data(self, new_data: list[dict]):
        """
        Sincroniza new_data contra el storage.
        """
        for entry in new_data:
            entry_id = entry["id"]

            if not self.storage.exists(entry_id):
                self.storage.insert(valid_from = datetime.now(), entry=entry)
            else:
                old_entry = self.storage.get_by_id(entry_id)
                if self._has_changed(old_entry, entry):
                    self.storage.close(old_entry, valid_to = datetime.now())
                    self.storage.insert(valid_from = datetime.now(), entry=entry)

    def _has_changed(self, old, new: dict) -> bool:
        for k, v in new.items():
            if k == "id":
                continue
            if k not in old.index:
                return True
            if pd.isna(old[k]) and pd.isna(v):
                continue
            if old[k] != v:
                return True
        return False


    def get_entry_by_id(self, entry_id):
        return self.storage.get_by_id(entry_id)

    def existing_ids(self):
        return self.storage.get_all().index \
            if hasattr(self.storage.get_all(), "index") \
            else None
