import pandas as pd
from storage.storage import Storage
from scrapper.Scrapper import Scrapper
from datetime import datetime
class Synchronizer:
    def __init__(self, storage: Storage, scrapper: Scrapper):
        self.storage = storage
        self.scrapper = scrapper

    def sync_data(self, new_df: pd.DataFrame):
        """
        Sincroniza new_df contra el storage.
        """
        for entry_id, row in new_df.iterrows():
            entry = row.to_dict()

            if not self.storage.exists(entry_id):
                self.storage.insert(
                    valid_from=datetime.now(),
                    entry=entry,
                    entry_id=entry_id
                )
            else:
                old_entry = self.storage.get_by_id(entry_id)
                if self._has_changed(old_entry, entry):
                    self.storage.close(
                        entry_id=entry_id,
                        valid_to=datetime.now()
                    )
                    self.storage.insert(
                        valid_from=datetime.now(),
                        entry=entry,
                        entry_id=entry_id
                    )

        self.storage.save()



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

    def close_endend_urls(self):
        data = self.storage.get_all()
        for entry_id, entry in data.iterrows():
            url = entry.get("url", None)
            if url:
                has_error_410 = self.scrapper.check_url_change(entry_id, entry)
                if has_error_410:
                    self.storage.close(entry_id, valid_to = datetime.now())
        self.storage.save()


    def check_if_null_antiguedad_is_a_estrenar(self):
        data = self.storage.get_all()
        data_with_null_antiguedad = data[ data["antiguedad"].isna() ]
        for entry_id, entry in data_with_null_antiguedad.iterrows():
            url = entry.get("url", None)
            if url:
                is_a_estrenar = self.scrapper.check_if_a_estrenar(entry_id, entry)
                if is_a_estrenar:
                    self.storage.update(
                            entry_id,
                            {"antiguedad": "A Estrenar"}
                        )
        self.storage.save()
