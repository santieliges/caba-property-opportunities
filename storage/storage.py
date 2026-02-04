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
    def insert(self, entry: dict,  valid_from):
        pass

    @abstractmethod
    def update(self, entry_id, entry: dict, valid_from):
        pass

    @abstractmethod
    def delete(self, entry_id, valid_to):
        pass

    @abstractmethod
    def exists(self, entry_id) -> bool:
        pass

    @abstractmethod
    def close(self, entry_id, valid_to):
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
