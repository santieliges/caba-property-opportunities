from abc import ABC, abstractmethod

class Scrapper(ABC):

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def extract_lat_lon(self):
        pass

    @abstractmethod
    def extract_listings_from_page(self, entry_id):
        pass

    @abstractmethod
    def extract_all_pages(self, entry: dict):
        pass

    @abstractmethod
    def download_image(self, entry_id, entry: dict):
        pass

    @abstractmethod
    def save_api_calls(self, entry_id):
        pass

    @abstractmethod
    def scroll_page(self, entry_id) -> bool:
        pass