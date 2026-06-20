import asyncio
from unittest.mock import AsyncMock, Mock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd

from scraper_service.routine_job.routine_job import RoutineJob
from scraper_service.scraper.argenprop_scraper import ArgenPropScraper
from scraper_service.updater.dataSource import ScrappingDataSource
from scraper_service.updater.updater import Updater


class StaticDataSource:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def fetch(self, entry_id, entry):
        self.calls.append((entry_id, entry))
        return self.result


class FixedSampler:
    def __init__(self, value):
        self.value = value

    def sample(self, size=1):
        return np.array([self.value])


class FakeScraper:
    def __init__(self):
        self.started = 0
        self.closed = 0

    async def start(self):
        self.started += 1

    async def close(self):
        self.closed += 1


class FakeStorage:
    def __init__(self):
        self.data = pd.DataFrame(
            [
                {"id": 1, "url": "url-1"},
                {"id": 2, "url": "url-2"},
                {"id": 3, "url": "url-3"},
            ]
        )
        self.save_calls = 0

    def get_all(self):
        return self.data

    def save(self):
        self.save_calls += 1


def test_updater_syncs_one_entry_from_primary_source():
    synchronizer = Mock()
    updated_entry = {"id": 1, "precio": 100}
    source = StaticDataSource((updated_entry, 200))
    updater = Updater(synchronizer=synchronizer, data_source=source)

    result = asyncio.run(updater.sync_data(1, {"id": 1}))

    assert result == updated_entry
    synchronizer.sync_entry.assert_called_once_with(1, updated_entry)


def test_updater_reports_failed_source_and_closes_missing_entries():
    synchronizer = Mock()
    unavailable_source = StaticDataSource((None, 403))
    updater = Updater(synchronizer, unavailable_source)

    result = asyncio.run(updater.sync_data(1, {"id": 1}))

    assert result is None
    assert len(unavailable_source.calls) == 1
    synchronizer.sync_entry.assert_not_called()

    closed_updater = Updater(synchronizer, StaticDataSource((None, 410)))
    assert asyncio.run(closed_updater.sync_data(2, {"id": 2})) == 410
    synchronizer.sync_entry.assert_called_with(2, None)


def test_updater_syncs_batch_and_isolates_entry_failures():
    updater = Mock(spec=Updater)
    updater.sync_data = AsyncMock(
        side_effect=[{"id": 1}, 410, RuntimeError("boom")]
    )
    updater.sync_batch = Updater.sync_batch.__get__(updater, Updater)

    result = asyncio.run(
        updater.sync_batch(
            [(1, {"id": 1}), (2, {"id": 2}), (3, {"id": 3})],
            max_concurrency=2,
        )
    )

    assert result == {"processed": 2, "closed": 1, "failed": 1}
    assert updater.sync_data.await_count == 3


def test_scrapping_data_source_propagates_http_error_status():
    scraper = object.__new__(ArgenPropScraper)
    scraper.detail_page = Mock()
    scraper.detail_page.goto = AsyncMock(return_value=Mock(status=410))
    scraper._looks_like_human_check = AsyncMock(return_value=False)
    source = ScrappingDataSource(scraper)

    result = asyncio.run(source.fetch(2, {"url": "https://example.test/2"}))

    assert result == (None, 410)


def test_routine_job_orchestrates_batches_and_scraper_lifecycle():
    storage = FakeStorage()
    scraper = FakeScraper()
    synchronizer = Mock()
    updater = Mock()
    updater.sync_batch = AsyncMock(
        side_effect=[
            {"processed": 2, "closed": 1, "failed": 0},
            {"processed": 0, "closed": 0, "failed": 1},
        ]
    )
    job = RoutineJob(storage, scraper, synchronizer, updater)

    result = asyncio.run(
        job.fetch_and_sync_data(
            batch_size_sampler=FixedSampler(2),
            batch_delay_sampler=FixedSampler(0),
            max_concurrency=2,
        )
    )

    assert result == {"processed": 2, "closed": 1, "failed": 1, "total": 3}
    assert updater.sync_batch.await_count == 2
    assert [
        call.kwargs["max_concurrency"]
        for call in updater.sync_batch.await_args_list
    ] == [2, 2]
    assert [len(call.args[0]) for call in updater.sync_batch.await_args_list] == [2, 1]
    assert storage.save_calls == 2
    assert scraper.started == 1
    assert scraper.closed == 1
