import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


class DashboardRenderTest(unittest.TestCase):
    def test_chart_builders_keep_raw_points_and_average_bars(self):
        from app import (
            _annual_type_volume_figure,
            _floor_distribution_figure,
            _floor_price_index_figure,
            _trade_scatter_figure,
        )
        from realestate_analysis.analysis import enrich_trades
        from realestate_analysis.config import TARGET_COMPLEX

        raw = pd.DataFrame(
            [
                {"deal_date": "2025-01-10", "price_won": 600_000_000, "exclusive_area": 84.80, "floor": 3, "building": "231동"},
                {"deal_date": "2025-02-10", "price_won": 700_000_000, "exclusive_area": 84.80, "floor": 5, "building": "231동"},
                {"deal_date": "2025-03-10", "price_won": 550_000_000, "exclusive_area": 84.73, "floor": 10, "building": "231동"},
                {"deal_date": "2025-04-10", "price_won": 800_000_000, "exclusive_area": 84.80, "floor": 18, "building": "231동"},
            ]
        )
        trades = enrich_trades(raw, TARGET_COMPLEX)

        trade_figure = _trade_scatter_figure(trades)
        self.assertEqual(sum(len(trace.x) for trace in trade_figure.data), len(trades))
        self.assertTrue(all(trace.type == "scatter" for trace in trade_figure.data))

        volume_figure = _annual_type_volume_figure(trades)
        self.assertTrue(all(trace.type == "bar" for trace in volume_figure.data))

        floor_figure = _floor_distribution_figure(trades)
        bar_points = sum(len(trace.x) for trace in floor_figure.data if trace.type == "bar")
        scatter_points = sum(len(trace.x) for trace in floor_figure.data if trace.type == "scatter")
        self.assertEqual(bar_points, 3)
        self.assertEqual(scatter_points, len(trades))

        index_figure = _floor_price_index_figure(trades)
        self.assertTrue(all(trace.type == "bar" for trace in index_figure.data))
        self.assertEqual(sum(len(trace.x) for trace in index_figure.data), 2)

    def test_prepare_trades_replaces_stale_floor_groups(self):
        from app import _prepare_trades

        stale = pd.DataFrame(
            [
                {
                    "deal_date": "2026-01-10",
                    "price_won": 660_000_000,
                    "exclusive_area": 84.80,
                    "floor": 10,
                    "building": "231동",
                    "floor_group": "중층",
                }
            ]
        )

        prepared = _prepare_trades(stale)

        self.assertEqual(prepared.iloc[0]["floor_group"], "중층 (6층~15층)")

    def test_cached_data_renders_charts_and_table(self):
        project_root = Path(__file__).resolve().parents[1]
        rows = [
            {"deal_date": "2025-01-10", "price_won": 600_000_000, "exclusive_area": 84.80, "floor": 10, "building": "231동"},
            {"deal_date": "2025-02-10", "price_won": 560_000_000, "exclusive_area": 84.73, "floor": 12, "building": "231동"},
            {"deal_date": "2025-03-10", "price_won": 580_000_000, "exclusive_area": 84.79, "floor": 18, "building": "231동"},
            {"deal_date": "2026-01-10", "price_won": 660_000_000, "exclusive_area": 84.80, "floor": 15, "building": "231동"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_path = temp_path / "data" / "cache" / "trades.json"
            cache_path.parent.mkdir(parents=True)
            cache_path.write_text(
                json.dumps({"fetched_at": "2026-07-29T00:00:00+09:00", "end_ym": "202607", "rows": rows}),
                encoding="utf-8",
            )
            previous = Path.cwd()
            try:
                os.chdir(temp_path)
                app = AppTest.from_file(project_root / "app.py").run(timeout=15)
            finally:
                os.chdir(previous)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("plotly_chart")), 5)
        self.assertEqual(len(app.dataframe), 1)

    def test_seed_data_renders_without_runtime_cache(self):
        project_root = Path(__file__).resolve().parents[1]
        rows = [
            {"deal_date": "2025-01-10", "price_won": 600_000_000, "exclusive_area": 84.80, "floor": 10, "building": "231동"},
            {"deal_date": "2025-02-10", "price_won": 560_000_000, "exclusive_area": 84.73, "floor": 12, "building": "231동"},
            {"deal_date": "2025-03-10", "price_won": 580_000_000, "exclusive_area": 84.79, "floor": 18, "building": "231동"},
            {"deal_date": "2026-01-10", "price_won": 660_000_000, "exclusive_area": 84.80, "floor": 15, "building": "231동"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            seed_path = temp_path / "data" / "seed" / "trades.json"
            seed_path.parent.mkdir(parents=True)
            seed_path.write_text(
                json.dumps({"fetched_at": "2026-07-29T00:00:00+09:00", "end_ym": "202607", "rows": rows}),
                encoding="utf-8",
            )
            previous = Path.cwd()
            try:
                os.chdir(temp_path)
                app = AppTest.from_file(project_root / "app.py").run(timeout=15)
                cache_created = (temp_path / "data" / "cache" / "trades.json").exists()
            finally:
                os.chdir(previous)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("plotly_chart")), 5)
        self.assertFalse(cache_created)


if __name__ == "__main__":
    unittest.main()
