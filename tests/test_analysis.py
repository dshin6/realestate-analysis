import unittest

import pandas as pd

from realestate_analysis.analysis import (
    annual_type_counts,
    annual_type_summary,
    building_summary,
    classify_floor,
    enrich_trades,
    estimate_fair_price,
    floor_average_summary,
    floor_price_index_by_type,
    map_plan_type,
)
from realestate_analysis.config import TARGET_COMPLEX


class AnalysisTest(unittest.TestCase):
    def test_maps_area_to_plan_type(self):
        self.assertEqual(map_plan_type(84.80, TARGET_COMPLEX), "A")
        self.assertEqual(map_plan_type(84.73, TARGET_COMPLEX), "B")
        self.assertEqual(map_plan_type(84.79, TARGET_COMPLEX), "C")
        self.assertEqual(map_plan_type(59.9, TARGET_COMPLEX), "기타")

    def test_floor_groups_use_common_fixed_ranges(self):
        self.assertEqual(classify_floor(1), "1층")
        self.assertEqual(classify_floor(2), "저층 (2층~5층)")
        self.assertEqual(classify_floor(5), "저층 (2층~5층)")
        self.assertEqual(classify_floor(6), "중층 (6층~15층)")
        self.assertEqual(classify_floor(15), "중층 (6층~15층)")
        self.assertEqual(classify_floor(16), "고층 (16층 이상)")
        self.assertEqual(classify_floor(30), "고층 (16층 이상)")

    def test_annual_premium_and_fair_price(self):
        raw = pd.DataFrame(
            [
                {"deal_date": "2025-01-10", "price_won": 600_000_000, "exclusive_area": 84.80, "floor": 10, "building": "231동"},
                {"deal_date": "2025-02-10", "price_won": 500_000_000, "exclusive_area": 84.73, "floor": 10, "building": "231동"},
                {"deal_date": "2026-01-10", "price_won": 660_000_000, "exclusive_area": 84.80, "floor": 10, "building": "231동"},
                {"deal_date": "2026-02-10", "price_won": 550_000_000, "exclusive_area": 84.73, "floor": 10, "building": "231동"},
            ]
        )
        enriched = enrich_trades(raw, TARGET_COMPLEX)
        summary = annual_type_summary(enriched)
        a_2026 = summary[(summary["year"] == 2026) & (summary["plan_type"] == "A")].iloc[0]
        self.assertGreater(a_2026["premium_pct"], 0)

        result = estimate_fair_price(enriched, "A", "중층 (6층~15층)", 700_000_000)
        self.assertIsNotNone(result)
        self.assertEqual(result.count, 2)
        self.assertAlmostEqual(result.asking_delta_pct, 11.1111, places=3)

    def test_volume_and_floor_average_summaries(self):
        raw = pd.DataFrame(
            [
                {"deal_date": "2025-01-10", "price_won": 600_000_000, "exclusive_area": 84.80, "floor": 3, "building": "231동"},
                {"deal_date": "2025-02-10", "price_won": 700_000_000, "exclusive_area": 84.80, "floor": 5, "building": "231동"},
                {"deal_date": "2025-03-10", "price_won": 550_000_000, "exclusive_area": 84.73, "floor": 10, "building": "231동"},
            ]
        )
        enriched = enrich_trades(raw, TARGET_COMPLEX)

        counts = annual_type_counts(enriched)
        a_count = counts[(counts["year"] == 2025) & (counts["plan_type"] == "A")]["trades"].iloc[0]
        self.assertEqual(a_count, 2)

        floors = floor_average_summary(enriched)
        low = floors[
            (floors["floor_group"] == "저층 (2층~5층)")
            & (floors["plan_type"] == "A")
        ].iloc[0]
        self.assertEqual(low["trades"], 2)
        self.assertEqual(low["average_price"], 650_000_000)
        middle_b = floors[
            (floors["floor_group"] == "중층 (6층~15층)")
            & (floors["plan_type"] == "B")
        ].iloc[0]
        self.assertEqual(middle_b["average_price"], 550_000_000)

    def test_floor_price_index_matches_displayed_period_averages(self):
        raw = pd.DataFrame(
            [
                {"deal_date": "2024-01-10", "price_won": 200_000_000, "exclusive_area": 84.80, "floor": 1, "building": "231동"},
                {"deal_date": "2024-02-10", "price_won": 400_000_000, "exclusive_area": 84.80, "floor": 18, "building": "231동"},
                {"deal_date": "2025-01-10", "price_won": 400_000_000, "exclusive_area": 84.80, "floor": 1, "building": "231동"},
                {"deal_date": "2025-02-10", "price_won": 400_000_000, "exclusive_area": 84.80, "floor": 1, "building": "231동"},
                {"deal_date": "2025-03-10", "price_won": 800_000_000, "exclusive_area": 84.80, "floor": 18, "building": "231동"},
                {"deal_date": "2025-04-10", "price_won": 600_000_000, "exclusive_area": 84.73, "floor": 10, "building": "231동"},
            ]
        )
        enriched = enrich_trades(raw, TARGET_COMPLEX)

        result = floor_price_index_by_type(enriched)

        a_first = result[
            (result["plan_type"] == "A") & (result["floor_group"] == "1층")
        ].iloc[0]
        a_high = result[
            (result["plan_type"] == "A") & (result["floor_group"] == "고층 (16층 이상)")
        ].iloc[0]
        expected_pct = ((200_000_000 + 400_000_000 + 400_000_000) / 3) / (
            (400_000_000 + 800_000_000) / 2
        ) * 100
        self.assertAlmostEqual(a_first["price_index_pct"], expected_pct)
        self.assertEqual(a_first["trades"], 3)
        self.assertEqual(a_high["price_index_pct"], 100.0)
        self.assertFalse((result["plan_type"] == "B").any())

    def test_building_summary_uses_average_price(self):
        raw = pd.DataFrame(
            [
                {"deal_date": "2025-01-10", "price_won": 500_000_000, "exclusive_area": 84.80, "floor": 10, "building": "231동"},
                {"deal_date": "2025-02-10", "price_won": 500_000_000, "exclusive_area": 84.80, "floor": 12, "building": "231동"},
                {"deal_date": "2025-03-10", "price_won": 1_000_000_000, "exclusive_area": 84.80, "floor": 15, "building": "231동"},
            ]
        )
        enriched = enrich_trades(raw, TARGET_COMPLEX)

        result = building_summary(enriched).iloc[0]

        self.assertAlmostEqual(result["average_price"], 2_000_000_000 / 3)
        self.assertEqual(result["trades"], 3)
        self.assertNotIn("median_price", result.index)


if __name__ == "__main__":
    unittest.main()
