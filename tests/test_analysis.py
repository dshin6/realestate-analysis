import unittest

import pandas as pd

from realestate_analysis.analysis import (
    annual_type_summary,
    classify_floor,
    enrich_trades,
    estimate_fair_price,
    map_plan_type,
)
from realestate_analysis.config import TARGET_COMPLEX


class AnalysisTest(unittest.TestCase):
    def test_maps_area_to_plan_type(self):
        self.assertEqual(map_plan_type(84.80, TARGET_COMPLEX), "A")
        self.assertEqual(map_plan_type(84.73, TARGET_COMPLEX), "B")
        self.assertEqual(map_plan_type(84.79, TARGET_COMPLEX), "C")
        self.assertEqual(map_plan_type(59.9, TARGET_COMPLEX), "기타")

    def test_floor_groups_use_building_max_floor(self):
        self.assertEqual(classify_floor(1, "235", TARGET_COMPLEX), "1층")
        self.assertEqual(classify_floor(3, "235", TARGET_COMPLEX), "저층")
        self.assertEqual(classify_floor(15, "235", TARGET_COMPLEX), "중층")
        self.assertEqual(classify_floor(27, "235", TARGET_COMPLEX), "고층")
        self.assertEqual(classify_floor(30, "235", TARGET_COMPLEX), "최상층")

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

        result = estimate_fair_price(enriched, "A", "중층", 700_000_000)
        self.assertIsNotNone(result)
        self.assertEqual(result.count, 2)
        self.assertAlmostEqual(result.asking_delta_pct, 11.1111, places=3)


if __name__ == "__main__":
    unittest.main()

