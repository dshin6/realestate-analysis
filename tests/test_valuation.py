import unittest

import pandas as pd

from realestate_analysis.valuation import build_price_index


def make_balanced_trades() -> pd.DataFrame:
    rows = []
    quarters = [
        ("2025-01-15", 0.80),
        ("2025-04-15", 0.90),
        ("2025-07-15", 1.00),
        ("2025-10-15", 1.20),
    ]
    for date, market in quarters:
        for _ in range(2):
            for plan_type, type_factor in [("A", 1.05), ("B", 1.00)]:
                for building, building_factor in [("231", 1.02), ("232", 0.98)]:
                    for floor, floor_factor in [(3, 0.94), (18, 1.00)]:
                        rows.append(
                            {
                                "deal_date": date,
                                "price_won": (
                                    700_000_000
                                    * market
                                    * type_factor
                                    * building_factor
                                    * floor_factor
                                ),
                                "plan_type": plan_type,
                                "building": building,
                                "floor": floor,
                            }
                        )
    return pd.DataFrame(rows)


class ValuationTest(unittest.TestCase):
    def test_quarterly_index_controls_composition_and_normalizes_latest(self):
        result = build_price_index(make_balanced_trades())

        self.assertIsNotNone(result)
        self.assertEqual(result.latest_quarter, "2025Q4")
        self.assertAlmostEqual(result.series.iloc[-1]["index"], 100.0, places=6)
        self.assertAlmostEqual(
            result.series.iloc[0]["index"],
            100 * 0.80 / 1.20,
            places=4,
        )
        self.assertFalse(result.series["is_sparse"].any())

    def test_index_requires_thirty_valid_trades(self):
        result = build_price_index(make_balanced_trades().iloc[:29])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
