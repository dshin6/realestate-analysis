import unittest

import pandas as pd

from realestate_analysis.valuation import (
    BacktestResult,
    backtest_valuation,
    build_price_index,
    evaluate_asking_price,
)


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


def make_valuation_trades() -> pd.DataFrame:
    frame = make_balanced_trades()
    extra = frame.copy()
    extra["deal_date"] = "2026-01-15"
    extra["price_won"] = extra["price_won"] * 1.10 / 1.20
    return pd.concat([frame, extra], ignore_index=True)


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

    def test_sparse_latest_quarter_is_not_used_as_reference(self):
        trades = make_balanced_trades()
        sparse_latest = trades.iloc[[0]].copy()
        sparse_latest["deal_date"] = "2026-01-15"
        sparse_latest["price_won"] = sparse_latest["price_won"] * 2
        trades = pd.concat([trades, sparse_latest], ignore_index=True)

        result = build_price_index(trades)

        self.assertEqual(result.latest_quarter, "2025Q4")
        stable_row = result.series.loc[
            result.series["quarter"] == "2025Q4"
        ].iloc[0]
        sparse_row = result.series.loc[
            result.series["quarter"] == "2026Q1"
        ].iloc[0]
        self.assertAlmostEqual(stable_row["index"], 100.0, places=6)
        self.assertTrue(bool(sparse_row["is_sparse"]))

    def test_index_requires_thirty_valid_trades(self):
        result = build_price_index(make_balanced_trades().iloc[:29])
        self.assertIsNone(result)

    def test_asking_uses_same_building_and_actual_floor_adjustment(self):
        trades = make_valuation_trades()
        result = evaluate_asking_price(
            trades,
            plan_type="A",
            building="231",
            floor=18,
            asking_price_won=820_000_000,
            backtest=BacktestResult(
                median_absolute_error_won=25_000_000,
                median_absolute_percentage_error_pct=4.0,
                interval_coverage_pct=70.0,
                cases=20,
            ),
        )

        self.assertIsNotNone(result)
        self.assertFalse(result.expanded_to_complex)
        self.assertGreaterEqual(result.count, 8)
        self.assertEqual(result.confidence, "높음")
        self.assertIn(
            result.status,
            {"저평가 가능", "적정 범위", "다소 높음", "높음"},
        )
        self.assertTrue(
            {"adjusted_price_won", "weight"}.issubset(
                result.comparables.columns
            )
        )

    def test_asking_expands_to_complex_when_building_sample_is_small(self):
        trades = make_valuation_trades()
        trades.loc[trades.index[2:], "building"] = "232"

        result = evaluate_asking_price(
            trades,
            plan_type="A",
            building="231",
            floor=18,
            asking_price_won=820_000_000,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.expanded_to_complex)

    def test_asking_returns_insufficient_when_interval_is_too_wide(self):
        trades = make_valuation_trades()
        trades.loc[trades.index % 16 < 8, "price_won"] *= 0.60
        trades.loc[trades.index % 16 >= 8, "price_won"] *= 1.40

        result = evaluate_asking_price(
            trades,
            plan_type="A",
            building="231",
            floor=18,
            asking_price_won=820_000_000,
        )

        self.assertEqual(result.status, "판단 자료 부족")

    def test_asking_returns_insufficient_with_fewer_than_eight_comparables(self):
        trades = make_valuation_trades()
        trades["plan_type"] = "B"
        trades.loc[trades.index[:7], "plan_type"] = "A"

        result = evaluate_asking_price(
            trades,
            plan_type="A",
            building="231",
            floor=18,
            asking_price_won=820_000_000,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.count, 7)
        self.assertEqual(result.status, "판단 자료 부족")

    def test_backtest_reports_time_ordered_error_metrics(self):
        result = backtest_valuation(make_valuation_trades(), max_quarters=2)

        self.assertIsNotNone(result)
        self.assertGreater(result.cases, 0)
        self.assertGreaterEqual(result.median_absolute_error_won, 0)
        self.assertGreaterEqual(
            result.median_absolute_percentage_error_pct,
            0.0,
        )
        self.assertGreaterEqual(result.interval_coverage_pct, 0.0)
        self.assertLessEqual(result.interval_coverage_pct, 100.0)


if __name__ == "__main__":
    unittest.main()
