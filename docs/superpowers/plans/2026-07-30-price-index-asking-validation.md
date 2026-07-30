# Price Index and Asking Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a composition-adjusted quarterly transaction price index and an evidence-backed asking-price assessment using user-entered type, building, actual floor, and asking price.

**Architecture:** Keep the existing descriptive summaries in `realestate_analysis/analysis.py` and add a focused `realestate_analysis/valuation.py` module for hedonic index fitting, comparable adjustment, asking classification, and time-ordered backtesting. `app.py` consumes public dataclasses from that module, renders one index chart, and replaces the current floor-group median comparison with an actual-floor valuation card.

**Tech Stack:** Python 3.12, NumPy 2.x, pandas 2.3+, Plotly 6.x, Streamlit 1.59+, `unittest`, Streamlit `AppTest`

## Global Constraints

- Use only the existing dependencies in `requirements.txt`; do not add scikit-learn or statsmodels.
- Use quarterly time effects because the dataset averages only a few trades per month.
- Normalize the most recent quarter with transactions to index `100.0`.
- Require at least 30 valid trades to fit a price index.
- Mark quarters with fewer than 3 trades as sparse.
- Apply building-coefficient ridge strength `5.0`; do not penalize quarter, type, or floor terms.
- Use actual-floor spline terms `floor`, `max(floor - 5, 0)`, and `max(floor - 15, 0)`.
- Use same-building comparables only when at least 8 exist; otherwise expand to all same-type complex trades.
- Adjust no more than the 30 highest-weight comparables.
- Use a two-year recency half-life.
- Use weighted 20th, 50th, and 80th percentiles for low, fair, and high prices.
- Return `판단 자료 부족` when fewer than 8 comparables remain or the interval width exceeds 20% of fair price.
- Do not scrape or automatically collect asking prices; the user enters the asking price.
- Preserve the current Apple-style responsive UI and validate at 360×800 and 1280×900.
- Follow red-green-refactor for every production behavior.

---

## File Structure

- Create `realestate_analysis/valuation.py`: hedonic model, quarterly index, comparable adjustment, asking classification, confidence, and backtest.
- Create `tests/test_valuation.py`: deterministic synthetic-data tests for every valuation rule.
- Modify `app.py`: index chart, summary cards, exact-floor asking form, valuation explanation, comparable table.
- Modify `tests/test_app.py`: chart-builder and Streamlit rendering expectations.
- Modify `README.md`: describe the index and manual asking-price workflow.
- Modify `memory-bank/implementation-plan.md`: record the implementation tasks and current state.
- Modify `memory-bank/progress.md`: record verified results only after all checks pass.
- Modify `memory-bank/architecture.md`: add the new valuation component and data flow.

---

### Task 1: Hedonic Quarterly Price Index

**Files:**
- Create: `realestate_analysis/valuation.py`
- Create: `tests/test_valuation.py`

**Interfaces:**
- Consumes: enriched trade frame with `deal_date`, `price_won`, `plan_type`, `building`, and `floor`.
- Produces: `build_price_index(frame: pd.DataFrame, min_observations: int = 30) -> PriceIndexResult | None`.
- Produces: `PriceIndexResult.series` with `quarter`, `index`, `trades`, and `is_sparse`.
- Produces: `_HedonicModel.predict_log_price(quarter, plan_type, building, floor) -> float`.

- [ ] **Step 1: Write a failing index test**

Create `tests/test_valuation.py` with a balanced synthetic dataset so type, building, floor, and time effects are identifiable:

```python
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
        for repeat in range(2):
            for plan_type, type_factor in [("A", 1.05), ("B", 1.00)]:
                for building, building_factor in [("231", 1.02), ("232", 0.98)]:
                    for floor, floor_factor in [(3, 0.94), (18, 1.00)]:
                        rows.append(
                            {
                                "deal_date": date,
                                "price_won": 700_000_000
                                * market
                                * type_factor
                                * building_factor
                                * floor_factor,
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
        self.assertAlmostEqual(result.series.iloc[0]["index"], 100 * 0.80 / 1.20, places=4)
        self.assertFalse(result.series["is_sparse"].any())

    def test_index_requires_thirty_valid_trades(self):
        result = build_price_index(make_balanced_trades().iloc[:29])
        self.assertIsNone(result)
```

- [ ] **Step 2: Run the index tests and verify RED**

Run:

```bash
python -m unittest tests.test_valuation.ValuationTest.test_quarterly_index_controls_composition_and_normalizes_latest tests.test_valuation.ValuationTest.test_index_requires_thirty_valid_trades -v
```

Expected: ERROR with `ModuleNotFoundError: No module named 'realestate_analysis.valuation'`.

- [ ] **Step 3: Implement the model and index**

Create `realestate_analysis/valuation.py` with these public result types and helpers:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MIN_INDEX_OBSERVATIONS = 30
SPARSE_QUARTER_TRADES = 3
BUILDING_RIDGE_STRENGTH = 5.0
UNKNOWN_BUILDING = "미공개"


@dataclass(frozen=True)
class _HedonicModel:
    coefficients: np.ndarray
    quarter_levels: tuple[str, ...]
    type_levels: tuple[str, ...]
    building_levels: tuple[str, ...]
    reference_type: str
    reference_building: str

    def predict_log_price(
        self,
        quarter: str,
        plan_type: str,
        building: str,
        floor: int,
    ) -> float:
        row = _design_row(
            quarter,
            plan_type,
            building,
            floor,
            self.quarter_levels,
            self.type_levels,
            self.building_levels,
            self.reference_type,
            self.reference_building,
        )
        return float(row @ self.coefficients)


@dataclass(frozen=True)
class PriceIndexResult:
    series: pd.DataFrame
    latest_quarter: str
    latest_index: float
    quarterly_change_pct: float | None
    yearly_change_pct: float | None
    model: _HedonicModel
```

Implement `_clean_frame`, `_floor_terms`, `_design_row`, `_fit_model`, and `build_price_index`:

```python
def _floor_terms(floor: int | float) -> tuple[float, float, float]:
    value = float(floor)
    return value, max(value - 5.0, 0.0), max(value - 15.0, 0.0)


def build_price_index(
    frame: pd.DataFrame,
    min_observations: int = MIN_INDEX_OBSERVATIONS,
) -> PriceIndexResult | None:
    work = _clean_frame(frame)
    if len(work) < min_observations:
        return None
    model = _fit_model(work)
    reference_floor = int(round(work["floor"].median()))
    reference_type = model.reference_type
    reference_building = model.reference_building
    rows = []
    for quarter in model.quarter_levels:
        rows.append(
            {
                "quarter": quarter,
                "level": np.exp(
                    model.predict_log_price(
                        quarter,
                        reference_type,
                        reference_building,
                        reference_floor,
                    )
                ),
                "trades": int((work["quarter"] == quarter).sum()),
            }
        )
    series = pd.DataFrame(rows)
    latest_level = float(series.iloc[-1]["level"])
    series["index"] = series["level"] / latest_level * 100.0
    series["is_sparse"] = series["trades"] < SPARSE_QUARTER_TRADES
    index_by_quarter = series.set_index("quarter")["index"]
    latest_period = pd.Period(str(series.iloc[-1]["quarter"]), freq="Q")
    previous_value = index_by_quarter.get(str(latest_period - 1))
    year_ago_value = index_by_quarter.get(str(latest_period - 4))
    quarterly_change = (
        float(100.0 / previous_value - 1) * 100
        if previous_value is not None
        else None
    )
    yearly_change = (
        float(100.0 / year_ago_value - 1) * 100
        if year_ago_value is not None
        else None
    )
    return PriceIndexResult(
        series=series[["quarter", "index", "trades", "is_sparse"]],
        latest_quarter=str(series.iloc[-1]["quarter"]),
        latest_index=100.0,
        quarterly_change_pct=quarterly_change,
        yearly_change_pct=yearly_change,
        model=model,
    )
```

Construct a NumPy design matrix with intercept, quarter dummies, type dummies, building dummies, and the three floor terms. Use `np.linalg.pinv(X.T @ X + penalty) @ X.T @ y`; set only building dummy diagonal entries in `penalty` to `5.0`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_valuation.ValuationTest.test_quarterly_index_controls_composition_and_normalizes_latest tests.test_valuation.ValuationTest.test_index_requires_thirty_valid_trades -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit the index**

```bash
git add realestate_analysis/valuation.py tests/test_valuation.py
git commit -m "feat: add adjusted transaction price index"
```

---

### Task 2: Comparable Adjustment and Asking Classification

**Files:**
- Modify: `realestate_analysis/valuation.py`
- Modify: `tests/test_valuation.py`

**Interfaces:**
- Consumes: `PriceIndexResult` from Task 1.
- Produces: `evaluate_asking_price(frame, plan_type, building, floor, asking_price_won, price_index=None, backtest=None) -> AskingPriceResult`.
- Produces: `backtest_valuation(frame, max_quarters: int = 8) -> BacktestResult | None`.
- Produces: a comparable frame containing `deal_date`, `building`, `floor`, `price_won`, `adjusted_price_won`, and `weight`.

- [ ] **Step 1: Write failing comparable and classification tests**

Append tests that use `make_balanced_trades()` repeated across additional dates:

```python
from realestate_analysis.valuation import (
    BacktestResult,
    evaluate_asking_price,
)


def make_valuation_trades() -> pd.DataFrame:
    frame = make_balanced_trades()
    extra = frame.copy()
    extra["deal_date"] = "2026-01-15"
    extra["price_won"] = extra["price_won"] * 1.10 / 1.20
    return pd.concat([frame, extra], ignore_index=True)


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
        {"adjusted_price_won", "weight"}.issubset(result.comparables.columns)
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
    trades.loc[trades.index % 2 == 0, "price_won"] *= 0.60
    trades.loc[trades.index % 2 == 1, "price_won"] *= 1.40

    result = evaluate_asking_price(
        trades,
        plan_type="A",
        building="231",
        floor=18,
        asking_price_won=820_000_000,
    )

    self.assertEqual(result.status, "판단 자료 부족")
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
python -m unittest tests.test_valuation -v
```

Expected: import errors for `BacktestResult` and `evaluate_asking_price`.

- [ ] **Step 3: Implement weighted comparables and classifications**

Add constants and result dataclasses:

```python
MIN_COMPARABLES = 8
HIGH_CONFIDENCE_COMPARABLES = 20
MEDIUM_CONFIDENCE_COMPARABLES = 10
MAX_COMPARABLES = 30
RECENCY_HALF_LIFE_DAYS = 365.25 * 2
MAX_INTERVAL_WIDTH_RATIO = 0.20


@dataclass(frozen=True)
class BacktestResult:
    median_absolute_error_won: int
    median_absolute_percentage_error_pct: float
    interval_coverage_pct: float
    cases: int


@dataclass(frozen=True)
class AskingPriceResult:
    count: int
    fair_price_won: int | None
    low_price_won: int | None
    high_price_won: int | None
    asking_delta_pct: float | None
    status: str
    confidence: str
    expanded_to_complex: bool
    same_building_ratio: float
    comparables: pd.DataFrame
    backtest: BacktestResult | None
```

Use this adjustment formula for every comparable:

```python
source_log = model.predict_log_price(
    source_quarter,
    plan_type,
    source_building,
    source_floor,
)
target_log = model.predict_log_price(
    price_index.latest_quarter,
    plan_type,
    target_building,
    target_floor,
)
adjusted_price = source_price * np.exp(target_log - source_log)
weight = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
```

Implement `_weighted_quantile` by sorting values, accumulating normalized positive weights, and using `np.interp`. Classification order must be:

```python
if count < MIN_COMPARABLES or (high - low) / fair > MAX_INTERVAL_WIDTH_RATIO:
    status = "판단 자료 부족"
elif asking_price_won < low:
    status = "저평가 가능"
elif asking_price_won <= high:
    status = "적정 범위"
elif asking_price_won <= high * 1.05:
    status = "다소 높음"
else:
    status = "높음"
```

Confidence order must be:

```python
if (
    count >= 20
    and same_building_ratio >= 0.50
    and backtest is not None
    and backtest.median_absolute_percentage_error_pct <= 7.5
):
    confidence = "높음"
elif (
    count >= 10
    and backtest is not None
    and backtest.median_absolute_percentage_error_pct <= 12.5
):
    confidence = "보통"
else:
    confidence = "낮음"
```

Implement `backtest_valuation` with at most the latest eight eligible quarters. For each validation quarter, fit only on earlier trades, evaluate each transaction using its actual type/building/floor as a pseudo-listing, and aggregate absolute error, absolute percentage error, and 20–80% interval coverage. Do not let validation-quarter or future trades enter the training frame.

- [ ] **Step 4: Run valuation tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_valuation -v
```

Expected: all valuation tests pass.

- [ ] **Step 5: Commit the asking evaluator**

```bash
git add realestate_analysis/valuation.py tests/test_valuation.py
git commit -m "feat: evaluate asking prices with adjusted comparables"
```

---

### Task 3: Price Index and Asking UI

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

**Interfaces:**
- Consumes: `PriceIndexResult`, `AskingPriceResult`, `build_price_index`, `evaluate_asking_price`, and `backtest_valuation`.
- Produces: `_price_index_figure(result: PriceIndexResult) -> go.Figure`.
- Produces: a Streamlit section titled `실거래 가격지수` and an expander titled `검토 중인 매물 호가 분석`.

- [ ] **Step 1: Write failing chart and render tests**

Extend `test_chart_builders_keep_raw_points_and_average_bars`:

```python
from app import _price_index_figure
from realestate_analysis.valuation import build_price_index
from tests.test_valuation import make_balanced_trades

index_result = build_price_index(make_balanced_trades())
index_figure = _price_index_figure(index_result)
self.assertTrue(all(trace.type == "scatter" for trace in index_figure.data))
self.assertEqual(index_figure.layout.yaxis.title.text, "실거래 가격지수")
```

Add a Streamlit render test with at least 32 balanced rows and assert:

```python
self.assertEqual(len(app.exception), 0)
self.assertIn("실거래 가격지수", [item.value for item in app.subheader])
self.assertIn("매물 동", [item.label for item in app.selectbox])
self.assertIn("매물 실제 층", [item.label for item in app.number_input])
self.assertIn("매물 호가(억원)", [item.label for item in app.number_input])
```

Update expected Plotly counts in existing cached/seed tests from `5` to `6` only for fixtures with at least 30 valid trades. Keep small-fixture tests at `5` and assert that an index-insufficient info message appears.

- [ ] **Step 2: Run app tests and verify RED**

Run:

```bash
python -m unittest tests.test_app -v
```

Expected: import failure for `_price_index_figure` and missing UI labels.

- [ ] **Step 3: Implement the index chart**

Add valuation imports and a chart builder:

```python
from realestate_analysis.valuation import (
    AskingPriceResult,
    PriceIndexResult,
    backtest_valuation,
    build_price_index,
    evaluate_asking_price,
)


def _price_index_figure(result: PriceIndexResult) -> go.Figure:
    data = result.series.copy()
    data["표본"] = data["is_sparse"].map({True: "표본 적음", False: "일반"})
    figure = px.line(
        data,
        x="quarter",
        y="index",
        markers=True,
        labels={"quarter": "거래 분기", "index": "실거래 가격지수"},
        hover_data={"trades": True, "is_sparse": False, "표본": True},
    )
    figure.update_traces(line={"color": "#0071E3", "width": 3}, marker={"size": 7})
    figure.add_hline(y=100, line_dash="dot", line_color="rgba(29,29,31,0.35)")
    figure.update_layout(yaxis={"title": "실거래 가격지수"})
    return _polish_chart(figure, height=380)
```

Render the index from the full `trades` frame, not the user-filtered frame. Place it after the top four metrics and before raw transaction charts. Show metrics for latest index, quarter change, year change, and latest-quarter trades. If fewer than 30 trades are valid, show `실거래 가격지수를 계산하려면 유효 거래가 30건 이상 필요합니다.`

- [ ] **Step 4: Replace the old asking form**

Replace the floor-group form with:

```python
with st.expander("검토 중인 매물 호가 분석", expanded=True):
    input_a, input_b, input_c, input_d = st.columns(4)
    plan_type = input_a.selectbox("매물 타입", ["A", "B", "C"])
    building = input_b.selectbox(
        "매물 동",
        sorted(value for value in trades["building"].unique() if value != "미공개"),
    )
    floor = input_c.number_input(
        "매물 실제 층",
        min_value=1,
        max_value=int(trades["floor"].max()),
        value=min(10, int(trades["floor"].max())),
        step=1,
    )
    asking_eok = input_d.number_input(
        "매물 호가(억원)",
        min_value=0.0,
        value=0.0,
        step=0.1,
    )
    st.text_input("매물 메모 또는 링크(선택)")
```

Build the backtest once per full trade frame using `@st.cache_data(show_spinner=False)`. When an asking price is entered, render:

- fair, low–high, asking difference, status, and confidence metrics;
- an expansion notice when `expanded_to_complex` is true;
- backtest MAPE and case count;
- the highest-weight comparable rows with original and adjusted prices;
- `판단 자료 부족` guidance without positive/negative pricing claims.

Do not persist the optional memo/link.

- [ ] **Step 5: Run app tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_app -v
```

Expected: all app tests pass with no Streamlit exceptions.

- [ ] **Step 6: Commit the UI**

```bash
git add app.py tests/test_app.py
git commit -m "feat: show price index and asking validation"
```

---

### Task 4: Documentation, Full Verification, and Visual QA

**Files:**
- Modify: `README.md`
- Modify: `memory-bank/implementation-plan.md`
- Modify: `memory-bank/progress.md`
- Modify: `memory-bank/architecture.md`

**Interfaces:**
- Consumes: verified implementation from Tasks 1–3.
- Produces: user-facing usage instructions and project records matching actual behavior.

- [ ] **Step 1: Update documentation**

Document these exact facts:

- The quarterly index controls for type, building, and actual floor and normalizes the latest transacted quarter to 100.
- Asking price is manually entered; it is not scraped.
- The fair-price range is based on index-, building-, and floor-adjusted comparables.
- `판단 자료 부족` is possible and is intentional.
- The result supports a purchase decision but is not an appraisal or future-price forecast.

Add Task 14 to `memory-bank/implementation-plan.md`, update `architecture.md` with `realestate_analysis/valuation.py`, and add only verified evidence to `progress.md`.

- [ ] **Step 2: Run the complete automated suite**

Run:

```bash
python -m compileall app.py realestate_analysis tests
python -m unittest discover -s tests -v
```

Expected: compile exits 0 and all tests pass.

- [ ] **Step 3: Verify the real 615-row dataset**

Run a read-only analysis script against `data/seed/trades.json` or the current cache and print:

- valid trade count;
- index quarter count and latest quarter;
- latest index, quarter change, and year change;
- backtest case count and MAPE;
- one A-type, published-building, actual-floor valuation;
- comparable count, range, status, and confidence.

Confirm all numeric values are finite, latest index is `100.0`, and no future transaction enters any backtest training fold.

- [ ] **Step 4: Run Streamlit and perform visual QA**

Run:

```bash
streamlit run app.py --server.headless true
```

Verify in a browser:

- 360×800: index metrics fit without horizontal overflow; chart labels do not overlap; all four asking inputs are reachable; result cards stack cleanly.
- 1280×900: index chart, asking card, confidence text, and comparable table align with the existing Apple visual system.
- Browser console contains no errors.
- Entering `0` leaves the asking judgment hidden.
- Entering a realistic asking price displays a result and evidence.

- [ ] **Step 5: Review the complete diff**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Confirm `.codex-remote-attachments/` remains untracked and no secrets, cache files, or temporary screenshots are staged.

- [ ] **Step 6: Commit verified docs**

```bash
git add README.md memory-bank/implementation-plan.md memory-bank/progress.md memory-bank/architecture.md
git commit -m "docs: explain price index and asking analysis"
```
