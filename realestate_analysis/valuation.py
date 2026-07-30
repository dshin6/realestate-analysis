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
        row, _ = _design_row(
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


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"deal_date", "price_won", "plan_type", "building", "floor"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required) + ["quarter"])

    work = frame[list(required)].copy()
    work["deal_date"] = pd.to_datetime(work["deal_date"], errors="coerce")
    work["price_won"] = pd.to_numeric(work["price_won"], errors="coerce")
    work["floor"] = pd.to_numeric(work["floor"], errors="coerce")
    work["plan_type"] = work["plan_type"].astype("string").str.strip()
    work["building"] = (
        work["building"]
        .astype("string")
        .fillna(UNKNOWN_BUILDING)
        .str.strip()
        .replace("", UNKNOWN_BUILDING)
    )
    work = work.dropna(subset=["deal_date", "price_won", "floor", "plan_type"])
    work = work[
        (work["price_won"] > 0)
        & (work["floor"] >= 1)
        & work["plan_type"].ne("")
    ].copy()
    work["floor"] = work["floor"].astype(int)
    work["quarter"] = work["deal_date"].dt.to_period("Q").astype(str)
    return work.sort_values("deal_date").reset_index(drop=True)


def _floor_terms(floor: int | float) -> tuple[float, float, float]:
    value = float(floor)
    return value, max(value - 5.0, 0.0), max(value - 15.0, 0.0)


def _design_row(
    quarter: str,
    plan_type: str,
    building: str,
    floor: int,
    quarter_levels: tuple[str, ...],
    type_levels: tuple[str, ...],
    building_levels: tuple[str, ...],
    reference_type: str,
    reference_building: str,
) -> tuple[np.ndarray, tuple[int, ...]]:
    values: list[float] = [1.0]
    values.extend(float(quarter == level) for level in quarter_levels[1:])
    values.extend(
        float(plan_type == level)
        for level in type_levels
        if level != reference_type
    )

    building_indexes: list[int] = []
    for level in building_levels:
        if level == reference_building:
            continue
        building_indexes.append(len(values))
        values.append(float(building == level))

    values.extend(_floor_terms(floor))
    return np.asarray(values, dtype=float), tuple(building_indexes)


def _fit_model(work: pd.DataFrame) -> _HedonicModel:
    quarter_levels = tuple(sorted(work["quarter"].unique()))
    type_levels = tuple(sorted(work["plan_type"].unique()))
    published_buildings = work.loc[
        work["building"] != UNKNOWN_BUILDING,
        "building",
    ]
    building_levels = tuple(sorted(published_buildings.unique()))
    reference_type = str(work["plan_type"].value_counts().idxmax())
    reference_building = (
        str(published_buildings.value_counts().idxmax())
        if not published_buildings.empty
        else UNKNOWN_BUILDING
    )

    rows: list[np.ndarray] = []
    building_indexes: tuple[int, ...] = ()
    for trade in work.itertuples(index=False):
        row, building_indexes = _design_row(
            str(trade.quarter),
            str(trade.plan_type),
            str(trade.building),
            int(trade.floor),
            quarter_levels,
            type_levels,
            building_levels,
            reference_type,
            reference_building,
        )
        rows.append(row)

    design = np.vstack(rows)
    target = np.log(work["price_won"].to_numpy(dtype=float))
    regularized_cross_product = design.T @ design
    for index in building_indexes:
        regularized_cross_product[index, index] += BUILDING_RIDGE_STRENGTH
    coefficients = (
        np.linalg.pinv(regularized_cross_product) @ design.T @ target
    )
    return _HedonicModel(
        coefficients=coefficients,
        quarter_levels=quarter_levels,
        type_levels=type_levels,
        building_levels=building_levels,
        reference_type=reference_type,
        reference_building=reference_building,
    )


def build_price_index(
    frame: pd.DataFrame,
    min_observations: int = MIN_INDEX_OBSERVATIONS,
) -> PriceIndexResult | None:
    work = _clean_frame(frame)
    if len(work) < min_observations:
        return None

    model = _fit_model(work)
    reference_floor = int(round(work["floor"].median()))
    rows = []
    for quarter in model.quarter_levels:
        rows.append(
            {
                "quarter": quarter,
                "level": np.exp(
                    model.predict_log_price(
                        quarter,
                        model.reference_type,
                        model.reference_building,
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
        float(100.0 / previous_value - 1.0) * 100.0
        if previous_value is not None
        else None
    )
    yearly_change = (
        float(100.0 / year_ago_value - 1.0) * 100.0
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


def _empty_comparables() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "deal_date",
            "building",
            "floor",
            "price_won",
            "adjusted_price_won",
            "weight",
        ]
    )


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    total_weight = float(ordered_weights.sum())
    if total_weight <= 0:
        return float(np.quantile(ordered_values, quantile))
    positions = (
        np.cumsum(ordered_weights) - ordered_weights * 0.5
    ) / total_weight
    return float(
        np.interp(
            quantile,
            positions,
            ordered_values,
            left=ordered_values[0],
            right=ordered_values[-1],
        )
    )


def _confidence_label(
    count: int,
    same_building_ratio: float,
    backtest: BacktestResult | None,
) -> str:
    if (
        count >= HIGH_CONFIDENCE_COMPARABLES
        and same_building_ratio >= 0.50
        and backtest is not None
        and backtest.median_absolute_percentage_error_pct <= 7.5
    ):
        return "높음"
    if (
        count >= MEDIUM_CONFIDENCE_COMPARABLES
        and backtest is not None
        and backtest.median_absolute_percentage_error_pct <= 12.5
    ):
        return "보통"
    return "낮음"


def _asking_status(
    asking_price_won: int,
    count: int,
    fair_price_won: int,
    low_price_won: int,
    high_price_won: int,
) -> str:
    if (
        count < MIN_COMPARABLES
        or (high_price_won - low_price_won) / fair_price_won
        > MAX_INTERVAL_WIDTH_RATIO
    ):
        return "판단 자료 부족"
    if asking_price_won < low_price_won:
        return "저평가 가능"
    if asking_price_won <= high_price_won:
        return "적정 범위"
    if asking_price_won <= high_price_won * 1.05:
        return "다소 높음"
    return "높음"


def evaluate_asking_price(
    frame: pd.DataFrame,
    plan_type: str,
    building: str,
    floor: int,
    asking_price_won: int,
    price_index: PriceIndexResult | None = None,
    backtest: BacktestResult | None = None,
) -> AskingPriceResult | None:
    work = _clean_frame(frame)
    index_result = price_index or build_price_index(work)
    if index_result is None:
        return None

    candidates = work[work["plan_type"] == plan_type].copy()
    same_building = candidates[candidates["building"] == building]
    expanded_to_complex = len(same_building) < MIN_COMPARABLES
    if not expanded_to_complex:
        candidates = same_building.copy()
    if candidates.empty:
        return AskingPriceResult(
            count=0,
            fair_price_won=None,
            low_price_won=None,
            high_price_won=None,
            asking_delta_pct=None,
            status="판단 자료 부족",
            confidence="낮음",
            expanded_to_complex=expanded_to_complex,
            same_building_ratio=0.0,
            comparables=_empty_comparables(),
            backtest=backtest,
        )

    model = index_result.model
    target_log_price = model.predict_log_price(
        index_result.latest_quarter,
        plan_type,
        building,
        floor,
    )
    latest_date = work["deal_date"].max()
    adjusted_prices = []
    weights = []
    for trade in candidates.itertuples(index=False):
        source_log_price = model.predict_log_price(
            str(trade.quarter),
            str(trade.plan_type),
            str(trade.building),
            int(trade.floor),
        )
        adjustment = np.exp(
            np.clip(target_log_price - source_log_price, -2.0, 2.0)
        )
        adjusted_prices.append(float(trade.price_won) * adjustment)
        age_days = max((latest_date - trade.deal_date).days, 0)
        weights.append(0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS))

    candidates["adjusted_price_won"] = adjusted_prices
    candidates["weight"] = weights
    candidates = (
        candidates.sort_values(
            ["weight", "deal_date"],
            ascending=[False, False],
        )
        .head(MAX_COMPARABLES)
        .copy()
    )

    values = candidates["adjusted_price_won"].to_numpy(dtype=float)
    comparable_weights = candidates["weight"].to_numpy(dtype=float)
    low_price_won = int(round(_weighted_quantile(values, comparable_weights, 0.20)))
    fair_price_won = int(round(_weighted_quantile(values, comparable_weights, 0.50)))
    high_price_won = int(round(_weighted_quantile(values, comparable_weights, 0.80)))
    count = int(len(candidates))
    same_building_ratio = float((candidates["building"] == building).mean())
    status = _asking_status(
        asking_price_won,
        count,
        fair_price_won,
        low_price_won,
        high_price_won,
    )
    confidence = _confidence_label(
        count,
        same_building_ratio,
        backtest,
    )
    asking_delta_pct = (
        float(asking_price_won / fair_price_won - 1.0) * 100.0
        if fair_price_won
        else None
    )
    comparable_columns = [
        "deal_date",
        "building",
        "floor",
        "price_won",
        "adjusted_price_won",
        "weight",
    ]
    return AskingPriceResult(
        count=count,
        fair_price_won=fair_price_won,
        low_price_won=low_price_won,
        high_price_won=high_price_won,
        asking_delta_pct=asking_delta_pct,
        status=status,
        confidence=confidence,
        expanded_to_complex=expanded_to_complex,
        same_building_ratio=same_building_ratio,
        comparables=candidates[comparable_columns].reset_index(drop=True),
        backtest=backtest,
    )


def backtest_valuation(
    frame: pd.DataFrame,
    max_quarters: int = 8,
) -> BacktestResult | None:
    work = _clean_frame(frame)
    eligible_quarters = []
    for quarter in sorted(work["quarter"].unique()):
        validation_period = pd.Period(quarter, freq="Q")
        training = work[
            work["deal_date"].dt.to_period("Q") < validation_period
        ]
        if len(training) >= MIN_INDEX_OBSERVATIONS:
            eligible_quarters.append(quarter)
    eligible_quarters = eligible_quarters[-max_quarters:]

    absolute_errors: list[float] = []
    percentage_errors: list[float] = []
    covered: list[bool] = []
    for quarter in eligible_quarters:
        validation_period = pd.Period(quarter, freq="Q")
        training = work[
            work["deal_date"].dt.to_period("Q") < validation_period
        ]
        validation = work[work["quarter"] == quarter]
        price_index = build_price_index(training)
        if price_index is None:
            continue
        for trade in validation.itertuples(index=False):
            result = evaluate_asking_price(
                training,
                plan_type=str(trade.plan_type),
                building=str(trade.building),
                floor=int(trade.floor),
                asking_price_won=int(trade.price_won),
                price_index=price_index,
            )
            if (
                result is None
                or result.fair_price_won is None
                or result.low_price_won is None
                or result.high_price_won is None
            ):
                continue
            error = abs(float(trade.price_won) - result.fair_price_won)
            absolute_errors.append(error)
            percentage_errors.append(error / float(trade.price_won) * 100.0)
            covered.append(
                result.low_price_won
                <= float(trade.price_won)
                <= result.high_price_won
            )

    if not absolute_errors:
        return None
    return BacktestResult(
        median_absolute_error_won=int(round(float(np.median(absolute_errors)))),
        median_absolute_percentage_error_pct=float(
            np.median(percentage_errors)
        ),
        interval_coverage_pct=float(np.mean(covered) * 100.0),
        cases=len(absolute_errors),
    )
