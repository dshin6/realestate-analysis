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
