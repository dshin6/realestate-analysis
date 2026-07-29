from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ComplexConfig


def map_plan_type(area: float, config: ComplexConfig, tolerance: float = 0.015) -> str:
    candidates = [(abs(area - known), label) for known, label in config.type_by_area.items()]
    distance, label = min(candidates)
    return label if distance <= tolerance else "기타"


def normalize_building(value: object) -> str:
    if value is None or pd.isna(value):
        return "미공개"
    text = str(value).strip().replace("동", "")
    return text or "미공개"


def classify_floor(floor: int) -> str:
    if floor == 1:
        return "1층"
    if 2 <= floor <= 5:
        return "저층 (2층~5층)"
    if 6 <= floor <= 15:
        return "중층 (6층~15층)"
    if floor >= 16:
        return "고층 (16층 이상)"
    return "층 정보 없음"


def enrich_trades(frame: pd.DataFrame, config: ComplexConfig) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["deal_date"] = pd.to_datetime(result["deal_date"])
    result["price_won"] = pd.to_numeric(result["price_won"])
    result["exclusive_area"] = pd.to_numeric(result["exclusive_area"])
    result["floor"] = pd.to_numeric(result["floor"]).astype(int)
    result["building"] = result["building"].map(normalize_building)
    result["plan_type"] = result["exclusive_area"].map(lambda area: map_plan_type(area, config))
    result["floor_group"] = result["floor"].map(classify_floor)
    result["year"] = result["deal_date"].dt.year
    result["quarter"] = result["deal_date"].dt.to_period("Q").astype(str)
    return result


def annual_type_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["year", "plan_type", "median_price", "trades", "premium_pct"])
    annual_market = frame.groupby("year")["price_won"].median().rename("market_median")
    work = frame.join(annual_market, on="year")
    work["normalized_price"] = work["price_won"] / work["market_median"]
    summary = (
        work.groupby(["year", "plan_type"], as_index=False)
        .agg(
            median_price=("price_won", "median"),
            trades=("price_won", "size"),
            normalized_median=("normalized_price", "median"),
        )
    )
    summary["premium_pct"] = (summary["normalized_median"] - 1) * 100
    return summary


def annual_type_counts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["year", "plan_type", "trades"])
    return (
        frame.groupby(["year", "plan_type"], as_index=False)
        .agg(trades=("price_won", "size"))
        .sort_values(["year", "plan_type"])
    )


def quarterly_type_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["quarter", "plan_type", "median_price", "trades", "premium_pct"])
    quarter_market = frame.groupby("quarter")["price_won"].median().rename("market_median")
    work = frame.join(quarter_market, on="quarter")
    work["normalized_price"] = work["price_won"] / work["market_median"]
    summary = (
        work.groupby(["quarter", "plan_type"], as_index=False)
        .agg(
            median_price=("price_won", "median"),
            trades=("price_won", "size"),
            normalized_median=("normalized_price", "median"),
        )
    )
    summary["premium_pct"] = (summary["normalized_median"] - 1) * 100
    return summary


def building_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["building", "plan_type", "average_price", "trades"])
    return (
        frame.groupby(["building", "plan_type"], as_index=False)
        .agg(average_price=("price_won", "mean"), trades=("price_won", "size"))
        .sort_values(["building", "plan_type"])
    )


def floor_average_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["floor_group", "plan_type", "average_price", "trades"])
    order = [
        "1층",
        "저층 (2층~5층)",
        "중층 (6층~15층)",
        "고층 (16층 이상)",
        "층 정보 없음",
    ]
    result = (
        frame.groupby(["floor_group", "plan_type"], as_index=False, observed=True)
        .agg(average_price=("price_won", "mean"), trades=("price_won", "size"))
    )
    result["floor_group"] = pd.Categorical(result["floor_group"], order, ordered=True)
    return result.sort_values(["floor_group", "plan_type"])


def floor_price_index_by_type(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "plan_type",
        "floor_group",
        "average_price",
        "trades",
        "price_index_pct",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    order = [
        "1층",
        "저층 (2층~5층)",
        "중층 (6층~15층)",
        "고층 (16층 이상)",
    ]
    summary = floor_average_summary(frame)
    summary = summary[summary["floor_group"].isin(order)].copy()
    high_floor = (
        summary[summary["floor_group"] == "고층 (16층 이상)"]
        .set_index("plan_type")["average_price"]
        .rename("high_floor_average")
    )
    result = summary.join(high_floor, on="plan_type").dropna(subset=["high_floor_average"])
    result["price_index_pct"] = result["average_price"] / result["high_floor_average"] * 100
    return result[columns].sort_values(["floor_group", "plan_type"])


def adjusted_premium_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "factor",
        "category",
        "reference",
        "premium_pct",
        "ci_low_pct",
        "ci_high_pct",
        "trades",
        "observations",
        "months",
        "is_reference",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    floor_order = [
        "1층",
        "저층 (2층~5층)",
        "중층 (6층~15층)",
        "고층 (16층 이상)",
    ]
    work = frame[
        (pd.to_numeric(frame["price_won"], errors="coerce") > 0)
        & frame["floor_group"].isin(floor_order)
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    work["month"] = pd.to_datetime(work["deal_date"]).dt.to_period("M").astype(str)
    type_levels = [
        value for value in ["A", "B", "C", "기타"] if value in work["plan_type"].unique()
    ]
    floor_levels = [value for value in floor_order if value in work["floor_group"].unique()]
    month_levels = sorted(work["month"].unique())
    type_reference = "B" if "B" in type_levels else str(work["plan_type"].value_counts().idxmax())
    floor_reference = (
        "고층 (16층 이상)"
        if "고층 (16층 이상)" in floor_levels
        else str(work["floor_group"].value_counts().idxmax())
    )

    design_columns = [np.ones(len(work), dtype=float)]
    coefficient_index: dict[tuple[str, str], int] = {}
    for month in month_levels[1:]:
        design_columns.append((work["month"] == month).to_numpy(dtype=float))
    for category in type_levels:
        if category == type_reference:
            continue
        coefficient_index[("타입", category)] = len(design_columns)
        design_columns.append((work["plan_type"] == category).to_numpy(dtype=float))
    for category in floor_levels:
        if category == floor_reference:
            continue
        coefficient_index[("층 구간", category)] = len(design_columns)
        design_columns.append((work["floor_group"] == category).to_numpy(dtype=float))

    design = np.column_stack(design_columns)
    target = np.log(work["price_won"].to_numpy(dtype=float))
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < design.shape[1] or len(work) <= rank:
        return pd.DataFrame(columns=columns)

    residuals = target - design @ coefficients
    bread = np.linalg.pinv(design.T @ design)
    weighted_design = design * residuals[:, None]
    meat = weighted_design.T @ weighted_design
    covariance = (len(work) / (len(work) - rank)) * bread @ meat @ bread
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))

    rows: list[dict[str, object]] = []
    for factor, levels, reference, source_column in [
        ("타입", type_levels, type_reference, "plan_type"),
        ("층 구간", floor_levels, floor_reference, "floor_group"),
    ]:
        if len(levels) < 2:
            continue
        for category in levels:
            is_reference = category == reference
            if is_reference:
                coefficient = 0.0
                standard_error = 0.0
            else:
                index = coefficient_index[(factor, category)]
                coefficient = float(coefficients[index])
                standard_error = float(standard_errors[index])
            rows.append(
                {
                    "factor": factor,
                    "category": category,
                    "reference": reference,
                    "premium_pct": float(np.expm1(coefficient) * 100),
                    "ci_low_pct": float(np.expm1(coefficient - 1.96 * standard_error) * 100),
                    "ci_high_pct": float(np.expm1(coefficient + 1.96 * standard_error) * 100),
                    "trades": int((work[source_column] == category).sum()),
                    "observations": int(len(work)),
                    "months": int(len(month_levels)),
                    "is_reference": is_reference,
                }
            )
    return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True)
class FairPriceResult:
    count: int
    median_won: int
    q25_won: int
    q75_won: int
    asking_delta_pct: float | None
    start_date: pd.Timestamp


def estimate_fair_price(
    frame: pd.DataFrame,
    plan_type: str,
    floor_group: str,
    asking_price_won: int | None = None,
    years: int = 3,
) -> FairPriceResult | None:
    if frame.empty:
        return None
    end = frame["deal_date"].max()
    start = end - pd.DateOffset(years=years)
    comparable = frame[
        (frame["deal_date"] >= start)
        & (frame["plan_type"] == plan_type)
        & (frame["floor_group"] == floor_group)
    ]["price_won"]
    if comparable.empty:
        return None
    median = int(comparable.median())
    delta = None
    if asking_price_won is not None and median:
        delta = (asking_price_won / median - 1) * 100
    return FairPriceResult(
        count=int(comparable.size),
        median_won=median,
        q25_won=int(comparable.quantile(0.25)),
        q75_won=int(comparable.quantile(0.75)),
        asking_delta_pct=delta,
        start_date=start,
    )


def won_to_eok(value: int | float) -> float:
    return float(value) / 100_000_000
