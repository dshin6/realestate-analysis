from __future__ import annotations

from dataclasses import dataclass

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
        return pd.DataFrame(columns=["building", "plan_type", "median_price", "trades"])
    return (
        frame.groupby(["building", "plan_type"], as_index=False)
        .agg(median_price=("price_won", "median"), trades=("price_won", "size"))
        .sort_values(["building", "plan_type"])
    )


def floor_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["floor_group", "median_price", "trades"])
    order = [
        "1층",
        "저층 (2층~5층)",
        "중층 (6층~15층)",
        "고층 (16층 이상)",
        "층 정보 없음",
    ]
    result = (
        frame.groupby("floor_group", as_index=False, observed=True)
        .agg(median_price=("price_won", "median"), trades=("price_won", "size"))
    )
    result["floor_group"] = pd.Categorical(result["floor_group"], order, ordered=True)
    return result.sort_values("floor_group")


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
