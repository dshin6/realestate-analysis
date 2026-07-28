from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import unquote

import pandas as pd
import requests

from .config import ComplexConfig, normalized_name


ENDPOINT = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"


class PublicDataApiError(RuntimeError):
    """공공데이터 API가 정상 결과를 반환하지 않았을 때 발생한다."""


def iter_months(start_ym: str, end_ym: str) -> Iterable[str]:
    year, month = int(start_ym[:4]), int(start_ym[4:])
    end_year, end_month = int(end_ym[:4]), int(end_ym[4:])
    while (year, month) <= (end_year, end_month):
        yield f"{year:04d}{month:02d}"
        month += 1
        if month == 13:
            year += 1
            month = 1


def _text(item: ET.Element, *names: str) -> str:
    for name in names:
        node = item.find(name)
        if node is not None and node.text:
            return node.text.strip()
    return ""


def parse_trade_xml(xml_text: str) -> tuple[list[dict], int]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise PublicDataApiError("API 응답을 XML로 읽을 수 없습니다.") from exc

    result_code = root.findtext(".//resultCode", default="").strip()
    result_message = root.findtext(".//resultMsg", default="").strip()
    if result_code and result_code not in {"00", "000"}:
        raise PublicDataApiError(f"공공데이터 API 오류 {result_code}: {result_message}")

    rows: list[dict] = []
    for item in root.findall(".//item"):
        amount_text = _text(item, "dealAmount", "거래금액").replace(",", "")
        area_text = _text(item, "excluUseAr", "전용면적")
        floor_text = _text(item, "floor", "층")
        year_text = _text(item, "dealYear", "년")
        month_text = _text(item, "dealMonth", "월")
        day_text = _text(item, "dealDay", "일")
        try:
            deal_date = datetime(
                int(year_text), int(month_text), int(day_text)
            ).date().isoformat()
            amount_won = int(float(amount_text)) * 10_000
            area = float(area_text)
            floor = int(float(floor_text))
        except (TypeError, ValueError):
            continue

        rows.append(
            {
                "deal_date": deal_date,
                "price_won": amount_won,
                "exclusive_area": area,
                "floor": floor,
                "apartment_name": _text(item, "aptNm", "아파트"),
                "building": _text(item, "aptDong", "동"),
                "legal_dong": _text(item, "umdNm", "법정동"),
                "jibun": _text(item, "jibun", "지번"),
                "apt_seq": _text(item, "aptSeq"),
                "cancel_date": _text(item, "cdealDay", "해제사유발생일"),
                "cancelled": bool(_text(item, "cdealDay", "해제사유발생일")),
            }
        )

    total_count_text = root.findtext(".//totalCount", default=str(len(rows))).strip()
    try:
        total_count = int(total_count_text)
    except ValueError:
        total_count = len(rows)
    return rows, total_count


def fetch_month(
    service_key: str,
    lawd_code: str,
    deal_ym: str,
    session: requests.Session | None = None,
) -> list[dict]:
    client = session or requests.Session()
    page, rows = 1, []
    while True:
        response = client.get(
            ENDPOINT,
            params={
                "serviceKey": unquote(service_key.strip()),
                "LAWD_CD": lawd_code,
                "DEAL_YMD": deal_ym,
                "pageNo": page,
                "numOfRows": 1000,
            },
            timeout=30,
        )
        response.raise_for_status()
        page_rows, total_count = parse_trade_xml(response.text)
        for row in page_rows:
            row["source_lawd_code"] = lawd_code
        rows.extend(page_rows)
        if len(rows) >= total_count or not page_rows:
            return rows
        page += 1


def _filter_target(rows: list[dict], config: ComplexConfig) -> pd.DataFrame:
    aliases = {normalized_name(name) for name in config.aliases}
    selected = [
        row
        for row in rows
        if normalized_name(row.get("apartment_name", "")) in aliases
        and row.get("legal_dong", "") == config.legal_dong
        and not row.get("cancelled", False)
    ]
    frame = pd.DataFrame(selected)
    if frame.empty:
        return frame

    # 동일 월을 구·신 법정동코드로 조회했을 때 생기는 중복만 제거한다.
    duplicate_key = [
        "deal_date",
        "price_won",
        "exclusive_area",
        "floor",
        "apartment_name",
        "building",
        "jibun",
    ]
    return frame.drop_duplicates(subset=duplicate_key).sort_values("deal_date")


def load_trades(
    service_key: str,
    config: ComplexConfig,
    end_ym: str,
    cache_path: Path = Path("data/cache/trades.json"),
    force_refresh: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, str]:
    if cache_path.exists() and not force_refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("end_ym") == end_ym:
            return pd.DataFrame(payload.get("rows", [])), payload.get("fetched_at", "")

    months = list(iter_months(config.start_ym, end_ym))
    all_rows: list[dict] = []
    total_calls = len(months) * len(config.lawd_codes)
    completed = 0
    with requests.Session() as session:
        for deal_ym in months:
            for lawd_code in config.lawd_codes:
                all_rows.extend(fetch_month(service_key, lawd_code, deal_ym, session))
                completed += 1
                if progress:
                    progress(completed, total_calls)

    frame = _filter_target(all_rows, config)
    fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"fetched_at": fetched_at, "end_ym": end_ym, "rows": frame.to_dict("records")},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return frame, fetched_at


def current_deal_month() -> str:
    today = datetime.now().date()
    return f"{today.year:04d}{today.month:02d}"
