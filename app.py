from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from realestate_analysis.analysis import (
    annual_type_counts,
    building_summary,
    enrich_trades,
    estimate_fair_price,
    floor_average_summary,
    won_to_eok,
)
from realestate_analysis.api import PublicDataApiError, current_deal_month, load_trades
from realestate_analysis.config import TARGET_COMPLEX


CACHE_PATH = Path("data/cache/trades.json")
SEED_PATH = Path("data/seed/trades.json")
TYPE_COLORS = {"A": "#2F6B4F", "B": "#D08C45", "C": "#4E79A7", "기타": "#8D8D8D"}


def _service_key() -> str:
    try:
        return str(st.secrets.get("DATA_GO_KR_SERVICE_KEY", "")).strip()
    except Exception:
        return ""


def _price_label(value: float) -> str:
    return f"{won_to_eok(value):,.2f}억"


def _prepare_trades(frame: pd.DataFrame) -> pd.DataFrame:
    return enrich_trades(frame, TARGET_COMPLEX)


def _trade_scatter_figure(frame: pd.DataFrame) -> go.Figure:
    chart_data = frame.copy()
    chart_data["거래일"] = chart_data["deal_date"]
    chart_data["거래가격(억원)"] = chart_data["price_won"] / 100_000_000
    chart_data["타입"] = chart_data["plan_type"]
    chart_data["동"] = chart_data["building"]
    chart_data["실제 층"] = chart_data["floor"]
    figure = px.scatter(
        chart_data,
        x="거래일",
        y="거래가격(억원)",
        color="타입",
        color_discrete_map=TYPE_COLORS,
        hover_data=["동", "실제 층"],
    )
    figure.update_traces(marker={"size": 7, "opacity": 0.7, "line": {"width": 0.5, "color": "#FAF8F3"}})
    figure.update_layout(legend_title_text="타입", hovermode="closest")
    return figure


def _annual_type_volume_figure(frame: pd.DataFrame) -> go.Figure:
    chart_data = annual_type_counts(frame)
    chart_data["연도"] = chart_data["year"].astype(str)
    figure = px.bar(
        chart_data,
        x="연도",
        y="trades",
        color="plan_type",
        barmode="group",
        color_discrete_map=TYPE_COLORS,
        labels={"trades": "거래 건수", "plan_type": "타입"},
    )
    figure.update_layout(legend_title_text="타입")
    return figure


def _floor_distribution_figure(frame: pd.DataFrame) -> go.Figure:
    summary = floor_average_summary(frame)
    groups = [str(value) for value in summary["floor_group"]]
    positions = list(range(len(groups)))
    average_eok = summary["average_price"] / 100_000_000

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=positions,
            y=average_eok,
            name="구간 평균",
            marker={
                "color": "rgba(47, 107, 79, 0.30)",
                "line": {"color": "#2F6B4F", "width": 1.5},
            },
            text=[
                f"평균 {price:,.2f}억<br>{int(trades):,}건"
                for price, trades in zip(average_eok, summary["trades"])
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="층 구간 평균: %{y:,.2f}억<extra></extra>",
        )
    )

    points = frame.sort_values(["floor_group", "deal_date", "price_won"]).reset_index(drop=True).copy()
    points["거래가격(억원)"] = points["price_won"] / 100_000_000
    group_positions = {group: index for index, group in enumerate(groups)}
    points["_x"] = [
        group_positions[str(group)] + ((index % 17) - 8) * 0.022
        for index, group in enumerate(points["floor_group"])
    ]

    for plan_type in ["A", "B", "C", "기타"]:
        subset = points[points["plan_type"] == plan_type]
        if subset.empty:
            continue
        custom_data = list(
            zip(
                subset["deal_date"].dt.strftime("%Y-%m-%d"),
                subset["building"],
                subset["floor"],
                subset["plan_type"],
            )
        )
        figure.add_trace(
            go.Scatter(
                x=subset["_x"],
                y=subset["거래가격(억원)"],
                mode="markers",
                name=f"{plan_type} 실거래",
                marker={
                    "color": TYPE_COLORS[plan_type],
                    "size": 7,
                    "opacity": 0.65,
                    "line": {"color": "#FAF8F3", "width": 0.5},
                },
                customdata=custom_data,
                hovertemplate=(
                    "계약일: %{customdata[0]}<br>"
                    "타입: %{customdata[3]}<br>"
                    "동: %{customdata[1]}<br>"
                    "실제 층: %{customdata[2]}층<br>"
                    "거래가격: %{y:,.2f}억<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        barmode="overlay",
        xaxis={"tickmode": "array", "tickvals": positions, "ticktext": groups, "title": "층 구간"},
        yaxis={"title": "거래가격(억원)", "rangemode": "tozero"},
        legend_title_text="표시",
        margin={"t": 55},
    )
    return figure


def _load_data(service_key: str, force_refresh: bool) -> tuple[pd.DataFrame, str]:
    progress_bar = st.progress(0, text="국토교통부 실거래를 가져오는 중입니다.")

    def update_progress(done: int, total: int) -> None:
        progress_bar.progress(done / total, text=f"실거래 수집 중 · {done}/{total}회 조회")

    frame, fetched_at = load_trades(
        service_key=service_key,
        config=TARGET_COMPLEX,
        end_ym=current_deal_month(),
        cache_path=CACHE_PATH,
        seed_path=SEED_PATH,
        force_refresh=force_refresh,
        progress=update_progress,
    )
    progress_bar.empty()
    return frame, fetched_at


def _render_setup() -> None:
    st.info("공공데이터포털 인증키를 로컬 설정에 넣으면 실거래를 불러올 수 있습니다.")
    st.code(
        "cp .streamlit/secrets.toml.example .streamlit/secrets.toml\n"
        "# secrets.toml을 열어 DATA_GO_KR_SERVICE_KEY 값을 입력\n"
        "streamlit run app.py",
        language="bash",
    )
    st.caption("인증키는 채팅이나 소스코드에 붙여 넣지 마세요. secrets.toml은 Git에서 제외됩니다.")


def main() -> None:
    st.set_page_config(page_title="한빛마을 실거래 분석", page_icon="🏠", layout="wide")
    st.title("한빛마을 한화꿈에그린 실거래 분석")
    st.caption("A/B/C 평면 타입과 동·층에 따른 실제 거래가격 차이를 확인합니다.")

    service_key = _service_key()
    if not service_key and not CACHE_PATH.exists() and not SEED_PATH.exists():
        _render_setup()
        st.stop()

    with st.sidebar:
        st.header("데이터")
        force_refresh = st.button("최신 데이터 다시 받기", disabled=not bool(service_key))

    if "trades" not in st.session_state or force_refresh:
        try:
            trades, fetched_at = _load_data(service_key, force_refresh)
        except (PublicDataApiError, OSError, ValueError) as exc:
            st.error(f"실거래 데이터를 불러오지 못했습니다: {exc}")
            st.info("인증키와 공공데이터포털의 API 활용 상태를 확인한 뒤 다시 시도해 주세요.")
            st.stop()
        except Exception as exc:
            st.error(f"네트워크 또는 API 요청 중 오류가 발생했습니다: {exc}")
            st.stop()
        st.session_state["trades"] = trades
        st.session_state["fetched_at"] = fetched_at

    trades = _prepare_trades(st.session_state["trades"])
    fetched_at = st.session_state.get("fetched_at", "")
    if trades.empty:
        st.warning("현재 조건에서 대상 단지의 정상 실거래를 찾지 못했습니다.")
        st.caption("단지명·법정동 코드 또는 API 응답 필드가 변경됐는지 확인이 필요합니다.")
        st.stop()

    min_year, max_year = int(trades["year"].min()), int(trades["year"].max())
    available_types = [value for value in ["A", "B", "C", "기타"] if value in trades["plan_type"].unique()]
    with st.sidebar:
        st.header("분석 조건")
        year_range = st.slider("거래 연도", min_year, max_year, (min_year, max_year))
        selected_types = st.multiselect("평면 타입", available_types, default=available_types)
        buildings = sorted(trades["building"].unique())
        selected_buildings = st.multiselect("동", buildings, default=buildings)

    filtered = trades[
        trades["year"].between(*year_range)
        & trades["plan_type"].isin(selected_types)
        & trades["building"].isin(selected_buildings)
    ]
    if filtered.empty:
        st.warning("선택한 조건에 해당하는 거래가 없습니다. 필터 범위를 넓혀 주세요.")
        st.stop()

    latest_date = filtered["deal_date"].max()
    recent = filtered[filtered["deal_date"] >= latest_date - pd.DateOffset(months=12)]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("선택 거래", f"{len(filtered):,}건")
    col2.metric("최근 거래일", latest_date.strftime("%Y-%m-%d"))
    col3.metric("최근 12개월 중앙값", _price_label(recent["price_won"].median()) if not recent.empty else "자료 없음")
    col4.metric("최근 12개월 거래", f"{len(recent):,}건")

    st.subheader("전체 실거래 가격")
    st.plotly_chart(_trade_scatter_figure(filtered), width="stretch")
    st.caption("점 하나가 실거래 한 건입니다. 색상은 A/B/C 평면 타입을 구분합니다.")

    st.subheader("연도별 타입 거래 건수")
    st.plotly_chart(_annual_type_volume_figure(filtered), width="stretch")
    st.caption("거래 건수는 타입별 거래 활발도를 보여주지만 선호도를 직접 측정한 값은 아닙니다.")

    st.subheader("층 구간별 평균과 실거래 분포")
    st.plotly_chart(_floor_distribution_figure(filtered), width="stretch")
    st.caption("반투명 막대는 구간 평균, 점은 개별 실거래입니다. 거래 수와 점의 분포를 함께 확인하세요.")

    st.subheader("동·타입별 가격 비교")
    building_data = building_summary(filtered)
    building_data["중앙값(억원)"] = building_data["median_price"] / 100_000_000
    fig_building = px.bar(
        building_data,
        x="building",
        y="중앙값(억원)",
        color="plan_type",
        barmode="group",
        color_discrete_map=TYPE_COLORS,
        labels={"building": "동", "plan_type": "타입"},
        hover_data={"trades": True, "median_price": False},
    )
    st.plotly_chart(fig_building, width="stretch")
    st.caption("동 정보가 공개되지 않은 거래는 '미공개'로 묶입니다.")

    with st.expander("검토 중인 매물 가격 비교", expanded=True):
        form_left, form_mid, form_right = st.columns(3)
        plan_type = form_left.selectbox("매물 타입", ["A", "B", "C"])
        floor_group = form_mid.selectbox(
            "매물 층 구간",
            ["1층", "저층 (2층~5층)", "중층 (6층~15층)", "고층 (16층 이상)"],
        )
        asking_eok = form_right.number_input("매물 가격(억원, 선택)", min_value=0.0, step=0.1, value=0.0)
        estimate = estimate_fair_price(
            trades,
            plan_type,
            floor_group,
            int(asking_eok * 100_000_000) if asking_eok else None,
        )
        if estimate:
            st.write(
                f"최근 3년 유사 거래 **{estimate.count}건**의 중앙값은 "
                f"**{_price_label(estimate.median_won)}**, 중간 50% 범위는 "
                f"**{_price_label(estimate.q25_won)}~{_price_label(estimate.q75_won)}**입니다."
            )
            if estimate.asking_delta_pct is not None:
                st.metric("매물가와 유사 거래 중앙값 차이", f"{estimate.asking_delta_pct:+.1f}%")
        else:
            st.info("최근 3년에 선택 조건과 같은 거래가 없습니다. 다른 층 구간이나 타입을 선택해 보세요.")

    st.subheader("개별 실거래")
    detail = filtered.sort_values("deal_date", ascending=False)[
        ["deal_date", "price_won", "plan_type", "exclusive_area", "building", "floor", "floor_group"]
    ].copy()
    detail["거래가격"] = detail["price_won"].map(_price_label)
    detail = detail.rename(
        columns={
            "deal_date": "계약일",
            "plan_type": "타입",
            "exclusive_area": "전용면적(㎡)",
            "building": "동",
            "floor": "층",
            "floor_group": "층 구간",
        }
    ).drop(columns="price_won")
    st.dataframe(detail, width="stretch", hide_index=True)

    st.divider()
    st.caption(
        f"출처: 국토교통부 아파트 매매 실거래가 상세 Open API · 로컬 수집 시각: {fetched_at or '미상'} · "
        "취소 거래 제외 · 동 미공개 거래는 '미공개' 표시"
    )
    st.warning("표본 수가 적은 조건의 백분율은 크게 흔들릴 수 있습니다. 거래 수와 실제 계약일을 함께 확인하세요.")


if __name__ == "__main__":
    main()
