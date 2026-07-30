from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from realestate_analysis.analysis import (
    adjusted_premium_summary,
    annual_type_counts,
    building_summary,
    enrich_trades,
    estimate_fair_price,
    floor_average_summary,
    floor_price_index_by_type,
    won_to_eok,
)
from realestate_analysis.api import PublicDataApiError, current_deal_month, load_trades
from realestate_analysis.config import TARGET_COMPLEX


CACHE_PATH = Path("data/cache/trades.json")
SEED_PATH = Path("data/seed/trades.json")
TYPE_COLORS = {"A": "#0071E3", "B": "#34C759", "C": "#FF9F0A", "기타": "#8E8E93"}
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False, "responsive": True}
APPLE_STYLES = """
<style>
:root {
    --apple-blue: #0071e3;
    --apple-text: #1d1d1f;
    --apple-muted: #6e6e73;
    --apple-surface: rgba(255, 255, 255, 0.88);
    --apple-border: rgba(0, 0, 0, 0.09);
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
        "SF Pro Text", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 12% 0%, rgba(0, 113, 227, 0.08), transparent 28rem),
        #f5f5f7;
    color: var(--apple-text);
}

.block-container {
    max-width: 1180px;
    padding-top: 3.5rem;
    padding-bottom: 5rem;
}

.apple-eyebrow {
    color: var(--apple-blue);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    margin: 0 0 0.6rem;
    text-transform: uppercase;
}

.apple-subtitle {
    color: var(--apple-muted);
    font-size: clamp(1rem, 2vw, 1.18rem);
    line-height: 1.65;
    margin: -0.4rem 0 2rem;
    max-width: 44rem;
}

h1, h2, h3, h4 {
    color: var(--apple-text);
    letter-spacing: -0.035em;
}

h1 {
    font-size: clamp(2.15rem, 4.5vw, 3.25rem) !important;
    font-weight: 750 !important;
    line-height: 1.08 !important;
    max-width: 68rem;
}

h2, h3, h4 {
    font-weight: 700 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--apple-surface);
    border: 1px solid var(--apple-border);
    border-radius: 1.25rem;
    box-shadow: 0 16px 44px rgba(0, 0, 0, 0.055);
    backdrop-filter: blur(20px);
}

div[data-testid="stMetric"] {
    min-height: 8.5rem;
    padding: 1.15rem 1.2rem;
    background: var(--apple-surface);
    border: 1px solid var(--apple-border);
    border-radius: 1.15rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.045);
}

div[data-testid="stMetricLabel"] {
    color: var(--apple-muted);
}

div[data-testid="stMetricValue"] {
    color: var(--apple-text);
    font-weight: 700;
    letter-spacing: -0.035em;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 999px;
    font-weight: 650;
    min-height: 2.75rem;
}

div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.76);
    border: 1px solid var(--apple-border);
    border-radius: 1rem;
    overflow: hidden;
}

div[data-testid="stPlotlyChart"] {
    background: #ffffff;
    border: 1px solid var(--apple-border);
    border-radius: 1.25rem;
    padding: 0.35rem;
    box-shadow: 0 12px 34px rgba(0, 0, 0, 0.045);
    overflow: hidden;
}

@media (max-width: 640px) {
    .block-container {
        padding: 4.5rem 1rem 4rem;
    }

    .apple-subtitle {
        margin-bottom: 1.35rem;
    }

    div[data-testid="stMetric"] {
        min-height: 7rem;
    }

    div[data-testid="stPlotlyChart"] {
        border-radius: 1rem;
        padding: 0;
    }
}
</style>
"""


def _inject_styles() -> None:
    st.markdown(APPLE_STYLES, unsafe_allow_html=True)


def _polish_chart(figure: go.Figure, *, height: int = 390) -> go.Figure:
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, Apple SD Gothic Neo, sans-serif", "color": "#1D1D1F", "size": 13},
        hoverlabel={"bgcolor": "#1D1D1F", "font_color": "#FFFFFF", "bordercolor": "#1D1D1F"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "title_text": "",
        },
        margin={"l": 18, "r": 18, "t": 56, "b": 42},
    )
    figure.update_xaxes(showgrid=False, zeroline=False, automargin=True)
    figure.update_yaxes(gridcolor="rgba(0,0,0,0.07)", zeroline=False, automargin=True)
    return figure


def _show_chart(figure: go.Figure) -> None:
    st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG)


def _service_key() -> str:
    try:
        return str(st.secrets.get("DATA_GO_KR_SERVICE_KEY", "")).strip()
    except Exception:
        return ""


def _price_label(value: float) -> str:
    return f"{won_to_eok(value):,.2f}억"


def _prepare_trades(frame: pd.DataFrame) -> pd.DataFrame:
    return enrich_trades(frame, TARGET_COMPLEX)


def _filter_by_month_range(
    frame: pd.DataFrame,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
) -> pd.DataFrame:
    start = pd.Timestamp(start_month).to_period("M").to_timestamp()
    end_exclusive = pd.Timestamp(end_month).to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
    return frame[(frame["deal_date"] >= start) & (frame["deal_date"] < end_exclusive)]


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
    return _polish_chart(figure)


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
    return _polish_chart(figure)


def _floor_distribution_figure(frame: pd.DataFrame) -> go.Figure:
    summary = floor_average_summary(frame)
    groups = [str(value) for value in summary["floor_group"].drop_duplicates()]
    positions = list(range(len(groups)))
    group_positions = {group: index for index, group in enumerate(groups)}
    plan_types = [value for value in ["A", "B", "C", "기타"] if value in summary["plan_type"].unique()]
    spacing = 0.22
    type_offsets = {
        plan_type: (index - (len(plan_types) - 1) / 2) * spacing
        for index, plan_type in enumerate(plan_types)
    }

    figure = go.Figure()
    for plan_type in plan_types:
        subset = summary[summary["plan_type"] == plan_type]
        average_eok = subset["average_price"] / 100_000_000
        figure.add_trace(
            go.Bar(
                x=[group_positions[str(group)] + type_offsets[plan_type] for group in subset["floor_group"]],
                y=average_eok,
                width=spacing * 0.82,
                name=f"{plan_type} 평균",
                legendgroup=plan_type,
                marker={
                    "color": TYPE_COLORS[plan_type],
                    "opacity": 0.88,
                    "line": {"color": TYPE_COLORS[plan_type], "width": 1.5},
                },
                text=[
                    f"{price:,.2f}억<br>{int(trades):,}건"
                    for price, trades in zip(average_eok, subset["trades"])
                ],
                textposition="outside",
                cliponaxis=False,
                customdata=subset[["trades"]].to_numpy(),
                hovertemplate=(
                    f"타입: {plan_type}<br>"
                    "층 구간 평균: %{y:,.2f}억<br>"
                    "거래 건수: %{customdata[0]:,}건<extra></extra>"
                ),
            )
        )

    points = frame.sort_values(["floor_group", "deal_date", "price_won"]).reset_index(drop=True).copy()
    points["거래가격(억원)"] = points["price_won"] / 100_000_000
    points["_x"] = [
        group_positions[str(group)]
        + type_offsets[plan_type]
        + ((index % 7) - 3) * 0.012
        for index, (group, plan_type) in enumerate(zip(points["floor_group"], points["plan_type"]))
    ]

    for plan_type in plan_types:
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
                legendgroup=plan_type,
                showlegend=False,
                marker={
                    "color": TYPE_COLORS[plan_type],
                    "size": 7,
                    "opacity": 0.35,
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
        legend_title_text="타입별 평균",
        margin={"t": 55},
    )
    return _polish_chart(figure, height=430)


def _floor_price_index_figure(frame: pd.DataFrame) -> go.Figure:
    chart_data = floor_price_index_by_type(frame).copy()
    chart_data["고층 대비 가격(%)"] = chart_data["price_index_pct"]
    chart_data["평균가격(억원)"] = chart_data["average_price"] / 100_000_000
    chart_data["타입"] = chart_data["plan_type"]
    figure = px.bar(
        chart_data,
        x="floor_group",
        y="고층 대비 가격(%)",
        color="타입",
        barmode="group",
        color_discrete_map=TYPE_COLORS,
        text_auto=".1f",
        labels={"floor_group": "층 구간"},
        hover_data={
            "평균가격(억원)": ":.2f",
            "trades": True,
            "price_index_pct": False,
            "average_price": False,
            "plan_type": False,
        },
    )
    figure.add_hline(
        y=100,
        line_dash="dash",
        line_color="#4D5562",
        annotation_text="고층 기준 100%",
        annotation_position="top left",
    )
    figure.update_traces(texttemplate="%{y:.1f}%", textposition="outside", cliponaxis=False)
    figure.update_layout(
        legend_title_text="타입",
        yaxis={"title": "고층 대비 가격(%)", "rangemode": "tozero"},
        margin={"t": 55},
    )
    return _polish_chart(figure)


def _adjusted_premium_figure(
    frame: pd.DataFrame,
    factor: str,
    *,
    summary: pd.DataFrame | None = None,
) -> go.Figure:
    if factor not in {"타입", "층 구간"}:
        raise ValueError(f"지원하지 않는 비교 기준입니다: {factor}")
    if summary is None:
        summary = adjusted_premium_summary(frame)

    subset = summary[
        (summary["factor"] == factor) & (~summary["is_reference"])
    ].copy()
    figure = go.Figure()
    if subset.empty:
        return figure

    reference = str(subset.iloc[0]["reference"])
    reference_label = "B타입" if factor == "타입" and reference == "B" else (
        "고층" if factor == "층 구간" and reference == "고층 (16층 이상)" else reference
    )
    category_labels = (
        [f"{category}타입" for category in subset["category"]]
        if factor == "타입"
        else [
            str(category).split(" (", 1)[0] for category in subset["category"]
        ]
    )
    premium = subset["premium_pct"].to_numpy()
    descriptions = [f"{value:+.1f}%" for value in premium]
    colors = (
        [TYPE_COLORS.get(str(category), "#8D8D8D") for category in subset["category"]]
        if factor == "타입"
        else ["#0071E3"] * len(subset)
    )
    figure.add_trace(
        go.Bar(
            x=premium,
            y=category_labels,
            orientation="h",
            marker={"color": colors, "opacity": 0.92},
            text=descriptions,
            textposition="inside",
            insidetextanchor="middle",
            constraintext="inside",
            textfont={"color": "#FFFFFF", "size": 12},
            cliponaxis=False,
            customdata=subset[
                ["ci_low_pct", "ci_high_pct", "trades", "months"]
            ].to_numpy(),
            hovertemplate=(
                "%{y}<br>"
                f"{reference_label} 대비: %{{x:+.2f}}%<br>"
                "95% 범위: %{customdata[0]:+.2f}% ~ %{customdata[1]:+.2f}%<br>"
                "거래 건수: %{customdata[2]:,}건<br>"
                "분석 월 수: %{customdata[3]:,}개월<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_dash="dash", line_color="#4D5562")
    _polish_chart(figure, height=260 if factor == "타입" else 320)
    figure.update_layout(
        showlegend=False,
        xaxis={"title": "기준 대비 가격 차이(%)", "zeroline": False, "automargin": True},
        yaxis={"title": "", "autorange": "reversed", "automargin": True},
        margin={"t": 24, "l": 12, "r": 48, "b": 55},
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
    st.set_page_config(
        page_title="한빛마을 실거래 분석",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()
    st.markdown('<p class="apple-eyebrow">Hanbit Village · Real Estate Intelligence</p>', unsafe_allow_html=True)
    st.title("한빛마을 한화꿈에그린 실거래 분석")
    st.markdown(
        '<p class="apple-subtitle">실거래 흐름과 A/B/C 평면 타입, 동·층별 가격 차이를 '
        "한눈에 비교해 매수 판단의 기준을 만듭니다.</p>",
        unsafe_allow_html=True,
    )

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

    min_month = trades["deal_date"].min().to_period("M").to_timestamp()
    max_month = trades["deal_date"].max().to_period("M").to_timestamp()
    month_options = pd.date_range(min_month, max_month, freq="MS").to_list()
    available_types = [value for value in ["A", "B", "C", "기타"] if value in trades["plan_type"].unique()]
    buildings = sorted(trades["building"].unique())
    with st.container(border=True):
        st.markdown("#### 분석 조건")
        st.caption("기간 조절은 휴대폰에서도 항상 이곳에 표시됩니다.")
        month_range = st.select_slider(
            "거래 기간",
            options=month_options,
            value=(min_month, max_month),
            format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m"),
        )
        filter_left, filter_right = st.columns(2)
        with filter_left:
            selected_types = st.multiselect("평면 타입", available_types, default=available_types)
        with filter_right:
            selected_buildings = st.multiselect("동", buildings, default=buildings)

    period_trades = _filter_by_month_range(trades, *month_range)
    filtered = period_trades[
        period_trades["plan_type"].isin(selected_types)
        & period_trades["building"].isin(selected_buildings)
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
    _show_chart(_trade_scatter_figure(filtered))
    st.caption("점 하나가 실거래 한 건입니다. 색상은 A/B/C 평면 타입을 구분합니다.")

    st.subheader("연도별 타입 거래 건수")
    _show_chart(_annual_type_volume_figure(filtered))
    st.caption("거래 건수는 타입별 거래 활발도를 보여주지만 선호도를 직접 측정한 값은 아닙니다.")

    st.subheader("층 구간별 평균과 실거래 분포")
    _show_chart(_floor_distribution_figure(filtered))
    st.caption(
        "진한 막대는 A/B/C 타입별 구간 평균, 투명한 같은 색 점은 해당 타입의 개별 실거래입니다. "
        "막대의 거래 수와 점의 분포를 함께 확인하세요."
    )

    st.subheader("고층 대비 층 구간 가격")
    floor_index = floor_price_index_by_type(filtered)
    if floor_index.empty:
        st.info("선택한 조건에 고층(16층 이상) 거래가 없어 가격 비율을 계산할 수 없습니다.")
    else:
        _show_chart(_floor_price_index_figure(filtered))
        st.caption(
            "현재 선택 기간에서 각 타입·층 구간 평균을 같은 타입의 고층(16층 이상) 평균과 비교했습니다. "
            "위 평균 막대와 동일한 가격 기준이며, 고층 거래가 없는 타입은 계산에서 제외됩니다."
        )

    st.subheader("시장 변동을 제외한 타입·층 가격 차이")
    adjusted_premium = adjusted_premium_summary(filtered)
    if adjusted_premium.empty:
        st.info("선택한 기간의 표본이 부족해 타입·층 가격 차이를 계산할 수 없습니다.")
    else:
        st.markdown("#### 타입별 가격 차이 · B타입 기준")
        _show_chart(
            _adjusted_premium_figure(
                filtered,
                "타입",
                summary=adjusted_premium,
            )
        )
        st.markdown("#### 층별 가격 차이 · 고층 기준")
        _show_chart(
            _adjusted_premium_figure(
                filtered,
                "층 구간",
                summary=adjusted_premium,
            )
        )
        st.caption(
            "같은 거래월의 시장가격 변동과 타입·층 차이를 함께 반영한 비교입니다. "
            "정확한 95% 범위와 거래 수는 막대에 마우스를 올리면 확인할 수 있습니다."
        )

    st.subheader("동·타입별 가격 비교")
    building_data = building_summary(filtered)
    building_data["평균가격(억원)"] = building_data["average_price"] / 100_000_000
    fig_building = px.bar(
        building_data,
        x="building",
        y="평균가격(억원)",
        color="plan_type",
        barmode="group",
        color_discrete_map=TYPE_COLORS,
        labels={"building": "동", "plan_type": "타입"},
        hover_data={"trades": True, "average_price": False},
    )
    _polish_chart(fig_building)
    _show_chart(fig_building)
    st.caption("막대는 선택 기간의 동·타입별 평균가격입니다. 동 정보가 공개되지 않은 거래는 '미공개'로 묶입니다.")

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
