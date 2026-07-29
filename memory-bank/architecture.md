# Architecture

마지막 갱신: 2026-07-29
상태: 첫 버전 구현

## Current Scope

단일 Streamlit 화면에서 대상 단지의 실거래 수집, 필터링, 가격 분석과 표 표시를 제공한다.

## System Context

사용자 → 로컬 또는 Streamlit Community Cloud 앱 → 국토교통부 Open API → 실행 환경의 JSON 캐시 → pandas 분석 → Plotly 차트와 거래표 순서로 동작한다.

## Components

- `app.py`: 화면 상태, 필터, 차트, 매물가 비교와 출처·주의사항 표시
- `realestate_analysis/config.py`: 대상 단지 식별정보와 면적 타입 설정
- `realestate_analysis/api.py`: XML 파싱, 월별 API 수집, 대상 단지 필터와 캐시
- `realestate_analysis/analysis.py`: 타입·층 분류, 연도별 요약과 적정가격 비교
- `tests/`: 외부 API 없이 파서와 계산 규칙을 확인하는 단위 테스트

## Data Flow

1. `.streamlit/secrets.toml`에서 인증키를 읽는다.
2. 2007년 3월부터 현재 월까지 동탄구 법정동코드 `41597`로 월별 API를 조회한다.
3. 첫 실행은 `data/seed/trades.json`의 검증된 공개 초기 데이터를 읽는다.
4. 초기 데이터 이후의 누락 월만 API로 조회하며, 수동 새로고침은 최근 3개월을 다시 조회한다.
5. 취소 거래와 다른 단지를 제외하고 중복을 제거해 실행 환경의 `data/cache/trades.json`에 저장한다.
6. 전용면적을 A/B/C 타입으로, 실제 층을 고정 층 구간으로 분류한다.
7. 연도별 중앙값·프리미엄과 최근 3년 유사 거래 범위를 계산해 화면에 표시한다.

Community Cloud에서는 GitHub 기본 브랜치의 `app.py`와 `requirements.txt`를 사용한다. 인증키는 Community Cloud Secrets에 주입하며 저장소에는 포함하지 않는다. 기본 브랜치 변경 시 Streamlit이 앱을 자동으로 다시 배포한다.

## Data Model

- 거래 한 건: 계약일, 가격(원), 전용면적(㎡), 타입, 동, 실제 층, 층 구간, 취소 여부
- 분석 시간 단위: 전체 거래는 계약일, 장기 추세는 연도
- 가격 대표값: 이상치 영향이 비교적 작은 중앙값
- 단순 프리미엄: 개별 가격 ÷ 같은 연도 단지 중앙값의 타입별 중앙값 - 1

## Boundaries

- 원본 데이터와 가공 데이터를 구분한다.
- 분석 로직과 화면 표시를 분리한다.
- 사실, 계산 결과와 사용자 가정을 구분한다.
- API 키는 코드·문서·로그·캐시에 저장하지 않는다.

## Testing and Validation

- XML 필드·금액 단위·오류 응답 파싱 단위 테스트
- 면적 타입, 층 구간, 프리미엄과 유사 거래 계산 단위 테스트
- 키 미설정 상태 및 실제 키 설정 상태의 Streamlit 스모크 테스트

## Known Limitations

- Community Cloud 인스턴스가 재시작되면 실행 캐시는 사라지지만 저장소의 초기 데이터로 즉시 화면을 복원한다.
- 초기 데이터 기준 월 이후의 변경분만 자동 수집하므로 오래된 거래의 사후 정정은 최근 3개월 수동 새로고침 범위 밖일 수 있다.
- API 제공 필드와 행정구역 코드는 실제 응답 검증 뒤 조정될 수 있다.
- 동 미공개 거래는 동별 가격 비교에 제한이 있지만 층 구간 분석에는 포함된다.
