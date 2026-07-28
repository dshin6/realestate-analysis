# Implementation Plan

마지막 갱신: 2026-07-29
상태: GitHub 및 Streamlit Community Cloud 배포 준비 중

## Goal

동탄시범한빛마을한화꿈에그린의 2007년 이후 실거래를 수집해 A/B/C 타입과 동·층 구간별 가격 차이를 보여주는 로컬 Streamlit 대시보드를 만든다.

## Current Work Mode

Standard로 진행한다. 작은 결과를 빠르게 확인하되 인증키는 로컬 secrets 파일에서만 관리한다.

## Constraints

- 첫 번째 사용 가능한 결과를 가능한 한 빨리 만든다.
- 검증되지 않은 기능과 추상화를 미리 추가하지 않는다.
- 실제로 확보 가능한 데이터를 기준으로 범위를 정한다.
- 분석에는 데이터 출처, 기준일, 계산식과 주요 가정을 남긴다.

## Tasks

### Task 1: 공공데이터 수집

상태: 구현 및 실데이터 검증 완료

- 국토교통부 아파트 매매 실거래가 상세 API의 월별 XML을 파싱한다.
- 동탄구 코드 `41597`로 과거 계약분까지 월별 조회한다.
- 취소 거래를 제외하고 가공한 거래를 로컬 캐시에 저장한다.

### Task 2: 타입·층 분석

상태: 구현 완료

- 전용면적 84.80/84.73/84.79㎡를 A/B/C로 분류한다.
- 동별 최고층 대비 1층·저층·중층·고층·최상층으로 분류한다.
- 연도별 타입 중앙값과 같은 연도 단지 중앙값 대비 단순 프리미엄을 계산한다.
- 최근 3년 같은 타입·층 구간 거래의 중앙값과 사분위 범위를 계산한다.

### Task 3: Streamlit 화면

상태: 구현 및 실데이터 렌더링 검증 완료

- 핵심 수치, 연도별 추세, 프리미엄, 층 구간, 매물가 비교와 거래표를 한 화면에 표시한다.
- 키 미설정, 로딩, 빈 결과와 오류 상태를 명시한다.
- 인증키는 로컬 secrets 파일에서만 읽는다.

### Task 4: GitHub 및 Streamlit Community Cloud 배포

상태: 진행 중

- 공개 GitHub 저장소 `realestate-analysis`에 소스와 문서를 게시한다.
- `.streamlit/secrets.toml`, 로컬 캐시, 가상환경과 실행 도구를 저장소에서 제외한다.
- Community Cloud에서 기본 브랜치의 `app.py`를 Python 3.12로 실행한다.
- `DATA_GO_KR_SERVICE_KEY`는 Community Cloud Secrets에만 등록한다.
- 기본 브랜치 push 시 자동 재배포되는 구성을 사용하고 Docker는 추가하지 않는다.

## Verification

- `python -m unittest discover -s tests -v`
- `streamlit run app.py`
- 실제 인증키 입력 후 최초 전체 수집, 차트와 거래표 렌더링 확인
- 공개 GitHub 저장소 파일 목록에서 인증키와 로컬 생성물을 제외했는지 확인
- Community Cloud 배포 URL에서 대시보드 로딩 확인

## Deferred Ideas

- 동 정보 미공개 거래를 보완할 합법적 출처 검토
- 사용자가 요청한 다른 한빛마을 단지 추가
