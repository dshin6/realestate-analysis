# 한빛마을 한화꿈에그린 실거래 분석

국토교통부 아파트 매매 실거래가 상세 API를 이용해 `동탄시범한빛마을한화꿈에그린`의 타입·동·층별 가격을 비교하는 Streamlit 대시보드입니다.

## 처음 실행하기

Python 3.10 이상이 필요합니다.

현재 작업 PC에는 Windows용 `.venv`와 필요한 라이브러리가 이미 준비되어 있습니다. `.streamlit/secrets.toml`에 인증키를 입력한 뒤 PowerShell에서 바로 실행할 수 있습니다.

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

다른 PC에서 처음 설치할 때만 아래 과정을 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

`.streamlit/secrets.toml`을 열어 공공데이터포털 인증키를 입력한 뒤 실행합니다.

```bash
streamlit run app.py
```

Windows PowerShell에서는 가상환경 활성화 명령이 `.venv\Scripts\Activate.ps1`입니다.

## Streamlit Community Cloud 배포

이 저장소는 별도 Docker 이미지 없이 Streamlit Community Cloud에서 바로 실행할 수 있습니다.

1. [Streamlit Community Cloud](https://share.streamlit.io/)에 GitHub 계정으로 로그인합니다.
2. `Create app`에서 이 GitHub 저장소와 기본 브랜치를 선택합니다.
3. Main file path를 `app.py`로 지정하고 Python 3.12를 선택합니다.
4. Advanced settings의 Secrets에 다음 값을 입력합니다.

```toml
DATA_GO_KR_SERVICE_KEY = "공공데이터포털_인증키"
```

5. Deploy를 누릅니다.

배포 후 GitHub 기본 브랜치에 변경사항을 push하면 앱이 자동으로 갱신됩니다. 실제 인증키는 GitHub 파일이나 로그에 기록하지 않습니다.

저장소에는 빠른 첫 화면을 위한 검증된 공개 실거래 초기 데이터가 포함됩니다. 평상시에는 마지막 수집 월 이후 데이터만 추가로 조회하며, `최신 데이터 다시 받기`는 최근 3개월을 다시 조회해 실행 중 캐시를 갱신합니다.

## 데이터와 해석 주의사항

- 출처: 국토교통부 아파트 매매 실거래가 상세 Open API
- 금액 단위: API의 만원을 원으로 변환해 계산
- A/B/C 타입: 전용면적 84.80/84.73/84.79㎡를 기준으로 매핑
- 층 구간: 1층, 저층(2~5층), 중층(6~15층), 고층(16층 이상)
- 타입 프리미엄: 같은 연도의 단지 전체 중앙값으로 단순 보정한 지표이며 선호도를 직접 측정한 값이 아닙니다.
- 동 정보는 소유권 이전등기 완료 여부 등에 따라 비어 있을 수 있습니다.
- 취소 거래는 분석에서 제외합니다.
