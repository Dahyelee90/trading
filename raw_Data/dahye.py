import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf # yfinance를 사용하여 데이터 로드

st.title("Hello Dahye 👋")
st.markdown(
    """ 
    Streamlit Financial Analyzer (Yahoo Finance Only)
    
    이 앱은 CSV 파일 없이 **Yahoo Finance에서 실시간으로 가져온 최신 데이터**를 사용합니다.
    (ROE, ROA, Z-Score 등 과거 시계열 데이터는 Yahoo Finance API에서 제공되지 않으므로, 
    **최신 시점의 재무 정보만 KPI와 테이블에 표시**됩니다.)
    """
)

if st.button("Send balloons!"):
    st.balloons()

st.set_page_config(page_title="Financial Ratios (Yahoo Only)", layout="wide")

# ---------------- Utils ----------------
def safe_div(a, b):
    # 안전한 나누기 함수 유지
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = np.where((b == 0) | (~np.isfinite(b)), np.nan, a / b)
    # Series 대신 단일 값으로 반환될 수 있음
    return out[0] if isinstance(out, np.ndarray) and out.size == 1 else out

# ✅ Altman Z-Score 계산 함수 강화
def calculate_altman_zscore(balance_sheet: pd.DataFrame, income_stmt: pd.DataFrame, market_cap: float) -> tuple:
    """
    Altman Z-Score (제조업 공개 기업용)을 계산합니다.
    Z = 1.2*T1 + 1.4*T2 + 3.3*T3 + 0.6*T4 + 1.0*T5
    Returns: Z-Score (float) and a dictionary of raw inputs
    """
    
    # 헬퍼 함수: 재무제표에서 항목을 안전하게 가져오고 None이면 NaN 반환
    def get_value(series, keys):
        for key in keys:
            val = series.get(key)
            if pd.notna(val) and val is not None:
                return val
        return np.nan
        
    raw_inputs = {
        'total_assets': np.nan,
        'working_capital': np.nan,
        'retained_earnings': np.nan,
        'ebit': np.nan,
        'total_liabilities': np.nan,
        'total_revenue': np.nan
    }
    
    if balance_sheet.empty or income_stmt.empty or market_cap is None or market_cap <= 0:
        return np.nan, raw_inputs

    # 최신 데이터는 첫 번째 컬럼에 위치
    bs = balance_sheet.iloc[:, 0]
    is_ = income_stmt.iloc[:, 0]

    try:
        # T1: Working Capital / Total Assets
        current_assets = get_value(bs, ['TotalCurrentAssets', 'CurrentAssets'])
        current_liabilities = get_value(bs, ['TotalCurrentLiabilities', 'CurrentLiabilities'])
        total_assets = get_value(bs, ['TotalAssets'])
        
        working_capital = safe_div(current_assets, 1) - safe_div(current_liabilities, 1)
        raw_inputs['working_capital'] = working_capital
        raw_inputs['total_assets'] = total_assets
        T1 = safe_div(working_capital, total_assets)

        # T2: Retained Earnings / Total Assets
        retained_earnings = get_value(bs, ['RetainedEarnings', 'RetainedEarningsTotal'])
        raw_inputs['retained_earnings'] = retained_earnings
        T2 = safe_div(retained_earnings, total_assets)

        # T3: EBIT / Total Assets
        ebit = get_value(is_, ['Ebit'])
        raw_inputs['ebit'] = ebit
        T3 = safe_div(ebit, total_assets)

        # T4: Market Value of Equity / Total Liabilities
        # Total Liabilities는 Total Assets - StockholdersEquity 또는 TotalLiabilitiesNetMinorityInterest
        total_liabilities = get_value(bs, ['TotalLiabilitiesNetMinorityInterest', 'TotalLiabilities'])
        
        # 만약 Total Liabilities를 못 찾으면, 자산-자기자본으로 대체 계산
        if pd.isna(total_liabilities) and total_assets is not np.nan:
             total_equity = get_value(bs, ['StockholdersEquity'])
             total_liabilities = safe_div(total_assets, 1) - safe_div(total_equity, 1)
        
        raw_inputs['total_liabilities'] = total_liabilities
        T4 = safe_div(market_cap, total_liabilities)

        # T5: Sales / Total Assets
        sales = get_value(is_, ['TotalRevenue', 'Sales'])
        raw_inputs['total_revenue'] = sales
        T5 = safe_div(sales, total_assets)
        
        # Z-Score 계산 (T-score 중 하나라도 NaN이면 NaN 반환)
        T_scores = [T1, T2, T3, T4, T5]
        
        # 모든 T-score가 숫자인지 확인
        if any(pd.isna(T_scores)):
             return np.nan, raw_inputs

        # Z = 1.2*T1 + 1.4*T2 + 3.3*T3 + 0.6*T4 + 1.0*T5
        Z = (1.2 * T1) + (1.4 * T2) + (3.3 * T3) + (0.6 * T4) + (1.0 * T5)
        
        return Z, raw_inputs
        
    except Exception as e:
        # print(f"Z-Score calculation failed: {e}")
        return np.nan, raw_inputs

# ✅ to_yahoo_symbol 함수는 이제 US 마켓만 처리하도록 간소화됩니다.
def to_yahoo_symbol(ticker: str, market: str) -> str:
    t = str(ticker).strip()
    return t # US 마켓은 티커가 그대로 Yahoo 심볼입니다.


# Yahoo Finance info 딕셔너리에서 필요한 재무 비율과 Z-Score 계산
@st.cache_data(ttl=3600, show_spinner="Fetching latest financial data and calculating Z-Score...")
def fetch_yahoo_ratios(ticker_list, market):
    ratios = {}
    
    for t in ticker_list:
        # 야후 심볼로 변환 (이제 US만 해당)
        yahoo_sym = to_yahoo_symbol(t, market)
        
        try:
            ticker = yf.Ticker(yahoo_sym)
            info = ticker.info
            
            # 1. Ticker Info (Ratios)
            data = {
                "roa": info.get("returnOnAssets"),
                "roe": info.get("returnOnEquity"),
                "payout_ratio": info.get("payoutRatio"),
                "dividend_yield": info.get("dividendYield"),
                "forward_pe": info.get("forwardPE"),
                "market_cap": info.get("marketCap"),
                "last_price": info.get("currentPrice"),
                "ebit": info.get("ebit"), 
                "ebitda": info.get("ebitda"), 
                "datadate": datetime.now().strftime('%Y-%m-%d'), 
                "yahoo_sym": yahoo_sym 
            }

            # 2. Z-Score 계산을 위한 재무제표 데이터 로드 (최신 분기)
            market_cap = data.get("market_cap")
            
            # quarterly_balance_sheet와 quarterly_income_stmt를 사용
            bs = ticker.quarterly_balance_sheet 
            is_ = ticker.quarterly_income_stmt

            zscore, raw_inputs = calculate_altman_zscore(bs, is_, market_cap)
            data["zscore"] = zscore # ✅ Z-Score 계산 값 할당
            
            # ✅ Z-Score 계산에 사용된 Raw 데이터 항목 추가
            data.update({
                'Z_WC_Raw': raw_inputs['working_capital'],
                'Z_RE_Raw': raw_inputs['retained_earnings'],
                'Z_EBIT_Raw': raw_inputs['ebit'],
                'Z_TL_Raw': raw_inputs['total_liabilities'],
                'Z_TA_Raw': raw_inputs['total_assets'],
                'Z_Sales_Raw': raw_inputs['total_revenue']
            })
            
            ratios[t] = data
            
        except Exception as e:
            # 에러 발생 시 Nan 값 할당
            ratios[t] = {"roa": np.nan, "roe": np.nan, "zscore": np.nan, "last_price": np.nan, "market_cap": np.nan, "ebit": np.nan, "ebitda": np.nan, "datadate": datetime.now().strftime('%Y-%m-%d'), "yahoo_sym": yahoo_sym,
                        'Z_WC_Raw': np.nan, 'Z_RE_Raw': np.nan, 'Z_EBIT_Raw': np.nan, 'Z_TL_Raw': np.nan, 'Z_TA_Raw': np.nan, 'Z_Sales_Raw': np.nan}
            
    df = pd.DataFrame.from_dict(ratios, orient='index')
    df = df.rename_axis('tic').reset_index()
    return df

# ---------------- Sidebar: Market & Data ----------------
st.sidebar.title("Controls")

market = st.sidebar.selectbox(
    "Market",
    ["US (Yahoo)"], 
    index=0
)

# Codespaces의 us_tickers_yahoo.csv를 이용해 티커 목록 로드
# us_tickers_yahoo.csv와 raw_Data 폴더를 탐색하여 로드
def load_all_tickers(current_market):
    # 티커 파일 경로 옵션 (CSV 또는 TXT)
    csv_paths = [Path("us_tickers_yahoo.csv"), Path("raw_Data/us_tickers_yahoo.csv")]
    txt_paths = [Path("nasdaqlisted.txt"), Path("raw_Data/nasdaqlisted.txt")] 

    default_list = ["AAPL", "MSFT", "GOOG", "TSLA", "RGTI"]
    
    # 1. CSV 파일 로드 시도
    for p in csv_paths:
        if p.exists():
            return pd.read_csv(p), p.name
    
    # 2. nasdaqlisted.txt 파일 로드 시도 (파싱 로직 추가)
    for p in txt_paths:
        if p.exists():
            try:
                # nasdaqlisted.txt 파일 형식에 맞춰 읽기: '|' 구분자, 첫 행 헤더, 마지막 행은 푸터
                # encoding='latin1' 사용 (대부분의 NASDAQ 파일은 UTF-8이 아닌 latin1 인코딩 사용)
                df = pd.read_csv(
                    p, 
                    sep='|', 
                    skiprows=0, # 헤더를 포함
                    skipfooter=1, # 마지막 푸터 행 제거
                    engine='python',
                    encoding='latin1'
                )
                
                # 심볼 컬럼을 'Symbol' 또는 첫 번째 컬럼으로 설정
                if 'Symbol' in df.columns:
                    df = df.rename(columns={'Symbol': 'symbol'})
                elif df.columns.size > 0:
                    df = df.rename(columns={df.columns[0]: 'symbol'})

                # ✅ 수정: 불필요한 컬럼 제거 (ETF, 테스트 이슈) 로직을 주석 처리하여 필터링 해제
                # df = df[~df['ETF'].astype(str).str.upper().str.contains('Y', na=False)]
                # df = df[~df['Test Issue'].astype(str).str.upper().str.contains('Y', na=False)]

                # Yahoo Finance에서 문제가 생기는 특수 문자 제거 (예: 점(.)을 하이픈(-)으로)
                df['symbol'] = df['symbol'].astype(str).str.replace('.', '-', regex=False)
                
                # 필요한 컬럼만 남기고 반환
                return df[['symbol']], p.name
                
            except Exception as e:
                st.warning(f"⚠️ Error reading/parsing {p.name}: {e}")
                
    # 3. 파일이 없을 경우 기본 리스트와 플래그를 반환
    return default_list, "(Default Tickers Used)" 

tks_data, tks_file = load_all_tickers(market)

# ✅ 수정: tks_data가 DataFrame인지 리스트인지 확인하고 tks_data를 tks로 할당
if isinstance(tks_data, pd.DataFrame):
    tks = tks_data
    tkr_col_name = "symbol" if "symbol" in tks.columns else tks.columns[0]
else:
    # default_list가 반환된 경우, display_tickers를 바로 설정하기 위해 tks를 빈 DataFrame으로 설정
    tks = pd.DataFrame()
    display_tickers_default = tks_data
    tkr_col_name = "symbol" # Placeholder


st.sidebar.caption(f"Loaded Tickers: `{tks_file}`")

# ---------------- Sidebar: Tickers / Date / Ratios ----------------

# 티커 선택 후보: tks DataFrame에서 가져옴
if not tks.empty and tkr_col_name in tks.columns:
    display_tickers = sorted(tks[tkr_col_name].dropna().astype(str).unique().tolist())
else:
    # ✅ 수정: 파일 없을 경우 미리 정의된 display_tickers_default 리스트 사용
    display_tickers = display_tickers_default


# ✅ 수정: 마켓에 따른 기본 티커 설정 (US만 남음)
default_ticker = "AAPL"

if default_ticker not in display_tickers and display_tickers:
    default_ticker = display_tickers[0]

sel_tickers = st.sidebar.multiselect("Tickers", display_tickers, default=[default_ticker] if default_ticker else [])

# 야후 API는 시계열 재무 데이터가 없으므로, 날짜 범위 슬라이더는 주가 차트용으로만 사용
date_min = datetime(2010, 1, 1).date()
date_max = datetime.now().date()

# ✅ 추가: 5년 전 날짜 계산
five_years_ago = (datetime.now() - timedelta(days=5 * 365)).date()
if five_years_ago < date_min:
    five_years_ago = date_min # 최소 날짜보다 이전이면 최소 날짜 사용

date_range = st.sidebar.slider(
    "Date range (Price Chart Only)",
    min_value=date_min, 
    max_value=date_max,
    value=(five_years_ago, date_max), # ✅ 수정: 5년 전을 기본 시작 값으로 설정
    format="YYYY-MM-DD"
)
start_dt, end_dt = [pd.Timestamp(d) for d in date_range]

# ✅ Z-Score 원재료 항목을 ratio_choices에 추가 (테이블 제거 후에도 데이터는 존재)
ratio_choices = [
    "roa", "roe", "ebit", "ebitda", "zscore",
    "Z_WC_Raw", "Z_RE_Raw", "Z_EBIT_Raw", "Z_TL_Raw", "Z_TA_Raw", "Z_Sales_Raw"
]
# ✅ plot_ratios 멀티셀렉트도 제거합니다.
# plot_ratios = st.sidebar.multiselect(
#     "Ratios to show (Latest Value)",
#     ratio_choices,
#     default=["roa", "roe", "zscore", "Z_WC_Raw", "Z_RE_Raw", "Z_EBIT_Raw", "Z_TL_Raw", "Z_TA_Raw", "Z_Sales_Raw"]
# )


# ---------------- Live data fetch & Compute ----------------

# 선택된 티커의 최신 재무 데이터 가져오기
comp_ratios = fetch_yahoo_ratios(sel_tickers, market) 

# 'f' DataFrame을 comp_ratios로 설정
if sel_tickers and not comp_ratios.empty:
    # comp_ratios DataFrame에 있는 티커만 필터링하여 loc 에러 방지
    valid_tickers = [t for t in sel_tickers if t in comp_ratios['tic'].values]
    if valid_tickers:
        f = comp_ratios.set_index('tic').loc[valid_tickers].reset_index().copy()
    else:
        f = pd.DataFrame(columns=['tic', 'datadate', 'last_price', 'market_cap'] + ratio_choices)
else:
    f = pd.DataFrame(columns=['tic', 'datadate', 'last_price', 'market_cap'] + ratio_choices)


# ---------------- Main ----------------
st.title("📊 Financial Ratios — Live Yahoo Data Only (US Market)")
st.caption("최신 재무 비율 (ROE, ROA 등)은 Yahoo Finance의 요약 정보를 기반으로 합니다.")

# KPI (single ticker)
k1, k2, k3, k4 = st.columns(4)
if len(sel_tickers) == 1 and not f.empty and f.iloc[0]['tic'] in sel_tickers: # f.iloc[0]이 유효한지 확인
    last = f.iloc[0]
    
    def kfmt(val, is_pct=False, is_zscore=False): 
        if pd.isna(val): return "–"
        if is_zscore: return f"{val:.3f}"
        if is_pct: return f"{val * 100:.2f}%"
        
        # 큰 값은 보기 쉽게 포맷팅 (예: 백만/억 단위)
        if abs(val) >= 1e9: # 10억 이상
            return f"{val/1e9:.2f}B"
        if abs(val) >= 1e6: # 100만 이상
            return f"{val/1e6:.2f}M"
            
        return f"{val:.2f}"
    
    k1.metric("ROE", kfmt(last.get("roe"), is_pct=True))
    k2.metric("ROA", kfmt(last.get("roa"), is_pct=True))
    k3.metric("Market Cap", kfmt(last.get("market_cap"))) # Market Cap을 k3에 표시
    k4.metric("Z-Score", kfmt(last.get("zscore"), is_zscore=True)) # ✅ Z-Score에 is_zscore=True 전달
else:
    k1.info("Select one ticker to show KPIs")

st.divider()

# Tabs: Chart/Columns (테이블 탭 제거)
tab_chart, tab_cols = st.tabs(["📈 Price Chart", "ℹ️ Columns"]) # ✅ 탭 목록에서 tab_table 제거

with tab_chart:
    st.subheader("Price Performance (Normalized)")

    if sel_tickers:
        price_data = {}
        
        # 날짜 범위 문자열로 변환
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')

        for t in sel_tickers:
            # ✅ 수정: 실제 야후 심볼을 사용 (comp_ratios에 저장된 yahoo_sym 컬럼 활용)
            # 유효한 yahoo_sym을 찾지 못하면 루프 건너뛰기
            if not comp_ratios.empty and t in comp_ratios['tic'].values:
                yahoo_sym = comp_ratios[comp_ratios['tic'] == t]['yahoo_sym'].iloc[0]
            else:
                continue # 티커 데이터가 없으면 다음 티커로 넘어감
            
            try:
                # Yahoo Finance에서 지정 기간의 종가 데이터(Close)를 가져옵니다.
                # start와 end 매개변수를 사용하여 슬라이더에서 선택된 기간을 적용합니다.
                hist = yf.Ticker(yahoo_sym).history(start=start_str, end=end_str, auto_adjust=True)["Close"].dropna()
                
                if len(hist) > 0:
                    # ✅ 누적 주가 성과 (정규화): 첫 번째 가격을 100으로 설정
                    normalized_hist = (hist / hist.iloc[0]) * 100
                    price_data[t] = normalized_hist.rename(t)
            except Exception:
                st.warning(f"⚠️ Could not fetch historical price data for {t} ({yahoo_sym}) from Yahoo.")
                pass

        if price_data:
            # 모든 정규화된 주가 데이터를 하나의 DataFrame으로 합칩니다.
            price_df = pd.DataFrame(price_data).sort_index()
            
            # 주가 차트 표시
            st.line_chart(price_df, height=350, use_container_width=True)
            st.caption("Price is normalized (indexed to 100) on the first available date.")
        else:
            st.info("No historical price data fetched for the selected tickers/date range.")

    else:
        st.info("Select at least one ticker to plot price performance.")


# with tab_table:  # ✅ tab_table 블록 전체 삭제됨


with tab_cols:
    st.write(sorted(comp_ratios.columns.tolist()))
    st.caption("Tip: 이 앱은 Yahoo Finance의 최신 요약 정보만 사용합니다.")

st.caption("© Dahye Lee — Streamlit demo (Yahoo integration)")
