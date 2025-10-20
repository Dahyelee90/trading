import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yfinance as yf # yfinance는 이제 최상단에 import

st.title("Hello Dahye 👋")
st.markdown(
    """ 
    This is a playground for you to try Streamlit and have fun. 

    **There's :rainbow[so much] you can build!**
    
    We prepared a few examples for you to get started. Just 
    click on the buttons above and discover what you can do 
    with Streamlit. 
    """
)

if st.button("Send balloons!"):
    st.balloons()


st.set_page_config(page_title="Financial Ratios (US/KRX + Yahoo)", layout="wide")

# ---------------- Utils ----------------
def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = np.where((b == 0) | (~np.isfinite(b)), np.nan, a / b)
    return pd.Series(out, index=a.index)

def to_pydt(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    return ts.min().to_pydatetime(), ts.max().to_pydatetime()

def to_yahoo_symbol(ticker: str, exch_hint: str = None, is_us: bool = True) -> str:
    """
    미국: 그대로 반환 (예: AAPL)
    한국: 숫자 6자리면 .KS/.KQ 붙여 반환 (예: 005930.KS). exch_hint가 KS/KQ면 우선.
    """
    t = str(ticker).strip()
    if is_us:
        return t
    if t.endswith(".KS") or t.endswith(".KQ"):
        return t
    if exch_hint in ("KS", "KQ"):
        return f"{t}.{exch_hint}"
    if t.isdigit() and len(t) == 6:
        return f"{t}.KS"  # 기본 KOSPI 가정
    return t

@st.cache_data(show_spinner=False)
def load_csvs_generic(data_path: str, tickers_path: str):
    comp = pd.read_csv(data_path)
    tks  = pd.read_csv(tickers_path)
    comp.rename(columns={c: c.lower() for c in comp.columns}, inplace=True)
    # 날짜
    if "datadate" in comp.columns:
        comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
    elif "date" in comp.columns:
        comp["datadate"] = pd.to_datetime(comp["date"], errors="coerce")
    else:
        comp["datadate"] = pd.to_datetime(comp.iloc[:, 0], errors="coerce")
    # 티커 열
    if "tic" not in comp.columns:
        if "ticker" in comp.columns: comp.rename(columns={"ticker":"tic"}, inplace=True)
        elif "code" in comp.columns: comp.rename(columns={"code":"tic"}, inplace=True)
    return comp, tks

def compute_ratios(df):
    out = df.copy()

    # 시총
    if {"prcc_f","csho"}.issubset(out.columns):
        out["mkt_cap"] = pd.to_numeric(out["prcc_f"], errors="coerce") * pd.to_numeric(out["csho"], errors="coerce")

    # 유동성
    if {"act","lct"}.issubset(out.columns):
        out["current_ratio"] = safe_div(out["act"], out["lct"])
    if {"act","invt","lct"}.issubset(out.columns):
        out["quick_ratio"] = safe_div(pd.to_numeric(out["act"], errors="coerce")-pd.to_numeric(out["invt"], errors="coerce"), out["lct"])

    # 레버리지
    if {"lt","at"}.issubset(out.columns):
        out["debt_to_assets"] = safe_div(out["lt"], out["at"])
    if {"lt","seq"}.issubset(out.columns):
        out["debt_to_equity"] = safe_div(out["lt"], out["seq"])

    # 수익성
    if {"sale","cogs"}.issubset(out.columns):
        out["gross_margin"] = safe_div(pd.to_numeric(out["sale"], errors="coerce")-pd.to_numeric(out["cogs"], errors="coerce"), out["sale"])
    if {"oiadp","sale"}.issubset(out.columns):
        out["ebit_margin"] = safe_div(out["oiadp"], out["sale"])
    if {"ni","sale"}.issubset(out.columns):
        out["net_margin"] = safe_div(out["ni"], out["sale"])
    if {"ni","at"}.issubset(out.columns):
        out["roa"] = safe_div(out["ni"], out["at"])
    if {"ni","seq"}.issubset(out.columns):
        out["roe"] = safe_div(out["ni"], out["seq"])

    # 효율성
    if {"sale","at"}.issubset(out.columns):
        out["asset_turnover"] = safe_div(out["sale"], out["at"])

    # Altman Z (상장 제조사 가정; EBIT≈OIADP)
    req_z = {"wcap","re","oiadp","at","lt","prcc_f","csho","sale"}
    if req_z.issubset(out.columns):
        x1 = safe_div(out["wcap"], out["at"])
        x2 = safe_div(out["re"], out["at"])
        x3 = safe_div(out["oiadp"], out["at"])
        mve = pd.to_numeric(out["prcc_f"], errors="coerce") * pd.to_numeric(out["csho"], errors="coerce")
        x4 = safe_div(mve, out["lt"])
        x5 = safe_div(out["sale"], out["at"])
        # ✅ 'altman_z'를 'zscore'로 변경
        out["zscore"] = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
    else:
        # ✅ 'altman_z'를 'zscore'로 변경
        out["zscore"] = np.nan

    return out

# ---------------- Sidebar: Market & Data ----------------
st.sidebar.title("Controls")

market = st.sidebar.selectbox(
    "Market",
    ["US (Yahoo)", "Korea KRX (Yahoo)"],
    index=0,
    help="KRX는 6자리 종목코드 + .KS/.KQ (예: 005930.KS) 규칙을 사용합니다."
)

# CSV 경로(프로젝트에 맞춰 준비)
from pathlib import Path

if market.startswith("US"):
    # ✅ 미국 전체 티커 데이터 (us_tickers_yahoo.csv)
    tickers_path = Path("us_tickers_yahoo.csv")
    if not tickers_path.exists():
        st.warning("⚠️ us_tickers_yahoo.csv not found. Run scripts/build_us_tickers.py first.")
        # fallback: SP500
        candidates = [("sp500_data.csv","sp500_tickers.csv"), ("../sp500_data.csv","../sp500_tickers.csv")]
        comp, tks, chosen = None, None, None
        for dp, tp in candidates:
            if Path(dp).exists() and Path(tp).exists():
                comp, tks = load_csvs_generic(dp, tp)
                chosen = (dp, tp)
                break
        if comp is None:
            comp, tks = load_csvs_generic(candidates[-1][0], candidates[-1][1])
            chosen = candidates[-1]
    else:
        # 미국 전체 티커는 fundamentals CSV 없이 ticker만 쓰는 경우도 OK
        tks = pd.read_csv(tickers_path)
        chosen = ("(no fundamental CSV)", tickers_path.name)
        # 기본 fundamental 데이터는 sp500_data.csv fallback
        fund_path = Path("sp500_data.csv")
        if not fund_path.exists():
            fund_path = Path("../sp500_data.csv")
        comp = pd.read_csv(fund_path) if fund_path.exists() else pd.DataFrame(columns=["tic","datadate"])
        comp["datadate"] = pd.to_datetime(comp.get("datadate"), errors="coerce")

else:
    # ✅ 한국 KRX 데이터 (기존 그대로 유지)
    candidates = [("krx_data.csv","krx_tickers.csv"), ("../krx_data.csv","../krx_tickers.csv")]
    comp, tks, chosen = None, None, None
    for dp, tp in candidates:
        if Path(dp).exists() and Path(tp).exists():
            comp, tks = load_csvs_generic(dp, tp)
            chosen = (dp, tp)
            break
    if comp is None:
        comp, tks = load_csvs_generic(candidates[-1][0], candidates[-1][1])
        chosen = candidates[-1]

# ✅ 데이터 후처리
if not comp.empty:
    comp = comp.dropna(subset=["datadate"])

st.sidebar.caption(f"Loaded: `{chosen[0]}`, `{chosen[1]}`")

# 티커 열/보조정보
tkr_col = "tic" if "tic" in tks.columns else ("ticker" if "ticker" in tks.columns else tks.columns[0])
exch_map = dict(zip(tks[tkr_col], tks["exch"])) if "exch" in tks.columns else {}
yahoo_col_exists = "yahoo" in tks.columns

# ---------------- Sidebar: Tickers / Date / Ratios ----------------
# ✅ 수정된 부분: 티커 목록 선택 후보를 재무 데이터(comp)가 아닌, 전체 티커 목록(tks)에서 가져옵니다.
tkr_col_name = "tic" if "tic" in tks.columns else ("symbol" if "symbol" in tks.columns else tks.columns[0])
default_ticker = ("AAPL" if market.startswith("US") else "005930")

if not tks.empty and tkr_col_name in tks.columns:
    display_tickers = sorted(tks[tkr_col_name].dropna().astype(str).unique().tolist())
else:
    # tks 파일 로드가 실패했거나 데이터가 없는 경우 comp에서 가져오는 기존 로직 유지
    display_tickers = sorted(comp["tic"].dropna().astype(str).unique().tolist())

# 기존 로직과 충돌 방지를 위해 default_ticker가 display_tickers에 없으면 첫 번째 항목을 사용
if default_ticker not in display_tickers and display_tickers:
    default_ticker = display_tickers[0]


sel_tickers = st.sidebar.multiselect("Tickers", display_tickers, default=[default_ticker] if default_ticker else [])

date_min, date_max = to_pydt(comp["datadate"])
date_range = st.sidebar.slider(
    "Date range",
    min_value=date_min, max_value=date_max,
    value=(date_min, date_max),
    format="YYYY-MM-DD"
)
start_dt, end_dt = [pd.Timestamp(d) for d in date_range]


ratio_choices = [
    "current_ratio","quick_ratio",
    "debt_to_assets","debt_to_equity",
    "gross_margin","ebit_margin","net_margin",
    "asset_turnover","roa","roe",
    "zscore" # ✅ 'altman_z'를 'zscore'로 변경
]
plot_ratios = st.sidebar.multiselect(
    "Ratios to plot",
    ratio_choices,
    # ✅ 'altman_z'를 'zscore'로 변경
    default=["roa","roe","ebit_margin","zscore"]
)

# ---------------- (선택) Live price refresh via Yahoo ----------------
st.sidebar.subheader("Live price refresh (Yahoo)")
if st.sidebar.button("Update last price for selected tickers"):
    is_us = market.startswith("US")
    live_tics = sel_tickers or display_tickers

    for t in live_tics:
        # 야후 심볼 변환
        if is_us:
            ysym = t
        else:
            if yahoo_col_exists:
                yv = tks.loc[tks[tkr_col].astype(str)==str(t), "yahoo"]
                if len(yv) and pd.notna(yv.iloc[0]):
                    ysym = str(yv.iloc[0])
                else:
                    ysym = to_yahoo_symbol(t, exch_map.get(t), is_us=False)
            else:
                ysym = to_yahoo_symbol(t, exch_map.get(t), is_us=False)

        try:
            hist = yf.Ticker(ysym).history(period="5d")["Close"].dropna()
            if len(hist):
                last = float(hist.iloc[-1])
                comp.loc[comp["tic"].astype(str)==str(t), "prcc_f"] = last
        except Exception:
            pass

    st.success(f"Refreshed {len(live_tics)} tickers from Yahoo ({'US' if is_us else 'KRX'}).")

# ---------------- Compute & Filter ----------------
comp_ratios = compute_ratios(comp)

f = comp_ratios.copy()
if sel_tickers:
    f = f[f["tic"].astype(str).isin([str(x) for x in sel_tickers])]
f = f[(f["datadate"] >= start_dt) & (f["datadate"] <= end_dt)]
f = f.sort_values(["tic","datadate"])

# ---------------- Main ----------------
st.title("📊 Financial Ratios — US / KRX (Yahoo)")
st.caption("CSV fundamentals + Yahoo last price. Altman Z uses EBIT≈OIADP and latest price where available.")

# KPI (single ticker)
k1, k2, k3, k4 = st.columns(4)
if len(sel_tickers) == 1 and not f.empty:
    last = f[f["tic"].astype(str)==str(sel_tickers[0])].sort_values("datadate").tail(1)
    def kfmt(series): 
        return "–" if last.empty or series.isna().all() else f"{float(series.values[0]):.2f}"
    k1.metric("ROA", kfmt(last.get("roa", pd.Series([np.nan]))))
    k2.metric("ROE", kfmt(last.get("roe", pd.Series([np.nan]))))
    k3.metric("EBIT Margin", kfmt(last.get("ebit_margin", pd.Series([np.nan]))))
    # ✅ 'Altman Z'를 'Z-Score'로 변경하고 컬럼 이름도 'zscore'로 변경
    k4.metric("Z-Score", kfmt(last.get("zscore", pd.Series([np.nan]))))
else:
    k1.info("Select one ticker to show KPIs")

st.divider()

# Tabs: Chart/Table/Columns
tab_chart, tab_table, tab_cols = st.tabs(["📈 Chart", "📋 Table", "ℹ️ Columns"])

with tab_chart:
    chart_mode = st.radio(
        "Chart mode",
        ["By ratio (multi-ticker)", "By ticker (multi-ratio)"],
        index=0, horizontal=True
    )
    normalize = st.checkbox("Normalize series (z-score per chart)", value=False)

    if chart_mode == "By ratio (multi-ticker)":
        if not plot_ratios:
            st.info("Select at least one ratio in the sidebar.")
        else:
            for col in [c for c in plot_ratios if c in f.columns]:
                st.subheader(col)
                wide = f.pivot_table(values=col, index="datadate", columns="tic").sort_index()
                if normalize and not wide.empty:
                    wide = (wide - wide.mean()) / wide.std(ddof=0)
                st.line_chart(wide, height=260, use_container_width=True)
    else:
        if not sel_tickers:
            st.info("Select at least one ticker.")
        else:
            for tic in sel_tickers:
                df_t = f[f["tic"].astype(str)==str(tic)].set_index("datadate").sort_index()
                cols = [c for c in plot_ratios if c in df_t.columns]
                if not cols:
                    continue
                st.subheader(f"{tic} — {', '.join(cols)}")
                plot_df = df_t[cols].copy()
                if normalize and not plot_df.empty:
                    plot_df = (plot_df - plot_df.mean()) / plot_df.std(ddof=0)
                st.line_chart(plot_df, height=300, use_container_width=True)
    
    # ----------------------------------------------------
    # 누적 주가 차트
    # ----------------------------------------------------
    st.divider()
    st.subheader("Price Performance (Normalized)")

    if sel_tickers:
        is_us = market.startswith("US")
        price_data = {}
        
        # 날짜 범위 문자열로 변환
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')

        for t in sel_tickers:
            # 야후 심볼 변환 로직 재사용
            ysym = t
            if not is_us:
                if yahoo_col_exists and tks is not None:
                    yv = tks.loc[tks[tkr_col].astype(str)==str(t), "yahoo"]
                    if len(yv) and pd.notna(yv.iloc[0]):
                        ysym = str(yv.iloc[0])
                    else:
                        ysym = to_yahoo_symbol(t, exch_map.get(t), is_us=False)
                else:
                    ysym = to_yahoo_symbol(t, exch_map.get(t), is_us=False)

            try:
                # Yahoo Finance에서 지정 기간의 종가 데이터(Close)를 가져옵니다.
                hist = yf.Ticker(ysym).history(start=start_str, end=end_str, auto_adjust=True)["Close"].dropna()
                
                if len(hist) > 0:
                    # ✅ 누적 주가 성과 (정규화): 첫 번째 가격을 100으로 설정
                    normalized_hist = (hist / hist.iloc[0]) * 100
                    price_data[t] = normalized_hist.rename(t)
            except Exception:
                # 데이터가 없거나 에러가 발생하면 경고 표시
                st.warning(f"⚠️ Could not fetch historical price data for {t} ({ysym}) from Yahoo.")
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

# ----------------------------------------------------
# ----------------------------------------------------


with tab_table:
    # ✅ 'altman_z'를 'zscore'로 변경
    show_cols = ["tic","datadate","prcc_f","mkt_cap"] + [c for c in ratio_choices if c in f.columns]
    st.dataframe(f[show_cols].sort_values(["tic","datadate"]), use_container_width=True, height=480)
    st.download_button(
        "⬇️ Download filtered CSV",
        f[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="ratios_filtered.csv",
        mime="text/csv"
    )

with tab_cols:
    st.write(sorted(comp_ratios.columns.tolist()))
    st.caption("Tip: For KRX, prepare krx_data.csv / krx_tickers.csv. Optional columns: exch(KS/KQ), yahoo(005930.KS).")

st.caption("© Dahye Lee — Streamlit demo (US/KRX Yahoo integration)")
