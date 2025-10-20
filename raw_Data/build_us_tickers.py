import pandas as pd
import requests
from io import StringIO
from pathlib import Path

NASDQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"  # NYSE/AMEX 등

OUT_CSV  = Path("us_tickers_yahoo.csv")

def fetch_table(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    txt = resp.text.strip()
    # 파일 마지막 줄은 "File Creation Time: ..." → 제거
    lines = [ln for ln in txt.splitlines() if not ln.startswith("File Creation Time")]
    txt2 = "\n".join(lines)
    df = pd.read_csv(StringIO(txt2), sep="|")
    # 마지막 빈 컬럼 제거 (보통 '|'로 끝나서 NaN열 생김)
    df = df.loc[:, ~df.columns.str.fullmatch(r"Unnamed:.*")]
    return df

def normalize_nasdaq(df: pd.DataFrame) -> pd.DataFrame:
    # nasdaqlisted.txt 스키마: Symbol, Security Name, Market Category, Test Issue, Financial Status, Round Lot Size, ETF, NextShares
    keep = df.rename(columns={"Symbol":"symbol","Security Name":"name","ETF":"is_etf","Test Issue":"is_test"})
    # ETF/테스트 이슈 제외
    if "is_etf" in keep:  keep = keep[keep["is_etf"].fillna("N")=="N"]
    if "is_test" in keep: keep = keep[keep["is_test"].fillna("N")=="N"]
    keep["exchange"] = "NASDAQ"
    return keep[["symbol","name","exchange"]]

def normalize_other(df: pd.DataFrame) -> pd.DataFrame:
    # otherlisted.txt 스키마: ACT Symbol, Security Name, Exchange, CQS Symbol, ETF, Round Lot Size, Test Issue, NASDAQ Symbol
    colmap = {
        "ACT Symbol":"symbol",
        "Security Name":"name",
        "Exchange":"exchange",
        "ETF":"is_etf",
        "Test Issue":"is_test",
    }
    keep = df.rename(columns=colmap)
    if "is_etf" in keep:  keep = keep[keep["is_etf"].fillna("N")=="N"]
    if "is_test" in keep: keep = keep[keep["is_test"].fillna("N")=="N"]
    return keep[["symbol","name","exchange"]]

def to_yahoo_symbol(sym: str) -> str:
    """야후 규칙: 점(.) → 하이픈(-). (예: BRK.B → BRK-B)"""
    sym = str(sym).strip()
    return sym.replace(".", "-")

def main():
    nas = normalize_nasdaq(fetch_table(NASDQ_URL))
    oth = normalize_other(fetch_table(OTHER_URL))
    allu = pd.concat([nas, oth], ignore_index=True).drop_duplicates(subset=["symbol"])
    # 야후용 심볼 추가
    allu["yahoo"] = allu["symbol"].apply(to_yahoo_symbol)
    # 보조 컬럼 정리
    allu = allu.sort_values(["exchange","symbol"]).reset_index(drop=True)
    OUT_CSV.write_text(allu.to_csv(index=False, encoding="utf-8"))
    print(f"Saved {len(allu):,} symbols → {OUT_CSV}")

if __name__ == "__main__":
    main()