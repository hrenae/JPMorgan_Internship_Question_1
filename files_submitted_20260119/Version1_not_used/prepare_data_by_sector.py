"""
prepare_data_by_sector.py
-------------------------

1. 从 Wikipedia 抓取 S&P 500 成分股及其 GICS Sector。
2. 对每个行业 (sector)，下载各公司季度财报（BS/IS/CF），
   映射到简化的 balance-sheet / flow 变量。
3. 对每个公司构建 LSTM 序列样本 (长度 SEQ_LEN，预测下一期 y_{t+1})。
4. 对每个行业，合并样本，随机打乱后拆成 train / test，
   并将 (X_train, y_train, X_test, y_test) + meta_train/meta_test
   保存到本地目录 DATA_DIR。
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO

# ----------------- 全局配置 -----------------

DATA_DIR = "data_by_sector"

FREQUENCY = "quarterly"       # "quarterly" or "annual"
SEQ_LEN = 2                   # LSTM 序列长度
MIN_PERIODS = SEQ_LEN + 1     # 每个公司至少需要的报表期数
MIN_TICKERS_PER_SECTOR = 50   # 每个行业至少 50 家公司（在 S&P500 成分中）
TRAIN_RATIO = 0.7             # 行业内部 train/test 比例

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Balance-sheet 目标变量
Y_COLS = ["C", "AR", "Inv", "K", "AP", "D", "E"]
# Flow 特征变量
X_COLS = [
    "S", "COGS", "OPEX", "Dep", "I",
    "NewDebt", "Repay", "EquityIssues", "Dividends"
]


# ----------------- 工具函数 -----------------

def sanitize_sector_name(sector: str) -> str:
    """把行业名变成文件名友好的形式"""
    s = sector.replace(" ", "_").replace("&", "and").replace("/", "-")
    return s


def fetch_sp500_constituents():
    """
    从 Wikipedia 抓 S&P500 成分股表，返回包含
    'Symbol' 和 'GICS Sector' 的 DataFrame。
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # 1. 下载页面
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    
    # 2. 尝试通过 id="constituents" 精确查找表格
    try:
        tables = pd.read_html(StringIO(r.text), attrs={"id": "constituents"})
        table = tables[0]
    except Exception:
        # 如果 id 查找失败，回退到遍历查找
        tables = pd.read_html(StringIO(r.text))
        table = None
        for t in tables:
            if "Symbol" in t.columns and "GICS Sector" in t.columns:
                table = t
                break
        
        if table is None:
            # 打印调试信息帮助定位
            print(f"Found {len(tables)} tables, but none had 'Symbol' and 'GICS Sector'.")
            for i, t in enumerate(tables):
                print(f"Table {i} columns: {t.columns.tolist()}")
            raise ValueError("Could not find the S&P 500 table with expected columns.")

    # 3. 提取数据
    df = table[["Symbol", "GICS Sector"]].copy()

    # Wikipedia 上 BRK.B 等要改成 yfinance 风格 BRK-B
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df


def pick_first_available(df, candidates, default=np.nan,
                         suffixes=("", "_BS", "_IS", "_CF")):
    """
    在 df 里依次寻找候选列名（尝试 _BS/_IS/_CF 后缀），找到就返回该列 Series。
    如果都找不到，返回标量 default。
    """
    for base in candidates:
        for suf in suffixes:
            col = base + suf
            if col in df.columns:
                return df[col]
    return default


def get_company_data(ticker_symbol,
                     frequency="quarterly",
                     fillna_with=0,
                     verbose=True,
                     show_columns=False):
    """
    下载单个公司的三大报表，并映射成模型需要的变量：
    Stocks y_t: C, AR, Inv, K, AP, D, E
    Flows  x_t: S, COGS, OPEX, Dep, I, NewDebt, Repay, EquityIssues, Dividends
    Meta       : Total_Assets_Actual, Total_Liab_Actual, NI
    """
    if verbose:
        print(f"\n=== Downloading {frequency} data for {ticker_symbol} ===")

    ticker = yf.Ticker(ticker_symbol)

    if frequency == "annual":
        bs = ticker.balance_sheet.T
        is_ = ticker.financials.T
        cf = ticker.cashflow.T
    elif frequency == "quarterly":
        bs = ticker.quarterly_balance_sheet.T
        is_ = ticker.quarterly_financials.T
        cf = ticker.quarterly_cashflow.T
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    if bs.empty or is_.empty or cf.empty:
        raise ValueError(f"Empty BS/IS/CF for {ticker_symbol}")

    bs = bs.sort_index(ascending=True)
    is_ = is_.sort_index(ascending=True)
    cf = cf.sort_index(ascending=True)

    df_raw = bs.join(is_, lsuffix="_BS", rsuffix="_IS").join(cf, rsuffix="_CF")

    if show_columns:
        print(f"\nColumns for {ticker_symbol}:")
        print(df_raw.columns.tolist())

    data = pd.DataFrame(index=df_raw.index)

    # ---------- Stocks ----------
    data["C"] = (
        pick_first_available(
            df_raw,
            [
                "Cash And Cash Equivalents",
                "CashAndCashEquivalents",
                "Cash",
                "Cash Only",
                "Cash And Short Term Investments",
                "CashAndShortTermInvestments",
                "Cash Cash Equivalents And Short Term Investments",
                "CashCashEquivalentsAndShortTermInvestments",
            ],
        )
        + pick_first_available(
            df_raw,
            ["Short Term Investments", "ShortTermInvestments"],
            default=0,
        )
    )

    data["AR"] = pick_first_available(
        df_raw,
        [
            "Accounts Receable",
            "Accounts Receivable",
            "AccountsReceivable",
            "Receivables",
            "Net Receivables",
            "NetReceivables",
        ],
    )

    data["Inv"] = pick_first_available(
        df_raw,
        ["Inventory", "Inventories", "Total Inventory", "TotalInventory"],
    )

    data["K"] = pick_first_available(
        df_raw,
        [
            "Net PPE",
            "NetPPE",
            "Property Plant Equipment Net",
            "PropertyPlantEquipmentNet",
            "Property, Plant & Equipment Net",
            "PropertyPlantAndEquipmentNet",
            "Net Tangible Assets",
            "NetTangibleAssets",
        ],
    )

    data["AP"] = pick_first_available(
        df_raw,
        [
            "Accounts Payable",
            "AccountsPayable",
            "Payables",
            "Trade And Other Payables",
            "TradeAndOtherPayables",
        ],
    )

    total_debt = pick_first_available(
        df_raw,
        ["Total Debt", "TotalDebt"],
        default=np.nan,
    )
    if isinstance(total_debt, (int, float)) and np.isnan(total_debt):
        short_debt = pick_first_available(
            df_raw,
            [
                "Short Long Term Debt",
                "ShortLongTermDebt",
                "Short Term Debt",
                "ShortTermDebt",
                "Current Debt",
                "CurrentDebt",
            ],
            default=0,
        )
        long_debt = pick_first_available(
            df_raw,
            [
                "Long Term Debt",
                "LongTermDebt",
                "Long Term Debt Noncurrent",
                "LongTermDebtNoncurrent",
            ],
            default=0,
        )
        data["D"] = short_debt + long_debt
    else:
        data["D"] = total_debt

    data["E"] = pick_first_available(
        df_raw,
        [
            "Total Stockholder Equity",
            "TotalStockholderEquity",
            "Stockholders Equity",
            "StockholdersEquity",
            "Total Equity Gross Minority Interest",
            "TotalEquityGrossMinorityInterest",
        ],
    )

    data["Total_Assets_Actual"] = pick_first_available(
        df_raw,
        ["Total Assets", "TotalAssets"],
    )
    data["Total_Liab_Actual"] = pick_first_available(
        df_raw,
        [
            "Total Liabilities Net Minority Interest",
            "TotalLiabilitiesNetMinorityInterest",
            "Total Liab",
            "TotalLiab",
        ],
    )

    # ---------- Flows ----------
    data["S"] = pick_first_available(
        df_raw,
        ["Total Revenue", "TotalRevenue", "Revenue", "Revenues"],
    )

    data["COGS"] = pick_first_available(
        df_raw,
        [
            "Cost Of Revenue",
            "CostOfRevenue",
            "Cost Of Goods Sold",
            "CostOfGoodsSold",
        ],
    )

    data["OPEX"] = pick_first_available(
        df_raw,
        [
            "Operating Expense",
            "OperatingExpense",
            "Selling General And Administration",
            "SellingGeneralAndAdministration",
        ],
    )

    data["Dep"] = pick_first_available(
        df_raw,
        [
            "Depreciation And Amortization",
            "DepreciationAndAmortization",
            "Depreciation Amortization Depletion",
            "DepreciationAmortizationDepletion",
        ],
    )

    capex = pick_first_available(
        df_raw,
        [
            "Capital Expenditure",
            "CapitalExpenditure",
            "Purchase Of Property Plant Equipment",
            "PurchaseOfPropertyPlantEquipment",
        ],
        default=0,
    )
    data["I"] = capex.abs() if isinstance(capex, pd.Series) else capex

    new_debt = pick_first_available(
        df_raw,
        ["Issuance Of Debt", "IssuanceOfDebt", "Net Borrowings", "NetBorrowings"],
        default=0,
    )
    data["NewDebt"] = new_debt

    repay_debt = pick_first_available(
        df_raw,
        ["Repayment Of Debt", "RepaymentOfDebt"],
        default=0,
    )
    data["Repay"] = repay_debt.abs() if isinstance(repay_debt, pd.Series) else repay_debt

    equity_issue = pick_first_available(
        df_raw,
        [
            "Issuance Of Capital Stock",
            "IssuanceOfCapitalStock",
            "Sale Of Stock",
            "SaleOfStock",
        ],
        default=0,
    )
    data["EquityIssues"] = equity_issue

    dividends = pick_first_available(
        df_raw,
        [
            "Cash Dividends Paid",
            "CashDividendsPaid",
            "Dividends Paid",
            "DividendsPaid",
        ],
        default=0,
    )
    data["Dividends"] = dividends.abs() if isinstance(dividends, pd.Series) else dividends

    data["NI"] = pick_first_available(
        df_raw,
        [
            "Net Income",
            "NetIncome",
            "Net Income Common Stockholders",
            "NetIncomeCommonStockholders",
        ],
    )

    # 基本清洗：去掉全 0 行
    if fillna_with is not None:
        data = data.fillna(fillna_with)

    data = data.loc[(data.notna() & (data != 0)).any(axis=1)]
    data = data.sort_index(ascending=True)

    if verbose:
        print(f"Final shape for {ticker_symbol}: {data.shape}")

    return data


def build_sequences_for_ticker(df, ticker, sector, seq_len):
    """
    对单个公司构建 LSTM 序列样本。
    返回 X_list, y_list, meta_rows (每个样本一条 meta).
    """
    feature_cols = Y_COLS + X_COLS

    missing_feats = [c for c in feature_cols if c not in df.columns]
    missing_targets = [c for c in Y_COLS if c not in df.columns]
    if missing_feats or missing_targets:
        raise ValueError(f"missing feats={missing_feats}, targets={missing_targets}")

    feat_vals = df[feature_cols].astype(float).values
    y_vals = df[Y_COLS].astype(float).values
    dates = df.index.to_pydatetime()
    T = len(df)

    if T <= seq_len:
        raise ValueError(f"T={T} <= seq_len={seq_len}")

    X_list, y_list, meta_rows = [], [], []
    for i in range(T - seq_len):
        X_seq = feat_vals[i : i + seq_len]
        y_tgt = y_vals[i + seq_len]
        X_list.append(X_seq)
        y_list.append(y_tgt)
        meta_rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "start_date": dates[i].strftime("%Y-%m-%d"),
                "end_date": dates[i + seq_len].strftime("%Y-%m-%d"),
            }
        )
    return X_list, y_list, meta_rows


# ----------------- 主程序 -----------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    sp500 = fetch_sp500_constituents()
    sector_counts = sp500["GICS Sector"].value_counts()
    candidate_sectors = sector_counts[sector_counts >= MIN_TICKERS_PER_SECTOR].index.tolist()

    print("Sectors with at least", MIN_TICKERS_PER_SECTOR, "tickers in S&P500:")
    for s in candidate_sectors:
        print(f"  - {s} (n={sector_counts[s]})")

    feature_cols = Y_COLS + X_COLS

    for sector in candidate_sectors:
        print("\n" + "=" * 80)
        print(f"Processing sector: {sector}")
        print("=" * 80)

        sector_safe = sanitize_sector_name(sector)
        tickers_in_sector = sp500.loc[sp500["GICS Sector"] == sector, "Symbol"].tolist()

        X_all, y_all, meta_all = [], [], []
        tickers_used = set()

        for tic in tickers_in_sector:
            try:
                df = get_company_data(
                    tic,
                    frequency=FREQUENCY,
                    fillna_with=0,
                    verbose=False,
                    show_columns=False,
                )
                if len(df) < MIN_PERIODS:
                    print(f"  Skip {tic}: only {len(df)} periods (<{MIN_PERIODS}).")
                    continue

                X_list, y_list, meta_rows = build_sequences_for_ticker(
                    df, tic, sector, SEQ_LEN
                )
                if not X_list:
                    print(f"  Skip {tic}: no sequences.")
                    continue

                X_all.extend(X_list)
                y_all.extend(y_list)
                meta_all.extend(meta_rows)
                tickers_used.add(tic)
                print(f"  Added {len(X_list)} samples from {tic}")

            except Exception as e:
                print(f"  Error processing {tic}: {e}")

        if len(tickers_used) < MIN_TICKERS_PER_SECTOR:
            print(
                f"\n[Warning] Sector {sector} only has {len(tickers_used)} tickers "
                f"with usable data (<{MIN_TICKERS_PER_SECTOR}). Skipping this sector."
            )
            continue

        if not X_all:
            print(f"No samples for sector {sector}, skipping.")
            continue

        X = np.stack(X_all, axis=0)
        y = np.stack(y_all, axis=0)
        meta = pd.DataFrame(meta_all)

        print(f"\nSector {sector} - total samples: {X.shape[0]}, "
              f"tickers used: {len(tickers_used)}")

        # 随机打乱再拆 train/test
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
        meta = meta.iloc[idx].reset_index(drop=True)

        N = X.shape[0]
        train_end = int(TRAIN_RATIO * N)

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:], y[train_end:]
        meta_train = meta.iloc[:train_end].reset_index(drop=True)
        meta_test = meta.iloc[train_end:].reset_index(drop=True)

        # 保存 npz + meta csv
        npz_path = os.path.join(DATA_DIR, f"{sector_safe}_dataset.npz")
        np.savez(
            npz_path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sector=np.array(sector),
        )
        meta_train.to_csv(
            os.path.join(DATA_DIR, f"{sector_safe}_meta_train.csv"),
            index=False,
        )
        meta_test.to_csv(
            os.path.join(DATA_DIR, f"{sector_safe}_meta_test.csv"),
            index=False,
        )

        print(f"Saved dataset for sector {sector} to {npz_path}")
        print(f"  Train samples: {X_train.shape[0]}")
        print(f"  Test  samples: {X_test.shape[0]}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
