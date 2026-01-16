"""
1. 对每个行业：
   - 载入 data_by_sector/<sector>_dataset.npz 和 meta_test.csv。
   - 载入该行业的两个 LSTM 模型 + 标准化参数。
   - 对测试集样本预测 (unconstrained / constraint-aware)，
     并通过 deterministic baseline 计算对应预测。
   - 计算每个 balance-sheet 科目的 RMSE / MAPE 对比表。

2. 为每个行业选若干示例公司（默认 meta_test 中的前若干个 ticker），
   对这些公司：
   - 用 deterministic baseline 模型模拟 0...T 的资产负债表路径；
   - 用训练好的 LSTM 模型做滚动一次步预测，得到每期的 y_t 预测；
   - 绘制 Equity 的时间序列：真实值 / baseline / unconstrained LSTM /
     constraint-aware LSTM 四条曲线。
   - [修改] 将图像保存为 PDF 文件到 models_by_sector/<sector>/plots/ 目录下。
"""

import os
import glob
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf

# 配置
DATA_DIR = "data_by_sector"
MODEL_DIR = "models_by_sector"

FREQUENCY = "quarterly"
SEQ_LEN = 2

Y_COLS = ["C", "AR", "Inv", "K", "AP", "D", "E"]
X_COLS = [
    "S", "COGS", "OPEX", "Dep", "I",
    "NewDebt", "Repay", "EquityIssues", "Dividends"
]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# --------- 与训练时一致的约束重构函数（用于反序列化 Constraint LSTM） ---------

def constraint_reconstruction(independent):
    """
    与 train_lstm_by_sector.py 中 Lambda 层使用的函数保持一致：
    输入 6 维 (C, AR, Inv, K, AP, D)，输出 7 维 (C, AR, Inv, K, AP, D, E)
    且强制 Assets = Liab + Equity.
    """
    C, AR, Inv, K, AP, D = tf.split(independent, 6, axis=-1)
    assets = C + AR + Inv + K
    liab = AP + D
    E = assets - liab
    full = tf.concat([C, AR, Inv, K, AP, D, E], axis=-1)
    return full


# ------------ 一些与数据和 deterministic baseline 相关的工具 ------------

def sanitize_sector_name(sector: str) -> str:
    return sector.replace(" ", "_").replace("&", "and").replace("/", "-")


def pick_first_available(df, candidates, default=np.nan,
                         suffixes=("", "_BS", "_IS", "_CF")):
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

    # Stocks
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

    # Equity：优先直接字段，若缺失则后面在 evaluate 中用 A-L 重建
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

    # Flows
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

    data = data.fillna(0)
    data = data.loc[(data.notna() & (data != 0)).any(axis=1)]
    data = data.sort_index(ascending=True)

    # 若 E 仍全 0 或缺失，用 A-L 重建
    if ("E" not in data.columns) or (data["E"] == 0).all():
        if "Total_Assets_Actual" in data.columns and "Total_Liab_Actual" in data.columns:
            data["E"] = data["Total_Assets_Actual"] - data["Total_Liab_Actual"]

    return data


def calibrate_deterministic_policies(df):
    params = {}
    # phi_AR
    if "AR" in df.columns and "S" in df.columns:
        mask = df["S"] != 0
        ratio = df.loc[mask, "AR"] / df.loc[mask, "S"]
        params["phi_AR"] = float(ratio.median()) if ratio.notna().any() else 0.0
    else:
        params["phi_AR"] = 0.0

    # phi_Inv
    if "Inv" in df.columns and "COGS" in df.columns:
        mask = df["COGS"] != 0
        ratio = df.loc[mask, "Inv"] / df.loc[mask, "COGS"]
        params["phi_Inv"] = float(ratio.median()) if ratio.notna().any() else 0.0
    else:
        params["phi_Inv"] = 0.0

    # phi_AP
    if "AP" in df.columns and "COGS" in df.columns:
        mask = df["COGS"] != 0
        ratio = df.loc[mask, "AP"] / df.loc[mask, "COGS"]
        params["phi_AP"] = float(ratio.median()) if ratio.notna().any() else 0.0
    else:
        params["phi_AP"] = 0.0

    # delta
    if "K" in df.columns and "I" in df.columns and len(df) > 1:
        K = df["K"].values
        I = df["I"].values
        deltas = []
        for t in range(1, len(df)):
            K_prev = K[t - 1]
            if K_prev != 0:
                delta_t = 1.0 - (K[t] - I[t]) / K_prev
                if np.isfinite(delta_t):
                    deltas.append(delta_t)
        if deltas:
            delta_hat = float(np.median(deltas))
            params["delta"] = float(np.clip(delta_hat, 0.0, 1.0))
        else:
            params["delta"] = 0.1
    else:
        params["delta"] = 0.1

    # p_div
    if "Dividends" in df.columns and "NI" in df.columns:
        mask = df["NI"] > 0
        if mask.any():
            ratio = df.loc[mask, "Dividends"] / df.loc[mask, "NI"]
            ratio = ratio[(ratio >= 0) & (ratio <= 1.5)]
            if ratio.notna().any():
                p_div = float(ratio.median())
                params["p_div"] = float(np.clip(p_div, 0.0, 1.0))
            else:
                params["p_div"] = 0.3
        else:
            params["p_div"] = 0.3
    else:
        params["p_div"] = 0.3

    return params


def simulate_deterministic_balance_sheet(df, params):
    required_cols = [
        "C", "AR", "Inv", "K", "AP", "D", "E",
        "S", "COGS", "Dep", "I",
        "NewDebt", "Repay", "EquityIssues", "NI",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for baseline: {missing}")

    idx = df.index
    n = len(df)

    C_sim = np.zeros(n)
    AR_sim = np.zeros(n)
    Inv_sim = np.zeros(n)
    K_sim = np.zeros(n)
    AP_sim = np.zeros(n)
    D_sim = np.zeros(n)
    E_sim = np.zeros(n)

    C_sim[0] = df["C"].iloc[0]
    AR_sim[0] = df["AR"].iloc[0]
    Inv_sim[0] = df["Inv"].iloc[0]
    K_sim[0] = df["K"].iloc[0]
    AP_sim[0] = df["AP"].iloc[0]
    D_sim[0] = df["D"].iloc[0]
    E_sim[0] = df["E"].iloc[0]

    phi_AR = params.get("phi_AR", 0.0)
    phi_Inv = params.get("phi_Inv", 0.0)
    phi_AP = params.get("phi_AP", 0.0)
    delta = params.get("delta", 0.1)
    p_div = params.get("p_div", 0.3)

    for t in range(1, n):
        S_t = df["S"].iloc[t]
        COGS_t = df["COGS"].iloc[t]
        Dep_t = df["Dep"].iloc[t]
        I_t = df["I"].iloc[t]
        NewDebt_t = df["NewDebt"].iloc[t]
        Repay_t = df["Repay"].iloc[t]
        EqIssues_t = df["EquityIssues"].iloc[t]
        NI_t = df["NI"].iloc[t]

        AR_t = phi_AR * S_t
        Inv_t = phi_Inv * COGS_t
        AP_t = phi_AP * COGS_t

        dAR_t = AR_t - AR_sim[t - 1]
        dInv_t = Inv_t - Inv_sim[t - 1]
        dAP_t = AP_t - AP_sim[t - 1]

        K_t = (1.0 - delta) * K_sim[t - 1] + I_t
        D_t = D_sim[t - 1] + NewDebt_t - Repay_t

        Div_t = p_div * NI_t if NI_t > 0 else 0.0
        E_t = E_sim[t - 1] + NI_t - Div_t + EqIssues_t

        NCF_t = (
            NI_t
            + Dep_t
            - dAR_t
            - dInv_t
            + dAP_t
            - I_t
            + NewDebt_t
            - Repay_t
            + EqIssues_t
            - Div_t
        )
        C_t = C_sim[t - 1] + NCF_t

        C_sim[t] = C_t
        AR_sim[t] = AR_t
        Inv_sim[t] = Inv_t
        K_sim[t] = K_t
        AP_sim[t] = AP_t
        D_sim[t] = D_t
        E_sim[t] = E_t

    sim_df = pd.DataFrame(
        {
            "C_sim": C_sim,
            "AR_sim": AR_sim,
            "Inv_sim": Inv_sim,
            "K_sim": K_sim,
            "AP_sim": AP_sim,
            "D_sim": D_sim,
            "E_sim": E_sim,
        },
        index=idx,
    )
    return sim_df


def compute_metrics(y_true, y_pred):
    rows = []
    for j, c in enumerate(Y_COLS):
        a = y_true[:, j].astype(float)
        p = y_pred[:, j].astype(float)
        diff = p - a
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mask = a != 0
        if mask.any():
            mape = float(np.mean(np.abs(diff[mask] / a[mask])) * 100.0)
        else:
            mape = np.nan
        rows.append({"Item": c, "RMSE": rmse, "MAPE(%)": mape})
    return pd.DataFrame(rows)


def identity_gap(y_pred):
    C_p, AR_p, Inv_p, K_p, AP_p, D_p, E_p = np.split(y_pred, 7, axis=-1)
    assets = C_p + AR_p + Inv_p + K_p
    liab_eq = AP_p + D_p + E_p
    return assets - liab_eq


# ------------ 逐行业评估 ------------

def evaluate_sector(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    sector = str(data["sector"])
    sector_safe = sanitize_sector_name(sector)

    print("\n" + "=" * 80)
    print(f"Evaluating sector: {sector}")
    print("=" * 80)

    # meta_test 里有 ticker 和 end_date，可用于 baseline 对齐
    meta_test_path = os.path.join(DATA_DIR, f"{sector_safe}_meta_test.csv")
    meta_test = pd.read_csv(meta_test_path)

    # 载入模型和标准化参数
    sector_dir = os.path.join(MODEL_DIR, sector_safe)
    uc_path = os.path.join(sector_dir, "unconstrained_lstm.h5")
    ca_path = os.path.join(sector_dir, "constraint_aware_lstm.h5")
    scalers_path = os.path.join(sector_dir, "scalers.npz")

    model_uc = tf.keras.models.load_model(uc_path, compile=False)
    model_ca = tf.keras.models.load_model(
        ca_path,
        compile=False,
        custom_objects={"constraint_reconstruction": constraint_reconstruction},
    )
    scalers = np.load(scalers_path)
    X_mean = scalers["X_mean"]
    X_std = scalers["X_std"]
    y_mean = scalers["y_mean"]
    y_std = scalers["y_std"]

    # 对测试集做归一化
    X_test_n = (X_test - X_mean) / X_std

    # LSTM 预测
    y_pred_uc_n = model_uc.predict(X_test_n, verbose=0)
    y_pred_ca_n = model_ca.predict(X_test_n, verbose=0)
    y_pred_uc = y_pred_uc_n * y_std + y_mean
    y_pred_ca = y_pred_ca_n * y_std + y_mean

    # deterministic baseline 预测
    print("\n  Building deterministic baselines for test tickers...")
    unique_tickers = meta_test["ticker"].unique()
    baseline_sim = {}
    for tic in unique_tickers:
        try:
            df_t = get_company_data(
                tic,
                frequency=FREQUENCY,
                fillna_with=0,
                verbose=False,
                show_columns=False,
            )
            if len(df_t) < 2:
                continue
            theta_t = calibrate_deterministic_policies(df_t)
            sim_t = simulate_deterministic_balance_sheet(df_t, theta_t)
            baseline_sim[tic] = sim_t
        except Exception as e:
            print(f"    Baseline error for {tic}: {e}")

    y_pred_det = np.zeros_like(y_test)
    y_pred_det[:] = np.nan

    for i, row in meta_test.iterrows():
        tic = row["ticker"]
        dt_str = row["end_date"]
        if tic not in baseline_sim:
            continue
        sim_df = baseline_sim[tic]
        dt = pd.to_datetime(dt_str)
        try:
            sim_row = sim_df.loc[dt]
        except KeyError:
            idx = sim_df.index
            pos = idx.get_loc(dt, method="nearest")
            sim_row = sim_df.iloc[pos]
        vals = [
            sim_row["C_sim"],
            sim_row["AR_sim"],
            sim_row["Inv_sim"],
            sim_row["K_sim"],
            sim_row["AP_sim"],
            sim_row["D_sim"],
            sim_row["E_sim"],
        ]
        y_pred_det[i, :] = np.array(vals, dtype=float)

    # 计算会计恒等式违背程度
    gap_uc = identity_gap(y_pred_uc)
    gap_ca = identity_gap(y_pred_ca)
    gap_det = identity_gap(y_pred_det)

    print("\n  Accounting Identity Violations (|Assets - Liab - Eq|):")
    print("    Unconstrained LSTM    max =", np.nanmax(np.abs(gap_uc)),
          "mean =", np.nanmean(np.abs(gap_uc)))
    print("    Constraint-Aware LSTM max =", np.nanmax(np.abs(gap_ca)),
          "mean =", np.nanmean(np.abs(gap_ca)))
    print("    Deterministic         max =", np.nanmax(np.abs(gap_det)),
          "mean =", np.nanmean(np.abs(gap_det)))

    # 指标对比表
    metrics_uc = compute_metrics(y_test, y_pred_uc)
    metrics_ca = compute_metrics(y_test, y_pred_ca)
    metrics_det = compute_metrics(y_test, y_pred_det)

    rows = []
    for col in Y_COLS:
        row_uc = metrics_uc[metrics_uc["Item"] == col].iloc[0]
        row_ca = metrics_ca[metrics_ca["Item"] == col].iloc[0]
        row_det = metrics_det[metrics_det["Item"] == col].iloc[0]
        rows.append(
            {
                "Item": col,
                "RMSE_Unconstrained": row_uc["RMSE"],
                "RMSE_ConstraintAware": row_ca["RMSE"],
                "RMSE_Deterministic": row_det["RMSE"],
                "MAPE_Unconstrained(%)": row_uc["MAPE(%)"],
                "MAPE_ConstraintAware(%)": row_ca["MAPE(%)"],
                "MAPE_Deterministic(%)": row_det["MAPE(%)"],
            }
        )
    comp_df = pd.DataFrame(rows)
    print("\n  Test-set comparison for sector:", sector)
    print(comp_df)

    out_csv = os.path.join(
        MODEL_DIR, f"comparison_{sanitize_sector_name(sector)}.csv"
    )
    comp_df.to_csv(out_csv, index=False)
    print("  Saved comparison metrics to", out_csv)

    # --------- 画示例公司的 Equity 曲线并保存为 PDF ---------
    
    # [修改] 创建保存图片的目录
    plots_dir = os.path.join(sector_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"  Generating plots in {plots_dir}...")

    example_tickers = meta_test["ticker"].unique()[:3]  # 每个行业画前 3 个测试公司
    for tic in example_tickers:
        try:
            plot_equity_paths_for_ticker(
                tic,
                sector,
                model_uc,
                model_ca,
                X_mean,
                X_std,
                y_mean,
                y_std,
                save_dir=plots_dir # [修改] 传入保存路径
            )
        except Exception as e:
            print(f"  Plot error for {tic}: {e}")


def plot_equity_paths_for_ticker(
    ticker,
    sector,
    model_uc,
    model_ca,
    X_mean,
    X_std,
    y_mean,
    y_std,
    save_dir=None # [修改] 增加参数
):
    """
    对单个公司绘制 0...T 的 Equity 路径比较：
      - 真实值
      - deterministic baseline
      - Unconstrained LSTM（滚动 one-step-ahead）
      - Constraint-aware LSTM
    """
    df = get_company_data(
        ticker,
        frequency=FREQUENCY,
        fillna_with=0,
        verbose=False,
        show_columns=False,
    )
    if len(df) <= SEQ_LEN:
        print(f"    Not enough periods for plotting {ticker}.")
        return

    theta = calibrate_deterministic_policies(df)
    sim_df = simulate_deterministic_balance_sheet(df, theta)

    # 准备特征和真实 y
    feature_cols = Y_COLS + X_COLS
    feat_vals = df[feature_cols].astype(float).values  # (T, 16)
    y_vals = df[Y_COLS].astype(float).values           # (T, 7)
    dates = df.index

    T = len(df)
    pred_uc = np.full_like(y_vals, np.nan)
    pred_ca = np.full_like(y_vals, np.nan)

    # 把 X_mean / X_std 压缩成 (1, n_features) 用于广播
    X_center = X_mean.reshape(1, -1)
    X_scale = X_std.reshape(1, -1)

    for t in range(SEQ_LEN, T):
        # 使用前 SEQ_LEN 期预测第 t 期
        X_seq = feat_vals[t - SEQ_LEN: t]  # (SEQ_LEN, 16)
        # 归一化到与训练时一致的尺度
        X_seq_n = (X_seq - X_center) / X_scale  # (SEQ_LEN, 16)
        X_seq_n = X_seq_n[np.newaxis, ...]      # (1, SEQ_LEN, 16)

        y_uc_n = model_uc.predict(X_seq_n, verbose=0)  # (1, 7)
        y_ca_n = model_ca.predict(X_seq_n, verbose=0)  # (1, 7)

        y_uc = y_uc_n * y_std + y_mean
        y_ca = y_ca_n * y_std + y_mean

        pred_uc[t] = y_uc[0]
        pred_ca[t] = y_ca[0]

    # 绘制 Equity (index 6 in Y_COLS)
    idx_E = Y_COLS.index("E")
    plt.figure(figsize=(8, 4))
    plt.plot(dates, y_vals[:, idx_E] / 1e9, marker="o", label="Actual Equity")
    plt.plot(
        dates,
        sim_df["E_sim"].values / 1e9,
        marker="x",
        linestyle="--",
        label="Deterministic baseline",
    )
    plt.plot(
        dates,
        pred_uc[:, idx_E] / 1e9,
        marker="s",
        linestyle="-.",
        label="Unconstrained LSTM",
    )
    plt.plot(
        dates,
        pred_ca[:, idx_E] / 1e9,
        marker="^",
        linestyle=":",
        label="Constraint-aware LSTM",
    )

    plt.title(f"{ticker} ({sector}) - Equity: Actual vs Baseline vs LSTMs")
    plt.xlabel("Period end date")
    plt.ylabel("Equity (billion)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # [修改] 保存逻辑
    if save_dir:
        safe_sector = sanitize_sector_name(sector)
        filename = f"{ticker}_{safe_sector}_equity.pdf"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"    Saved plot: {save_path}")
        plt.close() # 绘图后关闭，释放内存
    else:
        plt.show()


def main():
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_dataset.npz")))
    if not npz_files:
        print("No dataset npz files found in", DATA_DIR)
        return

    for npz_path in npz_files:
        evaluate_sector(npz_path)

    print("\nAll sectors evaluated.")


if __name__ == "__main__":
    main()
