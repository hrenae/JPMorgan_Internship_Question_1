"""
Deterministic balance sheet baseline
------------------------------------

This script implements the deterministic structural balance-sheet model
used as a baseline in the report.

For each ticker:
1. Download financial statements from Yahoo Finance (annual or quarterly).
2. Map Yahoo fields into model stocks y_t and flows x_t.
3. Use the first T-1 observations to estimate policy parameters
   theta = (phi_AR, phi_Inv, phi_AP, delta, p_div).
4. Starting from t=0, simulate the whole path up to t=T using these
   fixed parameters.
5. Treat y_T(sim) as a one-step-ahead forecast and compare it with
   the last actual observation y_T(actual).
6. Report:
   - In-sample path fit on the training window (0,...,T-1).
   - Out-of-sample one-step-ahead forecast error at T.
   - A plot of Equity (E) actual vs. deterministic baseline, with
     the last-period forecast highlighted.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Global config
# ----------------------------------------------------------------------
FREQUENCY = "quarterly"   # "quarterly" 或 "annual"
# FREQUENCY = "annual"
TICKERS = ["AAPL", "MSFT", "XOM", "WMT"]
MIN_PERIODS = 5 if FREQUENCY == "quarterly" else 4


Y_COLS   = ["C", "AR", "Inv", "K", "AP", "D", "E"]  # stock variables
X_COLS   = ["S", "COGS", "OPEX", "Dep", "I",
            "NewDebt", "Repay", "EquityIssues", "Dividends"]
META_COLS = ["Total_Assets_Actual", "Total_Liab_Actual", "NI"]


# ----------------------------------------------------------------------
# Helper: pick first available column in df
# ----------------------------------------------------------------------
def pick_first_available(df, candidates, default=np.nan,
                         suffixes=("", "_BS", "_IS", "_CF")):
    """
    Try multiple candidate base names (with optional suffixes) and
    return the first matching column in df. If nothing matches,
    return the scalar 'default'.
    """
    for base in candidates:
        for suf in suffixes:
            col = base + suf
            if col in df.columns:
                return df[col]
    return default


# ----------------------------------------------------------------------
# 1. Download and map company data to model variables
# ----------------------------------------------------------------------
def get_company_data(
    ticker_symbol,
    frequency=FREQUENCY,
    min_periods=3,
    fillna_with=0,
    verbose=True,
    show_columns=False,
):
    """
    Download and align a single firm's financial statements, then map
    them to the simplified stock and flow variables.

    Returns a DataFrame indexed by period end date, sorted ascending,
    with columns Y_COLS, X_COLS and META_COLS (where available).
    """
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

    if show_columns:
        print(f"\n[{ticker_symbol}] Balance sheet cols:", bs.columns.tolist())
        print(f"[{ticker_symbol}] Income statement cols:", is_.columns.tolist())
        print(f"[{ticker_symbol}] Cashflow statement cols:", cf.columns.tolist())

    # Outer join the three statements on date index
    df_raw = bs.join(is_, how="outer", rsuffix="_IS").join(
        cf, how="outer", rsuffix="_CF"
    ).sort_index()

    if verbose:
        print(f"[{ticker_symbol}] raw periods: {list(df_raw.index)}")

    data = pd.DataFrame(index=df_raw.index)

    # ----------------- Stocks (y_t) -----------------
    # Cash & cash equivalents (C)
    data["C"] = pick_first_available(
        df_raw,
        [
            "Cash And Cash Equivalents",
            "CashAndCashEquivalents",
            "Cash And Short Term Investments",
            "CashAndShortTermInvestments",
        ],
    )

    # Accounts receivable (AR)
    data["AR"] = pick_first_available(
        df_raw,
        [
            "Accounts Receivable",
            "AccountsReceivable",
            "Receivables",
            "Net Receivables",
            "NetReceivables",
        ],
    )

    # Inventories (Inv)
    data["Inv"] = pick_first_available(
        df_raw,
        [
            "Inventory",
            "Inventories",
            "Total Inventory",
            "TotalInventory",
        ],
    )

    # Net PPE (K)
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

    # Accounts payable (AP)
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

    # Total debt (D)
    total_debt = pick_first_available(
        df_raw,
        [
            "Total Debt",
            "TotalDebt",
        ],
        default=np.nan,
    )
    if isinstance(total_debt, (int, float)) and not np.isnan(total_debt):
        data["D"] = total_debt
    else:
        short_debt = pick_first_available(
            df_raw,
            [
                "Short Term Debt",
                "ShortTermDebt",
                "Short Long Term Debt",
                "ShortLongTermDebt",
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

    # Equity (E) – direct column if available
    data["E"] = pick_first_available(
        df_raw,
        [
            "Total Stockholder Equity",
            "TotalStockholderEquity",
            "Total Stockholders Equity",
            "TotalStockholdersEquity",
            "Stockholders Equity",
            "StockholdersEquity",
            "Total Equity",
            "TotalEquity",
            "Total Equity Gross Minority Interest",
            "TotalEquityGrossMinorityInterest",
        ],
    )

    # Total assets / liabilities for sanity checks and possible E fallback
    data["Total_Assets_Actual"] = pick_first_available(
        df_raw,
        [
            "Total Assets",
            "TotalAssets",
        ],
    )
    data["Total_Liab_Actual"] = pick_first_available(
        df_raw,
        [
            "Total Liabilities Net Minority Interest",
            "TotalLiabilitiesNetMinorityInterest",
            "Total Liabilities",
            "TotalLiabilities",
            "Total Liab",
            "TotalLiab",
        ],
    )

    # ----------------- Flows (x_t) -----------------
    data["S"] = pick_first_available(
        df_raw,
        [
            "Total Revenue",
            "TotalRevenue",
            "Revenue",
        ],
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
            "Selling General Administrative",
            "SellingGeneralAdministrative",
        ],
    )

    data["Dep"] = pick_first_available(
        df_raw,
        [
            "Depreciation",
            "DepreciationAndAmortization",
            "Depreciation And Amortization",
        ],
    )

    capex = pick_first_available(
        df_raw,
        [
            "Capital Expenditure",
            "CapitalExpenditure",
            "Capital Expenditures",
            "CapitalExpenditures",
        ],
        default=0,
    )
    # CAPEX is usually negative in CF statement; take absolute value
    if isinstance(capex, pd.Series):
        data["I"] = capex.abs()
    else:
        data["I"] = abs(capex)

    # Debt flows
    new_debt = pick_first_available(
        df_raw,
        [
            "Issuance Of Debt",
            "IssuanceOfDebt",
            "Net Borrowings",
            "NetBorrowings",
        ],
        default=0,
    )
    data["NewDebt"] = new_debt

    repay_debt = pick_first_available(
        df_raw,
        [
            "Repayment Of Debt",
            "RepaymentOfDebt",
        ],
        default=0,
    )
    if isinstance(repay_debt, pd.Series):
        data["Repay"] = repay_debt.abs()
    else:
        data["Repay"] = abs(repay_debt)

    # Equity flows
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
            "Payments Of Dividends",
            "PaymentsOfDividends",
        ],
        default=0,
    )
    if isinstance(dividends, pd.Series):
        data["Dividends"] = dividends.abs()
    else:
        data["Dividends"] = abs(dividends)

    # Net income (NI)
    data["NI"] = pick_first_available(
        df_raw,
        [
            "Net Income",
            "NetIncome",
            "Net Income Common Stockholders",
            "NetIncomeCommonStockholders",
        ],
    )

    # Optional: drop rows where both S and NI are zero (often missing data)
    if "S" in data.columns and "NI" in data.columns:
        mask_valid = ~((data["S"] == 0) & (data["NI"] == 0))
        data = data.loc[mask_valid]

    # Fill NaNs (if requested)
    if fillna_with is not None:
        data = data.fillna(fillna_with)

    if len(data) < min_periods:
        if verbose:
            print(f"[{ticker_symbol}] only {len(data)} usable periods (<{min_periods})")
        return data

    data = data.sort_index()

    if verbose:
        print(f"[{ticker_symbol}] final data shape: {data.shape}")

    return data


# ----------------------------------------------------------------------
# 2. Calibrate deterministic policy parameters from history
# ----------------------------------------------------------------------
def calibrate_deterministic_policies(df):
    """
    Estimate simple deterministic policy parameters from historical data:
      phi_AR, phi_Inv, phi_AP, delta, p_div.
    """
    params = {}

    # phi_AR ≈ AR / S
    if "AR" in df.columns and "S" in df.columns:
        mask = df["S"] != 0
        ratio = df.loc[mask, "AR"] / df.loc[mask, "S"]
        params["phi_AR"] = float(ratio.median()) if ratio.notna().any() else 0.0
    else:
        params["phi_AR"] = 0.0

    # phi_Inv ≈ Inv / COGS
    if "Inv" in df.columns and "COGS" in df.columns:
        mask = df["COGS"] != 0
        ratio = df.loc[mask, "Inv"] / df.loc[mask, "COGS"]
        params["phi_Inv"] = float(ratio.median()) if ratio.notna().any() else 0.0
    else:
        params["phi_Inv"] = 0.0

    # phi_AP ≈ AP / COGS
    if "AP" in df.columns and "COGS" in df.columns:
        mask = df["COGS"] != 0
        ratio = df.loc[mask, "AP"] / df.loc[mask, "COGS"]
        params["phi_AP"] = float(ratio.median()) if ratio.notna().any() else 0.0
    else:
        params["phi_AP"] = 0.0

    # delta from K_t = (1 - delta) K_{t-1} + I_t
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

    # p_div ≈ Div / NI (only for NI>0)
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


# ----------------------------------------------------------------------
# 3. Simulate deterministic balance-sheet path
# ----------------------------------------------------------------------
def simulate_deterministic_balance_sheet(df, params):
    """
    Simulate balance-sheet path y_t under the deterministic policy rules.
    df: DataFrame with true stocks y_t and flows x_t for t=0,...,T.
    params: dict with phi_AR, phi_Inv, phi_AP, delta, p_div.
    Returns a DataFrame with C_sim,...,E_sim for the same index.
    """
    required_cols = [
        "C",
        "AR",
        "Inv",
        "K",
        "AP",
        "D",
        "E",
        "S",
        "COGS",
        "Dep",
        "I",
        "NewDebt",
        "Repay",
        "EquityIssues",
        "NI",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    idx = df.index
    n = len(df)

    C_sim = np.zeros(n)
    AR_sim = np.zeros(n)
    Inv_sim = np.zeros(n)
    K_sim = np.zeros(n)
    AP_sim = np.zeros(n)
    D_sim = np.zeros(n)
    E_sim = np.zeros(n)

    # Initial condition at t=0: set equal to actual stocks
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

        # 1) Working-capital policy
        AR_t = phi_AR * S_t
        Inv_t = phi_Inv * COGS_t
        AP_t = phi_AP * COGS_t

        dAR_t = AR_t - AR_sim[t - 1]
        dInv_t = Inv_t - Inv_sim[t - 1]
        dAP_t = AP_t - AP_sim[t - 1]

        # 2) Fixed assets
        K_t = (1.0 - delta) * K_sim[t - 1] + I_t

        # 3) Debt
        D_t = D_sim[t - 1] + NewDebt_t - Repay_t

        # 4) Equity & dividend policy (clean surplus)
        Div_t = p_div * NI_t if NI_t > 0 else 0.0
        E_t = E_sim[t - 1] + NI_t - Div_t + EqIssues_t

        # 5) Cash via cash-flow identity
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


# ----------------------------------------------------------------------
# 4a. In-sample path fit on training window 0,...,T-1
# ----------------------------------------------------------------------
def evaluate_path_fit(actual_df, sim_df):
    """
    Compute RMSE and MAPE for each balance-sheet item over the
    training window (e.g. t=0,...,T-1).
    """
    common_idx = actual_df.index.intersection(sim_df.index)
    actual = actual_df.loc[common_idx, Y_COLS]
    sim = sim_df.loc[common_idx, [c + "_sim" for c in Y_COLS]]
    sim.columns = Y_COLS

    metrics = []
    for col in Y_COLS:
        a = actual[col].astype(float)
        s = sim[col].astype(float)
        diff = s - a
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        mask = a != 0
        if mask.any():
            mape = float(np.mean(np.abs(diff[mask] / a[mask])) * 100.0)
        else:
            mape = np.nan

        metrics.append({"Item": col, "RMSE": rmse, "MAPE(%)": mape})

    return pd.DataFrame(metrics)


# ----------------------------------------------------------------------
# 4b. One-step-ahead forecast error at T (last period)
# ----------------------------------------------------------------------
def evaluate_last_period_forecast(actual_df, sim_df):
    """
    Use only the last period t=T to compute RMSE and MAPE.
    For a single point, RMSE reduces to the absolute error.
    """
    common_idx = actual_df.index.intersection(sim_df.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping index between actual_df and sim_df")

    last_idx = common_idx[-1]
    a = actual_df.loc[last_idx, Y_COLS]
    s = sim_df.loc[last_idx, [c + "_sim" for c in Y_COLS]]
    s.index = Y_COLS

    metrics = []
    for col in Y_COLS:
        actual_val = float(a[col])
        sim_val = float(s[col])
        diff = sim_val - actual_val
        rmse = abs(diff)
        if actual_val != 0:
            mape = abs(diff / actual_val) * 100.0
        else:
            mape = np.nan
        metrics.append({"Item": col, "RMSE": rmse, "MAPE(%)": mape})

    return pd.DataFrame(metrics)


# ----------------------------------------------------------------------
# 5. Main: run baseline for a set of tickers
# ----------------------------------------------------------------------
def main():
    results = {}

    for t in TICKERS:
        try:
            df = get_company_data(
                t,
                frequency=FREQUENCY,
                min_periods=MIN_PERIODS,
                fillna_with=0,
                verbose=True,
                show_columns=False,
            )

            if len(df) < MIN_PERIODS:
                print(f"Skip {t}: not enough periods.")
                continue

            # --------- Ensure we have an Equity column E ----------
            if "E" not in df.columns or pd.isna(df["E"]).all():
                if (
                    "Total_Assets_Actual" in df.columns
                    and "Total_Liab_Actual" in df.columns
                ):
                    df["E"] = df["Total_Assets_Actual"] - df["Total_Liab_Actual"]
                    print(
                        f"[{t}] Constructed Equity E = Total_Assets_Actual - Total_Liab_Actual"
                    )
                else:
                    print(
                        f"Skip {t}: cannot construct Equity (missing E and total assets/liabilities)."
                    )
                    continue

            # Sort by date and split into train (0,...,T-1) and test (T)
            df = df.sort_index()
            train_df = df.iloc[:-1]
            test_idx = df.index[-1]
            print(
                f"Using {len(train_df)} periods to estimate policies, "
                f"holding out {test_idx} as one-step-ahead forecast target."
            )

            # Save full actual stocks
            actual_y = df[Y_COLS].copy()

            # 1) Calibrate parameters on training window only
            theta = calibrate_deterministic_policies(train_df)
            print(f"\nCalibrated params for {t}: {theta}")

            # 2) Simulate deterministic baseline over full horizon 0,...,T
            sim_df = simulate_deterministic_balance_sheet(df, theta)

            # 3) In-sample path fit on 0,...,T-1
            path_metrics = evaluate_path_fit(
                train_df[Y_COLS], sim_df.loc[train_df.index]
            )

            # 4) One-step-ahead forecast error at T
            forecast_metrics = evaluate_last_period_forecast(actual_y, sim_df)

            results[t] = {
                "data": df,
                "sim": sim_df,
                "params": theta,
                "path_metrics": path_metrics,
                "forecast_metrics": forecast_metrics,
            }

            print(f"\n=== {t} ({FREQUENCY}) deterministic baseline ===")
            print("In-sample path fit (train window 0,...,T-1):")
            print(
                path_metrics.to_string(
                    index=False, float_format=lambda x: f"{x:,.2f}"
                )
            )

            print("\nOne-step-ahead forecast error at T (last period):")
            print(
                forecast_metrics.to_string(
                    index=False, float_format=lambda x: f"{x:,.2f}"
                )
            )

            # Check accounting identity along simulated path
            assets_sim = (
                sim_df["C_sim"]
                + sim_df["AR_sim"]
                + sim_df["Inv_sim"]
                + sim_df["K_sim"]
            )
            liab_eq_sim = sim_df["AP_sim"] + sim_df["D_sim"] + sim_df["E_sim"]
            gap_sim = assets_sim - liab_eq_sim
            print(
                f"\nMax |Assets - Liab - Equity| along simulated path for {t}: "
                f"{gap_sim.abs().max():,.2f}"
            )
            print("-" * 60)

            # Plot: Equity actual vs baseline (last point is forecast)
            plt.figure(figsize=(8, 4))
            plt.plot(
                actual_y.index,
                actual_y["E"] / 1e9,
                marker="o",
                label="Actual Equity",
            )
            plt.plot(
                sim_df.index,
                sim_df["E_sim"] / 1e9,
                marker="x",
                linestyle="--",
                label="Deterministic baseline",
            )

            # Highlight last-period actual and forecast
            plt.scatter(
                [test_idx],
                [actual_y.loc[test_idx, "E"] / 1e9],
                s=80,
                marker="o",
                facecolors="none",
                edgecolors="C0",
                label="Actual (T)",
            )
            plt.scatter(
                [test_idx],
                [sim_df.loc[test_idx, "E_sim"] / 1e9],
                s=80,
                marker="x",
                edgecolors="C1",
                label="Forecast (T)",
            )

            freq_label = "Quarterly" if FREQUENCY == "quarterly" else "Annual"
            plt.title(f"{t} - Equity actual vs deterministic baseline ({freq_label})")
            plt.xlabel("Period end date")
            plt.ylabel("Equity (billion)")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error processing {t}: {e}")

    return results


if __name__ == "__main__":
    results = main()
