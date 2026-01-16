#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare annual (A) balance-sheet / income-statement / cash-flow data from yfinance,
and build "transition tables" suitable for the Pareja-style baseline simulator:

    y_t = f(x_t, y_{t-1}) + ε_t

Key outputs (written under OUT_DIR)
-----------------------------------
OUT_DIR/
  periods/            per-ticker annual period tables (levels + flows)
  transitions/        per-ticker annual transition tables (one row per year-to-year transition)
  panel_periods_A.csv concatenation of all annual periods
  panel_transitions_A.csv concatenation of all annual transitions
  universe.csv        tickers/sectors/labels used for this run
  build_log.csv       per-ticker status summary (counts + fetch errors)

Config is intentionally at the top of the file so you can run:
    python prepare_data_tsrl_baseline_v3.py
without command-line arguments.

Input universe
--------------
Reads tickers from DATA_CSV (DataPrepare.csv) with columns:
  * ticker
  * sector
  * Lable   (misspelled by design in the provided file; NaN means "train", 'test' means report tickers)

"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# External dependency used by design.
import yfinance as yf


# ---------------------------
# Configuration (edit here)
# ---------------------------
DATA_CSV = "DataPrepare.csv"     # ticker universe
OUT_DIR = "data_tsrl_annual_v3"  # output folder
PAUSE_S = 0.35                   # pause between yfinance calls (rate limiting)
RETRIES = 2                      # retries per statement fetch
REBUILD = False                  # if False, skip tickers already prepared
FREQ = "A"                       # fixed: annual only


# ---------------------------
# Helpers: robust extraction
# ---------------------------
def pick_value(series: pd.Series, keys_exact: List[str], keys_regex: Optional[List[str]] = None) -> Tuple[float, str]:
    """Pick a scalar value from a statement row (pd.Series), given candidate keys."""
    if series is None or series.empty:
        return np.nan, ""
    for k in keys_exact:
        if k in series.index:
            v = series.get(k)
            if pd.notna(v):
                return float(v), k
    if keys_regex:
        import re
        for pat in keys_regex:
            rx = re.compile(pat, flags=re.IGNORECASE)
            for idx in series.index:
                if rx.search(str(idx)):
                    v = series.get(idx)
                    if pd.notna(v):
                        return float(v), str(idx)
    return np.nan, ""


def sum_values(series: pd.Series, keys_exact: List[str], keys_regex: Optional[List[str]] = None) -> Tuple[float, str]:
    """Sum multiple candidate keys; ignore missing."""
    if series is None or series.empty:
        return np.nan, ""
    vals = []
    srcs = []
    for k in keys_exact:
        if k in series.index and pd.notna(series.get(k)):
            vals.append(float(series.get(k)))
            srcs.append(k)
    if keys_regex:
        import re
        for pat in keys_regex:
            rx = re.compile(pat, flags=re.IGNORECASE)
            for idx in series.index:
                if rx.search(str(idx)) and pd.notna(series.get(idx)):
                    vals.append(float(series.get(idx)))
                    srcs.append(str(idx))
    if len(vals) == 0:
        return np.nan, ""
    return float(np.nansum(vals)), "+".join(srcs)


def _to_frame(x) -> pd.DataFrame:
    if x is None:
        return pd.DataFrame()
    if isinstance(x, pd.DataFrame):
        return x
    try:
        return pd.DataFrame(x)
    except Exception:
        return pd.DataFrame()


def fetch_with_retries(ticker: str, attrs: List[str], retries: int, pause_s: float) -> Tuple[pd.DataFrame, str]:
    """Attempt to fetch a yfinance statement DataFrame from several candidate attributes."""
    err = ""
    for attempt in range(retries + 1):
        try:
            t = yf.Ticker(ticker)
            out = None
            for a in attrs:
                if hasattr(t, a):
                    out = getattr(t, a)
                    out = _to_frame(out)
                    if out is not None and not out.empty:
                        return out, ""
            err = f"no_valid_attr({attrs})"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        time.sleep(pause_s)
    return pd.DataFrame(), err


# ---------------------------
# Statement extraction: Balance Sheet
# ---------------------------
def extract_balance_sheet_one_period(series: pd.Series) -> Dict[str, object]:
    """Extract balance-sheet levels at a given period end."""
    out: Dict[str, object] = {}

    C, Csrc = pick_value(
        series,
        keys_exact=["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash Cash Equivalents And Short Term Investments", "CashCashEquivalentsAndShortTermInvestments", "Cash"],
        keys_regex=[r"cash( and cash equivalents)?$"],
    )
    AR, ARsrc = pick_value(
        series,
        keys_exact=["Accounts Receivable", "AccountsReceivable", "Receivables", "ReceivablesNet", "Net Receivables"],
        keys_regex=[r"accounts receivable", r"receivables"],
    )
    Inv, Invsrc = pick_value(
        series,
        keys_exact=["Inventory", "Inventories"],
        keys_regex=[r"inventory"],
    )
    K, Ksrc = pick_value(
        series,
        keys_exact=["Net PPE", "NetPPE", "Property Plant Equipment Net", "PropertyPlantEquipmentNet", "Property Plant And Equipment Net", "PropertyPlantAndEquipmentNet"],
        keys_regex=[r"ppe", r"property.*equipment.*net"],
    )
    AP, APsrc = pick_value(
        series,
        keys_exact=["Accounts Payable", "AccountsPayable", "Payables", "PayablesAndAccruedExpenses", "Payables And Accrued Expenses"],
        keys_regex=[r"accounts payable", r"payables( and accrued.*)?$"],
    )
    STD, STDsrc = pick_value(
        series,
        keys_exact=["Short Term Debt", "ShortTermDebt", "Current Debt", "CurrentDebt", "Current Debt And Capital Lease Obligation", "CurrentDebtAndCapitalLeaseObligation"],
        keys_regex=[r"short term debt", r"current debt"],
    )
    LTD, LTDsrc = pick_value(
        series,
        keys_exact=["Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation", "LongTermDebtAndCapitalLeaseObligation", "Long Term Debt Noncurrent", "LongTermDebtNoncurrent"],
        keys_regex=[r"long term debt"],
    )

    TA, TAsrc = pick_value(series, keys_exact=["Total Assets", "TotalAssets"], keys_regex=[r"total assets"])
    TL, TLsrc = pick_value(
        series,
        keys_exact=["Total Liabilities Net Minority Interest", "TotalLiabilitiesNetMinorityInterest", "Total Liabilities", "TotalLiabilities", "Total Liab"],
        keys_regex=[r"total liabilities", r"total liab"],
    )
    TCA, TCAsrc = pick_value(
        series,
        keys_exact=["Total Current Assets", "TotalCurrentAssets", "Current Assets", "CurrentAssets"],
        keys_regex=[r"total current assets", r"current assets$"],
    )
    TCL, TCLsrc = pick_value(
        series,
        keys_exact=["Total Current Liabilities", "TotalCurrentLiabilities", "Current Liabilities", "CurrentLiabilities"],
        keys_regex=[r"total current liabilities", r"current liabilities$"],
    )

    MI, MIsrc = pick_value(
        series,
        keys_exact=["Minority Interest", "MinorityInterest", "Minority Interests", "MinorityInterests", "Noncontrolling Interests", "NoncontrollingInterests"],
        keys_regex=[r"minority interest", r"noncontrolling interests?"],
    )
    E_stock, E_stock_src = pick_value(
        series,
        keys_exact=["Total Stockholder Equity", "TotalStockholderEquity", "Stockholders Equity", "StockholdersEquity", "Common Stock Equity", "CommonStockEquity"],
        keys_regex=[r"stockholders? equity", r"common stock equity"],
    )
    E_gross, E_gross_src = pick_value(
        series,
        keys_exact=["Total Equity Gross Minority Interest", "TotalEquityGrossMinorityInterest"],
        keys_regex=[r"total equity gross minority interest"],
    )

    # Prefer gross equity if present; else build gross from stockholders equity + minority interest if possible
    if pd.notna(E_gross):
        E_gross_report, E_gross_report_src_final = float(E_gross), E_gross_src
        E_report, E_report_src_final = float(E_gross), E_gross_src
    else:
        if pd.notna(E_stock):
            E_gross_report = float(E_stock) + float(MI) if pd.notna(MI) else float(E_stock)
            E_gross_report_src_final = f"{E_stock_src}+{MIsrc}" if pd.notna(MI) else E_stock_src
            E_report, E_report_src_final = float(E_stock), E_stock_src
        else:
            E_gross_report, E_gross_report_src_final, E_report, E_report_src_final = np.nan, "", np.nan, ""

    out.update(
        {
            "C": C,
            "C_src": Csrc,
            "AR": AR,
            "AR_src": ARsrc,
            "Inv": Inv,
            "Inv_src": Invsrc,
            "K": K,
            "K_src": Ksrc,
            "AP": AP,
            "AP_src": APsrc,
            "STD": STD,
            "STD_src": STDsrc,
            "LTD": LTD,
            "LTD_src": LTDsrc,
            "TA": TA,
            "TA_src": TAsrc,
            "TL": TL,
            "TL_src": TLsrc,
            "TCA": TCA,
            "TCA_src": TCAsrc,
            "TCL": TCL,
            "TCL_src": TCLsrc,
            "E_report": E_report,
            "E_report_src": E_report_src_final,
            "E_gross_report": E_gross_report,
            "E_gross_report_src": E_gross_report_src_final,
            "MI_report": MI,
            "MI_report_src": MIsrc,
        }
    )
    return out


# ---------------------------
# Statement extraction: Income Statement
# ---------------------------
def extract_income_one_period(series: pd.Series) -> Dict[str, object]:
    """Extract income-statement flows for the period."""
    out: Dict[str, object] = {}

    # Revenue
    S, Ssrc = pick_value(
        series,
        keys_exact=["Total Revenue", "TotalRevenue", "Revenue"],
        keys_regex=[r"total revenue", r"^revenue$"],
    )

    # COGS (or derive from Gross Profit when necessary)
    COGS, COGSsrc = pick_value(
        series,
        keys_exact=["Cost Of Revenue", "CostOfRevenue", "Cost of Goods Sold", "CostOfGoodsSold", "Cost Of Goods Sold"],
        keys_regex=[r"cost of revenue", r"cost of goods"],
    )

    if pd.isna(COGS) and pd.notna(S):
        gross, gross_src = pick_value(
            series,
            keys_exact=["Gross Profit", "GrossProfit"],
            keys_regex=[r"gross profit"],
        )
        if pd.notna(gross):
            COGS = float(max(S - gross, 0.0))
            COGSsrc = f"{Ssrc}-{gross_src}"

    # Operating expenses: prefer explicit OPEX, else sum typical sub-items, else derive from Operating Income
    OPEX, OPEXsrc = pick_value(
        series,
        keys_exact=["Operating Expense", "OperatingExpense", "Total Operating Expenses", "TotalOperatingExpenses"],
        keys_regex=[r"operating expense", r"total operating expenses"],
    )
    if pd.isna(OPEX):
        # try sum of SG&A, R&D, marketing, etc.
        sga, sga_src = pick_value(series, ["Selling General And Administration", "SellingGeneralAndAdministration", "Selling And Marketing Expenses", "SellingAndMarketingExpenses", "General And Administrative Expense"], [r"selling.*general", r"marketing", r"general and administrative"])
        rnd, rnd_src = pick_value(series, ["Research And Development", "ResearchAndDevelopment"], [r"research.*development", r"r&d"])
        marketing, marketing_src = pick_value(series, ["Selling And Marketing Expense", "SellingAndMarketingExpense"], [r"marketing"])
        vals = []
        srcs = []
        for v, s in [(sga, sga_src), (rnd, rnd_src), (marketing, marketing_src)]:
            if pd.notna(v):
                vals.append(float(v))
                srcs.append(s)
        if len(vals) > 0:
            OPEX = float(np.nansum(vals))
            OPEXsrc = "+".join(srcs)

    # Interest
    I, Isrc = pick_value(
        series,
        keys_exact=["Interest Expense", "InterestExpense", "Interest Expense Non Operating", "InterestExpenseNonOperating", "Interest Exp"],
        keys_regex=[r"interest expense"],
    )

    NI, NIsrc = pick_value(
        series,
        keys_exact=["Net Income", "NetIncome", "Net Income Common Stockholders", "NetIncomeCommonStockholders"],
        keys_regex=[r"net income"],
    )

    Tax, Taxsrc = pick_value(
        series,
        keys_exact=["Tax Provision", "TaxProvision", "Income Tax Expense", "IncomeTaxExpense"],
        keys_regex=[r"tax provision|income tax"],
    )

    DA, DAsrc = pick_value(
        series,
        keys_exact=["Depreciation And Amortization", "DepreciationAndAmortization"],
        keys_regex=[r"depreciation and amort"],
    )

    out.update(
        {"S": S, "S_src": Ssrc,
         "COGS": COGS, "COGS_src": COGSsrc,
         "OPEX": OPEX, "OPEX_src": OPEXsrc,
         "I": I, "I_src": Isrc,
         "NI": NI, "NI_src": NIsrc,
         "Tax": Tax, "Tax_src": Taxsrc,
         "DA": DA, "DA_src": DAsrc}
    )
    return out


# ---------------------------
# Statement extraction: Cash Flow
# ---------------------------
def extract_cashflow_one_period(series: pd.Series) -> Dict[str, object]:
    """Extract financing and investment cash-flow drivers."""
    out: Dict[str, object] = {}

    # Depreciation & amortization often appears here even if missing in income statement.
    DA_cf, DA_cf_src = pick_value(
        series,
        keys_exact=["Depreciation", "Depreciation And Amortization", "DepreciationAndAmortization", "ReconciledDepreciation", "Depreciation & Amortization"],
        keys_regex=[r"depreciation"],
    )

    equity_raw, equity_src = sum_values(
        series,
        keys_exact=["Issuance Of Stock", "IssuanceOfStock", "Sale Of Stock", "SaleOfStock",
                    "Common Stock Issuance", "CommonStockIssuance",
                    "Issuance Of Capital Stock", "IssuanceOfCapitalStock"],
        keys_regex=[r"issuance.*stock|sale of stock|common stock issuance"],
    )
    debt_iss_raw, debt_iss_src = sum_values(
        series,
        keys_exact=["Issuance Of Debt", "IssuanceOfDebt", "Long Term Debt Issuance", "LongTermDebtIssuance",
                    "Proceeds From Issuance Of Long Term Debt", "ProceedsFromIssuanceOfLongTermDebt"],
        keys_regex=[r"issuance of debt|proceeds.*long term debt"],
    )
    repay_raw, repay_src = sum_values(
        series,
        keys_exact=["Repayment Of Debt", "RepaymentOfDebt", "Long Term Debt Payments", "LongTermDebtPayments",
                    "Repayments Of Debt", "RepaymentsOfDebt"],
        keys_regex=[r"repay(ment|ments).*debt|debt payments"],
    )

    capex_raw, capex_src = pick_value(
        series,
        keys_exact=["Capital Expenditure", "CapitalExpenditure", "Capital Expenditures", "CapitalExpenditures"],
        keys_regex=[r"capital expend"],
    )
    div_raw, div_src = pick_value(
        series,
        keys_exact=["Dividends Paid", "DividendsPaid", "Cash Dividends Paid", "CashDividendsPaid"],
        keys_regex=[r"dividends paid"],
    )
    repurch_raw, repurch_src = sum_values(
        series,
        keys_exact=["Repurchase Of Stock", "RepurchaseOfStock", "Repurchase Of Capital Stock", "RepurchaseOfCapitalStock"],
        keys_regex=[r"repurchase.*stock|repurchase.*capital"],
    )

    # yfinance often stores CapEx/dividends as negative numbers (cash outflows).
    out.update(
        {
            "EquityIssues": np.nan if pd.isna(equity_raw) else float(max(equity_raw, 0.0)),
            "EquityIssues_src": equity_src,
            "NewDebt": np.nan if pd.isna(debt_iss_raw) else float(max(debt_iss_raw, 0.0)),
            "NewDebt_src": debt_iss_src,
            "Repay": np.nan if pd.isna(repay_raw) else float(abs(repay_raw)),
            "Repay_src": repay_src,
            "CapEx": np.nan if pd.isna(capex_raw) else float(abs(capex_raw)),
            "CapEx_src": capex_src,
            "Div": np.nan if pd.isna(div_raw) else float(abs(div_raw)),
            "Div_src": div_src,
            "Buyback": np.nan if pd.isna(repurch_raw) else float(abs(repurch_raw)),
            "Buyback_src": repurch_src,
            "DA_cf": DA_cf,
            "DA_cf_src": DA_cf_src,
        }
    )
    return out


# ---------------------------
# Build period tables and transitions
# ---------------------------
def build_period_table_from_statement(df_stmt: pd.DataFrame, freq: str, extractor_fn) -> pd.DataFrame:
    """Convert statement DataFrame (columns=period end) to tidy table (rows=period)."""
    df_stmt = _to_frame(df_stmt)
    if df_stmt is None or df_stmt.empty:
        return pd.DataFrame()

    # yfinance returns statement as rows=items, columns=dates
    cols = list(df_stmt.columns)
    rows = []
    for dt in cols:
        series = df_stmt[dt]
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        out = extractor_fn(series)
        out["date"] = pd.to_datetime(dt)
        out["freq"] = freq
        rows.append(out)
    return pd.DataFrame(rows)


def merge_statements(bsq: pd.DataFrame, incq: pd.DataFrame, cfq: pd.DataFrame) -> pd.DataFrame:
    """Outer merge on date/freq; preserve *_src columns."""
    parts = [p for p in [bsq, incq, cfq] if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    df = parts[0]
    for p in parts[1:]:
        df = pd.merge(df, p, on=["date", "freq"], how="outer", suffixes=("", "_dup"))
        dup_cols = [c for c in df.columns if c.endswith("_dup")]
        if dup_cols:
            df = df.drop(columns=dup_cols)
    return df


def compute_residuals_and_audits(df: pd.DataFrame) -> pd.DataFrame:
    """Compute implied equity and identity diagnostics."""
    if df is None or df.empty:
        return df

    # Ensure columns exist
    for c in ["C", "AR", "Inv", "K", "AP", "STD", "LTD", "TA", "TL", "TCA", "TCL", "E_gross_report", "E_report", "MI_report"]:
        if c not in df.columns:
            df[c] = np.nan

    # Masks for missingness
    for c in ["C","AR","Inv","K","AP","STD","LTD","TA","TL","TCA","TCL","E_gross_report","S","COGS","OPEX","I","NI","Tax","DA","CapEx","EquityIssues","NewDebt","Repay","Div","Buyback"]:
        if c not in df.columns:
            df[c] = np.nan
        df[f"mask_{c}"] = df[c].notna().astype(int)

    # Fill DA from cashflow when income statement missing
    if "DA_cf" in df.columns:
        df["DA"] = df["DA"].where(df["DA"].notna(), df["DA_cf"])

    # Implied equity: TA - TL if both known
    df["E_implied"] = df["TA"] - df["TL"]
    df["mask_E_implied"] = (df["TA"].notna() & df["TL"].notna()).astype(int)

    # Identity residuals
    df["audit_A_minus_L_plus_E"] = df["TA"] - (df["TL"] + df["E_gross_report"])
    df["audit_A_minus_L_plus_E_implied"] = df["TA"] - (df["TL"] + df["E_implied"])

    # Split residual categories (if possible)
    # OCA = TCA - (C+AR+Inv)
    df["OCA_implied"] = df["TCA"] - (df["C"] + df["AR"] + df["Inv"])
    df["ONCA_implied"] = (df["TA"] - df["TCA"]) - df["K"]

    # OCL = TCL - (AP+STD)
    df["OCL_implied"] = df["TCL"] - (df["AP"] + df["STD"])
    df["ONCL_implied"] = (df["TL"] - df["TCL"]) - df["LTD"]

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    return df


def make_transition_table(df_periods: pd.DataFrame) -> pd.DataFrame:
    """Create transitions: one row per (t-1 -> t) within the same ticker and frequency."""
    if df_periods is None or df_periods.empty:
        return pd.DataFrame()

    df = df_periods.sort_values("date").reset_index(drop=True).copy()

    y_fields = ["C", "AR", "Inv", "K", "AP", "STD", "LTD", "TA", "TL", "TCA", "TCL", "E_gross_report", "E_implied"]
    x_fields = ["S", "COGS", "OPEX", "I", "NI", "Tax", "DA", "CapEx", "EquityIssues", "NewDebt", "Repay", "Div", "Buyback"]

    for f in y_fields + x_fields:
        if f not in df.columns:
            df[f] = np.nan
    for f in y_fields:
        mc = f"mask_{f}"
        if mc not in df.columns:
            df[mc] = df[f].notna().astype(int)

    rows: List[Dict[str, object]] = []
    for i in range(1, len(df)):
        r_prev = df.iloc[i - 1]
        r_next = df.iloc[i]
        row: Dict[str, object] = {
            "date_prev": str(pd.to_datetime(r_prev["date"]).date()),
            "date_next": str(pd.to_datetime(r_next["date"]).date()),
            "freq": r_next.get("freq", ""),
        }
        # drivers x_t: use the period ending at date_next
        for xf in x_fields:
            row[xf] = r_next.get(xf)

        # y_{t-1} and y_t
        for yf in y_fields:
            row[f"{yf}_prev"] = r_prev.get(yf)
            row[f"{yf}_next"] = r_next.get(yf)
            row[f"mask_{yf}"] = r_next.get(f"mask_{yf}", int(pd.notna(r_next.get(yf))))

        # convenience equity aliases
        row["E_prev"] = row.get("E_gross_report_prev")
        if pd.isna(row["E_prev"]):
            row["E_prev"] = row.get("E_implied_prev")

        row["E_next"] = row.get("E_gross_report_next")
        if pd.isna(row["E_next"]):
            row["E_next"] = row.get("E_implied_next")

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------
# Main: read universe and build data
# ---------------------------
def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, p)
    if os.path.exists(cand):
        return cand
    # also allow running from project root
    cand2 = os.path.join(os.getcwd(), p)
    return cand2


def main():
    data_csv_path = _resolve_path(DATA_CSV)
    df_u = pd.read_csv(data_csv_path)

    # Normalize column naming (provided file uses 'Lable')
    if "label" not in df_u.columns and "Lable" in df_u.columns:
        df_u = df_u.rename(columns={"Lable": "label"})
    if "label" not in df_u.columns:
        df_u["label"] = np.nan

    df_u["label"] = df_u["label"].astype(str).replace({"nan": ""}).str.strip()
    df_u.loc[df_u["label"] == "", "label"] = "train"

    required = {"ticker", "sector", "label"}
    missing = required - set(df_u.columns)
    if missing:
        raise ValueError(f"DATA_CSV must contain columns {sorted(required)}; missing={sorted(missing)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    periods_dir = os.path.join(OUT_DIR, "periods")
    trans_dir = os.path.join(OUT_DIR, "transitions")
    os.makedirs(periods_dir, exist_ok=True)
    os.makedirs(trans_dir, exist_ok=True)

    # Sector mapping
    sectors = sorted({str(s) for s in df_u["sector"].dropna().unique().tolist()})
    sector_to_id = {s: i for i, s in enumerate(sectors)}

    # Persist universe (for downstream scripts)
    df_u = df_u.copy()
    df_u["sector_id"] = df_u["sector"].map(sector_to_id).astype(int)
    df_u.to_csv(os.path.join(OUT_DIR, "universe.csv"), index=False)

    # yfinance attribute candidates (annual only)
    BS_A_ATTRS = ["balance_sheet", "balancesheet"]
    INC_A_ATTRS = ["income_stmt", "incomestmt", "financials"]
    CF_A_ATTRS = ["cashflow", "cash_flow", "cashflow_stmt"]

    panel_rows: List[pd.DataFrame] = []
    trans_rows: List[pd.DataFrame] = []
    build_rows: List[Dict[str, object]] = []

    for _, row_u in df_u.iterrows():
        ticker = str(row_u["ticker"])
        sector = str(row_u["sector"])
        label = str(row_u["label"])
        sector_id = int(row_u["sector_id"])

        safe_name = ticker.replace("/", "_")
        out_period_path = os.path.join(periods_dir, f"{safe_name}.csv")
        out_trans_path = os.path.join(trans_dir, f"{safe_name}.csv")

        if (not REBUILD) and os.path.exists(out_period_path) and os.path.exists(out_trans_path):
            # lightweight logging
            build_rows.append({"ticker": ticker, "sector": sector, "label": label, "sector_id": sector_id, "n_periods": np.nan, "n_trans": np.nan, "skipped": 1})
            continue

        # fetch annual statements
        bs_a, err_bs_a = fetch_with_retries(ticker, BS_A_ATTRS, retries=RETRIES, pause_s=PAUSE_S)
        inc_a, err_inc_a = fetch_with_retries(ticker, INC_A_ATTRS, retries=RETRIES, pause_s=PAUSE_S)
        cf_a, err_cf_a = fetch_with_retries(ticker, CF_A_ATTRS, retries=RETRIES, pause_s=PAUSE_S)

        # build annual period table
        bsa = build_period_table_from_statement(bs_a, freq="A", extractor_fn=extract_balance_sheet_one_period)
        inca = build_period_table_from_statement(inc_a, freq="A", extractor_fn=extract_income_one_period)
        cfa = build_period_table_from_statement(cf_a, freq="A", extractor_fn=extract_cashflow_one_period)

        df_all = merge_statements(bsa, inca, cfa)
        if df_all.empty:
            build_rows.append(
                {
                    "ticker": ticker,
                    "sector": sector,
                    "label": label,
                    "sector_id": sector_id,
                    "n_periods": 0,
                    "n_trans": 0,
                    "fetch_err_bs_a": err_bs_a,
                    "fetch_err_inc_a": err_inc_a,
                    "fetch_err_cf_a": err_cf_a,
                    "skipped": 0,
                }
            )
            continue

        # post-process
        df_all = df_all.sort_values(["date", "freq"]).reset_index(drop=True)
        df_all.insert(0, "ticker", ticker)
        df_all.insert(1, "sector", sector)
        df_all.insert(2, "label", label)
        df_all.insert(3, "sector_id", sector_id)
        df_all["freq_id"] = 1  # annual

        df_all = compute_residuals_and_audits(df_all)
        df_all = add_time_features(df_all)

        # Export per-ticker annual periods
        df_all.to_csv(out_period_path, index=False)

        # Transition table for y_t = f(x_t, y_{t-1}) within annual block
        df_trans = make_transition_table(df_all[df_all["freq"] == "A"].copy())
        if not df_trans.empty:
            df_trans["ticker"] = ticker
            df_trans["sector"] = sector
            df_trans["label"] = label
            df_trans["sector_id"] = sector_id
            df_trans["freq"] = "A"
            front = ["ticker", "sector", "label", "sector_id", "freq"]
            rest = [c for c in df_trans.columns if c not in front]
            df_trans = df_trans[front + rest]
            df_trans.to_csv(out_trans_path, index=False)
        else:
            # still write an empty transition to mark completion
            pd.DataFrame().to_csv(out_trans_path, index=False)

        panel_rows.append(df_all)
        if not df_trans.empty:
            trans_rows.append(df_trans)

        build_rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "label": label,
                "sector_id": sector_id,
                "n_periods": int(len(df_all)),
                "n_trans": int(len(df_trans)),
                "fetch_err_bs_a": err_bs_a,
                "fetch_err_inc_a": err_inc_a,
                "fetch_err_cf_a": err_cf_a,
                "skipped": 0,
            }
        )

        time.sleep(PAUSE_S)

    # Save panels and build log
    df_log = pd.DataFrame(build_rows)
    df_log.to_csv(os.path.join(OUT_DIR, "build_log.csv"), index=False)

    df_panel = pd.concat(panel_rows, ignore_index=True) if panel_rows else pd.DataFrame()
    df_panel.to_csv(os.path.join(OUT_DIR, "panel_periods_A.csv"), index=False)

    df_trans_panel = pd.concat(trans_rows, ignore_index=True) if trans_rows else pd.DataFrame()
    df_trans_panel.to_csv(os.path.join(OUT_DIR, "panel_transitions_A.csv"), index=False)

    print(f"[OK] Data preparation complete. OUT_DIR={OUT_DIR}")
    print(f"     n_tickers={len(df_u)} | periods_rows={len(df_panel)} | trans_rows={len(df_trans_panel)}")


if __name__ == "__main__":
    main()
