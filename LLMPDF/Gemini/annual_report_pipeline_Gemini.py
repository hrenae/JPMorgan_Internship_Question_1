#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""annual_report_pipeline_gemini.py

End-to-end pipeline (Migrated to Gemini API):
  PDF -> locate statement pages -> slice -> Gemini 1.5 extracts tables (JSON)
  -> Python computes ratios -> tokens.md with accounting-identity constraints
  -> Gemini 1.5 drafts answers + CFO/CEO recommendations

Key changes from OpenAI version:
  - Uses `google.generativeai` SDK.
  - Uses Gemini File API for PDF ingestion (upload -> analyze -> delete).
  - Enforces JSON output via `response_mime_type`.

Run:
  python annual_report_pipeline_gemini.py --pdf 2023GM_annual_report.pdf --model gemini-1.5-pro

Requirements:
  pip install --upgrade google-generativeai pypdf
  (and set GOOGLE_API_KEY)
"""

import argparse
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os
#  Clash 软件是 7890
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY']  = 'http://127.0.0.1:7890'

import google.generativeai as genai
from pypdf import PdfReader, PdfWriter

# -----------------------------
# 0) Generic helpers
# -----------------------------
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def json_from_text(maybe_json_text: str) -> Dict[str, Any]:
    """Extract a JSON object from model output."""
    t = (maybe_json_text or "").strip()
    if not t:
        raise ValueError("Empty model output; cannot parse JSON.")

    # Remove markdown code blocks if present
    if t.startswith("```json"):
        t = t.replace("```json", "", 1)
    if t.startswith("```"):
        t = t.replace("```", "", 1)
    if t.endswith("```"):
        t = t[:-3]
    t = t.strip()

    # direct
    try:
        return json.loads(t)
    except Exception:
        pass

    # locate outermost object
    i = t.find("{")
    j = t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i : j + 1])
        except Exception:
            pass

    raise ValueError("Failed to parse JSON from model output.")


# -----------------------------
# 1) Gemini Setup & File Helper
# -----------------------------

def upload_pdf_to_gemini(path: Path):
    """Uploads the PDF to Gemini File API and waits for processing."""
    print(f"... Uploading {path.name} to Gemini ...")
    file_ref = genai.upload_file(path, mime_type="application/pdf")
    
    # Wait for processing (usually instant for small/sliced PDFs)
    while file_ref.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(1)
        file_ref = genai.get_file(file_ref.name)
    
    if file_ref.state.name == "FAILED":
        raise ValueError("Gemini failed to process the PDF file.")
        
    return file_ref


# -----------------------------
# 2) Locate statement pages (cheap heuristic on extracted text)
# -----------------------------
@dataclass
class StatementPages:
    income_page: int
    balance_page: int
    cashflow_page: Optional[int] = None


def page_score(text: str, kind: str) -> int:
    t = norm(text)
    score = 0

    if kind == "income":
        keys = [
            ("income statement", 8),
            ("statements of income", 8),
            ("net income", 6),
            ("operating income", 4),
            ("revenue", 3),
            ("sales", 2),
        ]
    elif kind == "balance":
        keys = [
            ("balance sheet", 8),
            ("balance sheets", 8),
            ("total assets", 6),
            ("total liabilities", 4),
            ("total equity", 3),
            ("current assets", 2),
        ]
    elif kind == "cashflow":
        keys = [
            ("cash flows", 8),
            ("statements of cash flows", 8),
            ("operating activities", 4),
            ("depreciation", 2),
            ("amortization", 2),
        ]
    else:
        keys = []

    for kw, w in keys:
        if kw in t:
            score += w

    # numeric density boost
    nums = len(re.findall(r"[\(\-]?\$?\d[\d,]*\.?\d*\)?", t))
    score += min(nums // 40, 6)

    return score


def find_statement_pages(pdf_path: Path, max_pages_scan: int = 300) -> StatementPages:
    reader = PdfReader(str(pdf_path))
    n = min(len(reader.pages), max_pages_scan)

    best_income = (-1, 0)
    best_balance = (-1, 0)
    best_cf = (-1, None)

    for i in range(n):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:
            text = ""

        si = page_score(text, "income")
        sb = page_score(text, "balance")
        sc = page_score(text, "cashflow")

        if si > best_income[0]:
            best_income = (si, i)
        if sb > best_balance[0]:
            best_balance = (sb, i)
        if sc > best_cf[0]:
            best_cf = (sc, i)

    cashflow_page = best_cf[1] if (best_cf[1] is not None and best_cf[0] >= 8) else None

    return StatementPages(
        income_page=best_income[1],
        balance_page=best_balance[1],
        cashflow_page=cashflow_page,
    )


# -----------------------------
# 3) Slice PDF to key pages
# -----------------------------

def slice_pdf(pdf_path: Path, page_indices: List[int]) -> Path:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for idx in page_indices:
        writer.add_page(reader.pages[idx])

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    with open(tmp.name, "wb") as f:
        writer.write(f)
    return Path(tmp.name)


# -----------------------------
# 4) Gemini extraction (tables as JSON)
# -----------------------------
EXTRACT_PROMPT = r"""
You are given a sliced annual report PDF with ONLY key financial statement pages.

Assume slice pages (0-based within slice):
- slice page 0: Income statement
- slice page 1: Balance sheet
- slice page 2: Cash flow statement (if present)

Task:
1) Extract ALL line items and their values by year from the income statement and balance sheet.
2) If cash flow is present, extract at least the line(s) needed for depreciation/amortization (for EBITDA).
3) Preserve sign: parentheses mean negative.
4) Do NOT invent missing rows/values; use null when not shown.

Return STRICT JSON ONLY with schema:
{
  "meta": {"currency": string|null, "units": string|null, "years": [int, ...]},
  "income_statement_rows": [{"label": string, "values": {"YYYY": number|null, ...}}],
  "balance_sheet_rows":   [{"label": string, "values": {"YYYY": number|null, ...}}],
  "cashflow_rows":        [{"label": string, "values": {"YYYY": number|null, ...}}] | []
}

Notes:
- The "years" list must include all years visible across the statements.
- Values must be in the reported units (do not rescale).
"""


def rows_list_to_table(rows: List[Dict[str, Any]]) -> Dict[str, Dict[int, float]]:
    """Convert rows list into label -> {year:int -> value:float}, dropping null values."""
    out: Dict[str, Dict[int, float]] = {}
    for r in rows or []:
        label = (r.get("label") or "").strip()
        values = r.get("values") or {}
        if not label or not isinstance(values, dict):
            continue
        ymap: Dict[int, float] = {}
        for ys, v in values.items():
            try:
                y = int(str(ys).strip())
            except Exception:
                continue
            fv = safe_float(v)
            if fv is None:
                continue
            ymap[y] = float(fv)
        if ymap:
            out[label] = ymap
    return out


def gemini_extract_tables_from_sliced_pdf(sliced_pdf_path: Path, model_name: str) -> Dict[str, Any]:
    # 1. Upload File
    file_ref = upload_pdf_to_gemini(sliced_pdf_path)
    
    try:
        # 2. Configure Model
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # 3. Generate Content
        # We pass the file_ref directly in the list
        response = model.generate_content([file_ref, EXTRACT_PROMPT])
        
        # 4. Parse
        return json_from_text(response.text)
        
    finally:
        # 5. Clean up remote file
        print(f" Cleaning up {file_ref.name}...")
        file_ref.delete()


# -----------------------------
# 5) Canonical mapping (regex) + metrics
# -----------------------------
CANONICAL_BS = {
    "C": [r"\bcash\b", r"cash and cash equivalents"],
    "AR": [r"accounts receivable", r"trade receivables", r"\breceivables\b"],
    "Inv": [r"\binventor(y|ies)\b"],
    "K": [r"property, plant", r"net ppe", r"\bppe\b", r"fixed assets"],
    "AP": [r"accounts payable", r"trade payables", r"\bpayables\b"],
    "STD": [r"short[- ]term debt", r"short[- ]term borrowings", r"current portion of long[- ]term debt"],
    "LTD": [r"long[- ]term debt", r"noncurrent debt", r"long[- ]term borrowings"],
    "TA": [r"total assets"],
    "TL": [r"total liabilities(?! and equity)"],
    "EQ_REPORT": [r"total equity", r"total shareholders'? equity", r"stockholders'? equity"],
    "TCA": [r"total current assets", r"\bcurrent assets\b"],
    "TCL": [r"total current liabilities", r"\bcurrent liabilities\b"],
}

CANONICAL_IS = {
    "REV": [r"total (net )?sales", r"total revenue", r"net revenue", r"\brevenue\b", r"\bsales\b"],
    "TOTAL_COSTS": [r"total costs and expenses", r"total expenses", r"total operating expenses"],
    "NI": [r"net income", r"net earnings", r"profit for the year"],
    "EBIT": [r"operating income", r"operating profit"],
    "INTEREST_EXP": [r"interest expense", r"finance costs", r"interest and other finance costs"],
}

CANONICAL_CF = {
    "DA": [r"depreciation", r"amortization", r"depreciation and amortization"],
}


def best_match_value(table: Dict[str, Dict[int, float]], patterns: List[str], target_year: int) -> Optional[float]:
    for label, ymap in table.items():
        l = norm(label)
        for p in patterns:
            if re.search(p, l):
                if target_year in ymap:
                    return ymap[target_year]
                # fallback to latest year in that row
                if ymap:
                    return ymap[sorted(ymap.keys())[-1]]
    return None


def canonicalize_tables(bs_raw, is_raw, cf_raw) -> Tuple[dict, dict, dict]:
    years = set()
    for tb in [bs_raw, is_raw, cf_raw]:
        for _, ymap in tb.items():
            years |= set(ymap.keys())
    real_years = sorted([y for y in years if y >= 1900])
    current_year = max(real_years) if real_years else max(years)

    bs_can = {"current_year": current_year}
    is_can: Dict[str, Dict[int, Optional[float]]] = {}
    cf_can: Dict[str, Dict[int, Optional[float]]] = {}

    for f, pats in CANONICAL_BS.items():
        out = {}
        for y in real_years if real_years else [current_year]:
            out[y] = best_match_value(bs_raw, pats, y)
        bs_can[f] = out

    for f, pats in CANONICAL_IS.items():
        out = {}
        for y in real_years if real_years else [current_year]:
            out[y] = best_match_value(is_raw, pats, y)
        is_can[f] = out

    if cf_raw:
        for f, pats in CANONICAL_CF.items():
            out = {}
            for y in real_years if real_years else [current_year]:
                out[y] = best_match_value(cf_raw, pats, y)
            cf_can[f] = out

    return bs_can, is_can, cf_can


def compute_metrics(bs: dict, is_: dict, cf: dict) -> dict:
    y = bs["current_year"]

    NI = is_["NI"].get(y)
    REV = is_["REV"].get(y)
    TOTAL_COSTS = is_["TOTAL_COSTS"].get(y)

    C = bs["C"].get(y)
    AR = bs["AR"].get(y)
    Inv = bs["Inv"].get(y)
    K = bs["K"].get(y)
    AP = bs["AP"].get(y)
    STD = bs["STD"].get(y)
    LTD = bs["LTD"].get(y)
    TA = bs["TA"].get(y)
    TL = bs["TL"].get(y)
    EQ = bs.get("EQ_REPORT", {}).get(y)

    E_imp = (TA - TL) if (TA is not None and TL is not None) else None

    cost_to_income = safe_div(TOTAL_COSTS, REV)

    TCA = bs.get("TCA", {}).get(y)
    TCL = bs.get("TCL", {}).get(y)
    if TCA is not None and TCL is not None:
        quick_ratio = safe_div((TCA - (Inv or 0.0)), TCL)
    elif TCL is not None:
        quick_ratio = safe_div(((C or 0.0) + (AR or 0.0)), TCL)
    else:
        quick_ratio = None

    debt = None
    if STD is not None or LTD is not None:
        debt = (STD or 0.0) + (LTD or 0.0)

    equity_for_ratios = EQ if EQ is not None else E_imp

    debt_to_equity = safe_div(debt, equity_for_ratios)
    debt_to_assets = safe_div(debt, TA)
    debt_to_capital = safe_div(debt, (debt + equity_for_ratios) if (debt is not None and equity_for_ratios is not None) else None)

    EBIT = is_["EBIT"].get(y)
    interest_exp = is_["INTEREST_EXP"].get(y)

    DA = cf.get("DA", {}).get(y)
    EBITDA = (EBIT + DA) if (EBIT is not None and DA is not None) else None

    debt_to_ebitda = safe_div(debt, EBITDA)
    interest_coverage = safe_div(EBIT, abs(interest_exp) if interest_exp is not None else None)

    residual_assets = None
    residual_liab = None
    if TA is not None and all(v is not None for v in [C, AR, Inv, K]):
        residual_assets = TA - (C + AR + Inv + K)
    if TL is not None and all(v is not None for v in [AP, STD, LTD]):
        residual_liab = TL - (AP + STD + LTD)

    return {
        "current_year": y,
        "net_income": NI,
        "cost_to_income_ratio": cost_to_income,
        "quick_ratio": quick_ratio,
        "debt_to_equity": debt_to_equity,
        "debt_to_assets": debt_to_assets,
        "debt_to_capital": debt_to_capital,
        "debt_to_ebitda": debt_to_ebitda,
        "interest_coverage_ratio": interest_coverage,
        "intermediate": {
            "REV": REV,
            "TOTAL_COSTS": TOTAL_COSTS,
            "DEBT": debt,
            "EBIT": EBIT,
            "DA": DA,
            "EBITDA": EBITDA,
            "TA": TA,
            "TL": TL,
            "EQ_REPORT": EQ,
            "E_imp": E_imp,
            "residual_assets_other": residual_assets,
            "residual_liab_other": residual_liab,
        },
    }


# -----------------------------
# 6) Token markdown writer (includes accounting constraints)
# -----------------------------

def build_tokens_md(
    pdf_path: Path,
    pages: StatementPages,
    bs_raw: Dict[str, Dict[int, float]],
    is_raw: Dict[str, Dict[int, float]],
    cf_raw: Dict[str, Dict[int, float]],
    bs_can: dict,
    is_can: dict,
    cf_can: dict,
    metrics: dict,
) -> str:
    y = metrics["current_year"]

    md: List[str] = []
    md.append("# Financial Statement Tokens (Gemini Extraction)\n")
    md.append(f"**Source PDF:** `{pdf_path}`\n")
    md.append(f"**Detected pages (0-based):** income={pages.income_page}, balance={pages.balance_page}, cashflow={pages.cashflow_page}\n")
    md.append(f"**Current year:** {y}\n\n")

    md.append("## Raw Extract Sample (Income Statement)\n")
    for k, v in list(is_raw.items())[:40]:
        md.append(f"- {k}: {v}\n")

    md.append("\n## Raw Extract Sample (Balance Sheet)\n")
    for k, v in list(bs_raw.items())[:40]:
        md.append(f"- {k}: {v}\n")

    if cf_raw:
        md.append("\n## Raw Extract Sample (Cash Flow)\n")
        for k, v in list(cf_raw.items())[:30]:
            md.append(f"- {k}: {v}\n")

    md.append("\n## Hard Accounting Constraints\n")
    md.append("CONSTRAINT|id=Eimp_identity|type=hard|expr=E_imp = TA - TL\n")
    md.append("CONSTRAINT|id=nonnegativity|type=hard|expr=C,AR,Inv,K,AP,STD,LTD,TA,TL >= 0\n")
    md.append("CONSTRAINT|id=assets_aggregation|type=hard|expr=TA >= C + AR + Inv + K  # allow OtherAssets >= 0\n")
    md.append("CONSTRAINT|id=liab_aggregation|type=hard|expr=TL >= AP + STD + LTD      # allow OtherLiab >= 0\n")

    def token_line(kind: str, field: str, year: int, value, unit: str = "reported_unit") -> str:
        v = "null" if value is None else f"{value:.6g}"
        return f"TOKEN|kind={kind}|field={field}|year={year}|value={v}|unit={unit}"

    md.append("\n## Canonical Tokens (y_t and x_t)\n")
    md.append("### Balance Sheet y_t\n")
    for f in ["C","AR","Inv","K","AP","STD","LTD","TA","TL","EQ_REPORT"]:
        md.append(token_line("BS", f, y, bs_can.get(f, {}).get(y)))
        md.append("\n")

    md.append("\n### Income Statement / Drivers x_t\n")
    for f in ["REV","TOTAL_COSTS","NI","EBIT","INTEREST_EXP"]:
        md.append(token_line("IS", f, y, is_can.get(f, {}).get(y)))
        md.append("\n")

    if cf_can.get("DA"):
        md.append("\n### Cash Flow (for EBITDA)\n")
        md.append(token_line("CF", "DA", y, cf_can.get("DA", {}).get(y)))
        md.append("\n")

    md.append("\n## Constraint Diagnostics (current year)\n")
    inter = metrics.get("intermediate", {})
    md.append(f"- E_imp (computed) = {inter.get('E_imp')}\n")
    md.append(f"- Residual OtherAssets = TA-(C+AR+Inv+K) = {inter.get('residual_assets_other')}\n")
    md.append(f"- Residual OtherLiab   = TL-(AP+STD+LTD) = {inter.get('residual_liab_other')}\n")

    md.append("\n## Computed Metrics (Python)\n")
    md.append("```json\n")
    md.append(json.dumps(metrics, indent=2))
    md.append("\n```\n")

    return "".join(md)


# -----------------------------
# 7) Gemini final answers
# -----------------------------

def call_gemini_for_answers(tokens_md: str, metrics: dict, model_name: str) -> str:
    y = metrics["current_year"]

    prompt = f"""
You are analyzing a company's annual report statements extracted from a PDF.

You are given:
(1) A markdown token file containing extracted statement values and explicit accounting constraints.
(2) Python-computed financial ratios (preferred as authoritative for arithmetic).

Your tasks:
A) Answer:
 i.   What's the net income of the current year ({y})?
 ii.  What's the cost-to-income ratio?
 iii. What's the quick ratio, debt-to-equity, debt-to-assets, debt-to-capital, debt-to-EBITDA?
 iv.  What's the interest coverage ratio?
B) Provide recommendations to the CFO/CEO based on these ratios:
   - liquidity risk, leverage risk, coverage, and operational efficiency.
   - explicitly mention any data limitations (e.g., missing items; EBITDA proxy; IFRS/US GAAP differences).
C) Ensure accounting consistency:
   - reference the constraint diagnostics (residuals) and interpret whether the extracted data seems internally coherent.

Return format:
1) A short bullet list with the numeric answers (include formulas in-line).
2) A concise recommendation section (5-10 bullets), prioritized by impact and feasibility.
""".strip()

    model = genai.GenerativeModel(model_name)
    
    # Simple text-based prompt sequence
    resp = model.generate_content([
        prompt,
        "=== TOKEN FILE (md) ===\n" + tokens_md
    ])

    return resp.text


# -----------------------------
# 8) Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to annual report PDF")
    ap.add_argument("--out_tokens", "--out_md", dest="out_tokens", default="tokens.md", help="Output tokens.md path")
    ap.add_argument("--model", default="gemini-1.5-pro", help="Gemini model name (e.g., gemini-1.5-pro, gemini-2.0-flash)")
    ap.add_argument("--no_llm", action="store_true", help="Skip Gemini final answer call; only generate tokens + metrics")
    ap.add_argument("--out_answer", default="answer.md", help="Where to save the LLM answer")
    ap.add_argument("--income_page", type=int, default=None, help="Override: income statement page index (0-based)")
    ap.add_argument("--balance_page", type=int, default=None, help="Override: balance sheet page index (0-based)")
    ap.add_argument("--cashflow_page", type=int, default=None, help="Override: cash flow statement page index (0-based or omit)")
    args = ap.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    
    genai.configure(api_key=api_key)

    pdf_path = Path(args.pdf).expanduser().resolve()

    # locate pages
    pages = find_statement_pages(pdf_path)
    if args.income_page is not None:
        pages.income_page = args.income_page
    if args.balance_page is not None:
        pages.balance_page = args.balance_page
    if args.cashflow_page is not None:
        pages.cashflow_page = args.cashflow_page

    # slice to reduce PDF size
    slice_pages = [pages.income_page, pages.balance_page]
    if pages.cashflow_page is not None:
        slice_pages.append(pages.cashflow_page)
    sliced_pdf = slice_pdf(pdf_path, slice_pages)

    # LLM extraction
    try:
        extracted = gemini_extract_tables_from_sliced_pdf(sliced_pdf, model_name=args.model)
    except Exception as e:
        # Cleanup temp file if error
        if sliced_pdf.exists():
            os.unlink(sliced_pdf)
        raise e

    # Cleanup local temp file
    if sliced_pdf.exists():
        os.unlink(sliced_pdf)

    meta = extracted.get("meta", {})
    years = []
    for y in (meta.get("years") or []):
        try:
            years.append(int(y))
        except Exception:
            pass
    years = sorted(set([y for y in years if y >= 1900]), reverse=True)

    # convert rows to table dicts
    is_raw = rows_list_to_table(extracted.get("income_statement_rows") or [])
    bs_raw = rows_list_to_table(extracted.get("balance_sheet_rows") or [])
    cf_raw = rows_list_to_table(extracted.get("cashflow_rows") or [])

    if not years:
        # fallback: infer years from extracted tables
        inferred = set()
        for tb in [is_raw, bs_raw, cf_raw]:
            for _, ymap in tb.items():
                inferred |= set([y for y in ymap.keys() if y >= 1900])
        years = sorted(inferred, reverse=True)

    if not years:
        raise RuntimeError("Cannot determine years from extracted statements.")

    # canonicalize + metrics
    bs_can, is_can, cf_can = canonicalize_tables(bs_raw, is_raw, cf_raw)
    current_year = bs_can["current_year"]

    bs_struct = {"current_year": current_year}
    for k in ["C", "AR", "Inv", "K", "AP", "STD", "LTD", "TA", "TL", "EQ_REPORT", "TCA", "TCL"]:
        bs_struct[k] = bs_can.get(k, {})

    is_struct = {k: is_can.get(k, {}) for k in ["REV", "TOTAL_COSTS", "NI", "EBIT", "INTEREST_EXP"]}
    cf_struct = {"DA": cf_can.get("DA", {})} if cf_can else {}

    metrics = compute_metrics(bs_struct, is_struct, cf_struct)

    tokens_md = build_tokens_md(
        pdf_path=pdf_path,
        pages=pages,
        bs_raw=bs_raw,
        is_raw=is_raw,
        cf_raw=cf_raw,
        bs_can=bs_can,
        is_can=is_can,
        cf_can=cf_can,
        metrics=metrics,
    )

    out_path = Path(args.out_tokens).resolve()
    out_path.write_text(tokens_md, encoding="utf-8")
    print(f"[OK] Tokens written to {out_path}")

    if args.no_llm:
        print("[OK] --no_llm set, skipping Gemini answer generation.")
        return

    answer = call_gemini_for_answers(tokens_md, metrics, model_name=args.model)
    out_ans = Path(args.out_answer).resolve()
    out_ans.write_text(answer, encoding="utf-8")
    print(f"[OK] Answer written to {out_ans}")


if __name__ == "__main__":
    main()