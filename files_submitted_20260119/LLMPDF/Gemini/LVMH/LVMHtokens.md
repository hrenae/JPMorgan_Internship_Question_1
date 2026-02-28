# Financial Statement Tokens (Gemini Extraction)
**Source PDF:** `D:\JPMorgan\LLMPDF\Gemini\LVMH\lvmh_Financialdocuments-December31%2C2024.pdf`
**Detected pages (0-based):** income=10, balance=25, cashflow=26
**Current year:** 2024

## Raw Extract Sample (Income Statement)
- Profit from recurring operations: {2024: 19571.0, 2023: 22802.0, 2022: 21055.0}
- Other operating income and expenses: {2024: -664.0, 2023: -242.0, 2022: -54.0}
- Operating profit: {2024: 18907.0, 2023: 22560.0, 2022: 21001.0}
- Net financial income/(expense): {2024: -792.0, 2023: -935.0, 2022: -888.0}
- Income taxes: {2024: -5157.0, 2023: -5673.0, 2022: -5362.0}
- Net profit before minority interests: {2024: 12958.0, 2023: 15952.0, 2022: 14751.0}
- Minority interests: {2024: -408.0, 2023: -778.0, 2022: -667.0}
- Net profit, Group share: {2024: 12550.0, 2023: 15174.0, 2022: 14084.0}

## Raw Extract Sample (Balance Sheet)
- Brands and other intangible assets: {2024: 26280.0, 2023: 25589.0, 2022: 25432.0}
- Goodwill: {2024: 20307.0, 2023: 24022.0, 2022: 24782.0}
- Property, plant and equipment: {2024: 29886.0, 2023: 27331.0, 2022: 23055.0}
- Right‑of‑use assets: {2024: 16620.0, 2023: 15679.0, 2022: 14615.0}
- Investments in joint ventures and associates: {2024: 1343.0, 2023: 991.0, 2022: 1066.0}
- Non‑current available for sale financial assets: {2024: 1632.0, 2023: 1363.0, 2022: 1109.0}
- Other non‑current assets: {2024: 1106.0, 2023: 1017.0, 2022: 1186.0}
- Deferred tax: {2024: 7344.0, 2023: 7012.0, 2022: 6952.0}
- Non‑current assets: {2024: 101719.0, 2023: 99984.0, 2022: 94906.0}
- Inventories and work in progress: {2024: 23669.0, 2023: 22952.0, 2022: 20319.0}
- Trade accounts receivable: {2024: 4730.0, 2023: 4728.0, 2022: 4258.0}
- Income taxes: {2024: 1231.0, 2023: 1148.0, 2022: 1211.0}
- Other current assets: {2024: 8455.0, 2023: 7723.0, 2022: 7488.0}
- Cash and cash equivalents: {2024: 9631.0, 2023: 7774.0, 2022: 7300.0}
- Current assets: {2024: 47471.0, 2023: 43710.0, 2022: 39740.0}
- Total assets: {2024: 149190.0, 2023: 143694.0, 2022: 134646.0}
- Equity, Group share: {2024: 67517.0, 2023: 61017.0, 2022: 55111.0}
- Minority interests: {2024: 1770.0, 2023: 1684.0, 2022: 1493.0}
- Equity: {2024: 69287.0, 2023: 62701.0, 2022: 56604.0}
- Long‑term borrowings: {2024: 12091.0, 2023: 11227.0, 2022: 10380.0}
- Non‑current lease liabilities: {2024: 14860.0, 2023: 13810.0, 2022: 12776.0}
- Non‑current provisions and other liabilities: {2024: 3856.0, 2023: 3880.0, 2022: 3902.0}
- Purchase commitments for minority interests’ shares: {2024: 8056.0, 2023: 11919.0, 2022: 12489.0}
- Non‑current liabilities: {2024: 46207.0, 2023: 47848.0, 2022: 46498.0}
- Short‑term borrowings: {2024: 10851.0, 2023: 10680.0, 2022: 9359.0}
- Current lease liabilities: {2024: 2972.0, 2023: 2728.0, 2022: 2632.0}
- Trade accounts payable: {2024: 8630.0, 2023: 9049.0, 2022: 8788.0}
- Current provisions and other liabilities: {2024: 10012.0, 2023: 9540.0, 2022: 9553.0}
- Current liabilities: {2024: 33696.0, 2023: 33145.0, 2022: 31543.0}
- Total liabilities and equity: {2024: 149190.0, 2023: 143694.0, 2022: 134646.0}

## Hard Accounting Constraints
CONSTRAINT|id=Eimp_identity|type=hard|expr=E_imp = TA - TL
CONSTRAINT|id=nonnegativity|type=hard|expr=C,AR,Inv,K,AP,STD,LTD,TA,TL >= 0
CONSTRAINT|id=assets_aggregation|type=hard|expr=TA >= C + AR + Inv + K  # allow OtherAssets >= 0
CONSTRAINT|id=liab_aggregation|type=hard|expr=TL >= AP + STD + LTD      # allow OtherLiab >= 0

## Canonical Tokens (y_t and x_t)
### Balance Sheet y_t
TOKEN|kind=BS|field=C|year=2024|value=9631|unit=reported_unit
TOKEN|kind=BS|field=AR|year=2024|value=4730|unit=reported_unit
TOKEN|kind=BS|field=Inv|year=2024|value=23669|unit=reported_unit
TOKEN|kind=BS|field=K|year=2024|value=29886|unit=reported_unit
TOKEN|kind=BS|field=AP|year=2024|value=8630|unit=reported_unit
TOKEN|kind=BS|field=STD|year=2024|value=null|unit=reported_unit
TOKEN|kind=BS|field=LTD|year=2024|value=null|unit=reported_unit
TOKEN|kind=BS|field=TA|year=2024|value=149190|unit=reported_unit
TOKEN|kind=BS|field=TL|year=2024|value=null|unit=reported_unit
TOKEN|kind=BS|field=EQ_REPORT|year=2024|value=null|unit=reported_unit

### Income Statement / Drivers x_t
TOKEN|kind=IS|field=REV|year=2024|value=null|unit=reported_unit
TOKEN|kind=IS|field=TOTAL_COSTS|year=2024|value=null|unit=reported_unit
TOKEN|kind=IS|field=NI|year=2024|value=null|unit=reported_unit
TOKEN|kind=IS|field=EBIT|year=2024|value=-664|unit=reported_unit
TOKEN|kind=IS|field=INTEREST_EXP|year=2024|value=null|unit=reported_unit

## Constraint Diagnostics (current year)
- E_imp (computed) = None
- Residual OtherAssets = TA-(C+AR+Inv+K) = 81274.0
- Residual OtherLiab   = TL-(AP+STD+LTD) = None

## Computed Metrics (Python)
```json
{
  "current_year": 2024,
  "net_income": null,
  "cost_to_income_ratio": null,
  "quick_ratio": -0.4883026381284221,
  "debt_to_equity": null,
  "debt_to_assets": null,
  "debt_to_capital": null,
  "debt_to_ebitda": null,
  "interest_coverage_ratio": null,
  "intermediate": {
    "REV": null,
    "TOTAL_COSTS": null,
    "DEBT": null,
    "EBIT": -664.0,
    "DA": null,
    "EBITDA": null,
    "TA": 149190.0,
    "TL": null,
    "EQ_REPORT": null,
    "E_imp": null,
    "residual_assets_other": 81274.0,
    "residual_liab_other": null
  }
}
```
