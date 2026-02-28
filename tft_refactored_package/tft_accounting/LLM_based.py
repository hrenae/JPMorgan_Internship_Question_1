from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .preprocessing import TARGET_SPECS
from .theory import (
    DT_MAX,
    DT_MIN,
    STATE_COLS,
    _get_first_finite,
    _get_logS,
    _has_full_truth_state,
    _is_financial_sector,
    _nan_to_zero_with_mask,
    _safe_name,
    _truth_state_from_row,
    ar1_next,
    fit_ar1,
    load_panel,
    simulate_step,
)


THETA_KEYS_EVAL = [
    "m_gross",
    "m_opex",
    "DSO",
    "DIO",
    "DPO",
    "alpha_OCA",
    "alpha_ONCA",
    "alpha_OCL",
    "alpha_ONCL",
    "kappa",
    "delta",
    "payout",
    "neteq_to_sales",
    "phi",
    "r_ST",
    "r_LT",
    "tau",
]
FLOW_KEYS_EVAL = ["COGS", "OPEX", "Tax", "NI", "Div", "Int", "TA", "TL", "NetEq"]
THETA_BOUNDS: Dict[str, Tuple[float, float, str]] = {
    spec.name: (
        float(0.0 if spec.lo is None else spec.lo),
        float(0.0 if spec.hi is None else spec.hi),
        str(spec.kind),
    )
    for spec in TARGET_SPECS
    if spec.name != "logS"
}


@dataclass
class LLMApiConfig:
    """Configuration for one OpenAI-compatible chat-completions endpoint.

    The implementation intentionally stays provider-agnostic. Any vendor that
    accepts a POST request shaped like a standard chat-completions API can be
    used by filling in ``base_url``, ``endpoint``, ``model``, and the API key.
    """

    enabled: bool = False
    base_url: str = "https://api.example.com/v1"
    endpoint: str = "/chat/completions"
    model: str = "your-model-name"
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: int = 120
    temperature: float = 0.0
    max_tokens: int = 3000
    extra_headers: Dict[str, str] = field(default_factory=dict)
    use_response_format_json: bool = True

    def resolve_api_key(self) -> str:
        if self.api_key.strip():
            return self.api_key.strip()
        env_name = self.api_key_env.strip()
        if env_name:
            return os.environ.get(env_name, "").strip()
        return ""

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.base_url.strip():
            raise ValueError("LLM API base_url is empty.")
        if not self.endpoint.strip():
            raise ValueError("LLM API endpoint is empty.")
        if not self.model.strip():
            raise ValueError("LLM model name is empty.")
        if not self.resolve_api_key():
            env_name = self.api_key_env.strip() or "<unset>"
            raise ValueError(
                "LLM API key is missing. Set api_key directly or export the environment variable "
                f"{env_name}."
            )


@dataclass
class LLMBacktestConfig:
    """Configuration for the LLM-based rolling backtester."""

    data_dir: str
    out_dir: str
    api: LLMApiConfig = field(default_factory=LLMApiConfig)
    mode: str = "backtest"
    warmup: int = 3
    min_ar1_points: int = 3
    save_one_file: bool = False
    disable_interest_for_banks: bool = False
    tickers: Optional[list[str] | str] = None
    prompt_history_window: int = 6
    rollout_context_window: int = 4
    save_raw_prompts: bool = True
    duplicate_tft_filename: bool = True
    continue_on_step_error: bool = True
    verbose: bool = True
    progress_every_calls: int = 1
    debug_preview_chars: int = 400
    retry_on_invalid_json: int = 1
    raw_json_subdir: str = "raw_json"
    save_full_raw_response: bool = True
    include_sanitizer_diagnostics: bool = True


class JsonRepository:
    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def dump(path: str, payload: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)


class OpenAICompatibleChatClient:
    """Minimal HTTP client for OpenAI-compatible chat APIs.

    If your vendor does not follow this schema, you usually only need to edit
    the ``complete_json`` method.
    """

    def __init__(self, config: LLMApiConfig) -> None:
        self.config = config
        self.config.validate()

    def _build_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/{self.config.endpoint.lstrip('/')}"

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.resolve_api_key()}",
            **self.config.extra_headers,
        }
        payload = {
            "model": self.config.model,
            "temperature": float(self.config.temperature),
            "max_tokens": int(self.config.max_tokens),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if bool(self.config.use_response_format_json):
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            self._build_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(self.config.timeout_s)) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            raise RuntimeError(f"LLM API HTTP error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API connection error: {exc}") from exc

        data = json.loads(raw)
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected LLM response schema: {data}") from exc

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif item.get("type") == "text" and isinstance(item.get("content"), str):
                        parts.append(item["content"])
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            content = "\n".join(parts)
        elif isinstance(content, (dict, tuple)):
            content = json.dumps(content, ensure_ascii=False)
        elif content is None:
            content = ""
        else:
            content = str(content)

        return {
            "request_payload": {
                "model": payload["model"],
                "temperature": payload["temperature"],
                "max_tokens": payload["max_tokens"],
                "response_format": payload.get("response_format"),
            },
            "raw_response": data,
            "text": content,
        }


class LLMResponseParser:
    @staticmethod
    def preview(text: Any, limit: int = 400) -> str:
        s = str(text if text is not None else "")
        s = s.replace("\n", "\\n")
        return s[:limit]

    @staticmethod
    def _normalize_text(text: Any) -> str:
        if isinstance(text, (dict, list)):
            return json.dumps(text, ensure_ascii=False)
        s = str(text if text is not None else "").strip().replace("﻿", "")
        s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        return s

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    @staticmethod
    def _extract_balanced_json_object(text: str) -> str:
        start = text.find("{")
        if start < 0:
            raise ValueError(f"Could not find a JSON object in the LLM output. preview={LLMResponseParser.preview(text)}")
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        raise ValueError(f"Could not find a balanced JSON object in the LLM output. preview={LLMResponseParser.preview(text)}")

    @classmethod
    def extract_json_text(cls, text: Any) -> str:
        stripped = cls._strip_code_fences(cls._normalize_text(text))
        return cls._extract_balanced_json_object(stripped)

    @staticmethod
    def _replace_char_outside_strings(text: str, src: str, dst: str) -> str:
        out: List[str] = []
        in_string = False
        escape = False
        for ch in text:
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                out.append(ch)
            elif ch == src:
                out.append(dst)
            else:
                out.append(ch)
        return "".join(out)

    @classmethod
    def _repair_common_json_issues(cls, text: str) -> List[str]:
        attempts: List[str] = []
        current = text
        attempts.append(current)

        repaired = re.sub(r",\s*([}\]])", r"\1", current)
        if repaired not in attempts:
            attempts.append(repaired)

        repaired2 = cls._replace_char_outside_strings(repaired, ";", ",")
        repaired2 = re.sub(r",\s*,+", ",", repaired2)
        repaired2 = re.sub(r",\s*([}\]])", r"\1", repaired2)
        if repaired2 not in attempts:
            attempts.append(repaired2)

        repaired3 = re.sub(r"//.*?$", "", repaired2, flags=re.MULTILINE)
        repaired3 = re.sub(r"/\*.*?\*/", "", repaired3, flags=re.DOTALL)
        repaired3 = re.sub(r",\s*,+", ",", repaired3)
        repaired3 = re.sub(r",\s*([}\]])", r"\1", repaired3)
        if repaired3 not in attempts:
            attempts.append(repaired3)

        return attempts

    @classmethod
    def parse(cls, text: Any) -> Dict[str, Any]:
        candidate = cls.extract_json_text(text)
        last_exc: Optional[Exception] = None
        for attempt in cls._repair_common_json_issues(candidate):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected dict JSON object, got {type(parsed).__name__}")
                return parsed
            except Exception as exc:
                last_exc = exc
        raise ValueError(
            f"Invalid JSON returned by model: {last_exc}; preview={cls.preview(candidate)}"
        )

    @classmethod
    def parse_with_meta(cls, text: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        candidate = cls.extract_json_text(text)
        attempts = cls._repair_common_json_issues(candidate)
        repair_names = [
            "original",
            "remove_trailing_commas",
            "replace_semicolons_and_clean_commas",
            "strip_comments_and_clean_commas",
        ]
        last_exc: Optional[Exception] = None
        for attempt_index, attempt in enumerate(attempts):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected dict JSON object, got {type(parsed).__name__}")
                meta = {
                    "repair_used": bool(attempt_index > 0),
                    "repair_name": repair_names[attempt_index] if attempt_index < len(repair_names) else f"attempt_{attempt_index}",
                    "num_candidate_attempts": int(len(attempts)),
                    "candidate_preview": cls.preview(candidate),
                    "accepted_preview": cls.preview(attempt),
                }
                return parsed, meta
            except Exception as exc:
                last_exc = exc
        raise ValueError(
            f"Invalid JSON returned by model: {last_exc}; preview={cls.preview(candidate)}"
        )


class LLMCallRuntimeError(RuntimeError):
    def __init__(self, message: str, *, snapshot: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.snapshot = snapshot or {}


class SectorPriorRepository:
    @staticmethod
    def load(data_dir: str) -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, float]]:
        path = os.path.join(data_dir, "sector_theta_medians.json")
        if not os.path.exists(path):
            return list(THETA_KEYS_EVAL), {}, {k: 0.0 for k in THETA_KEYS_EVAL}

        with open(path, "r", encoding="utf-8") as file:
            med = json.load(file)

        if "theta_cols" in med and "sector_medians" in med:
            theta_cols = [str(x) for x in med.get("theta_cols", THETA_KEYS_EVAL)]
            sector_medians_raw = med.get("sector_medians", {})
        else:
            sector_medians_raw = med
            if sector_medians_raw:
                first_values = next(iter(sector_medians_raw.values()))
                theta_cols = list(first_values.keys()) if isinstance(first_values, dict) else list(THETA_KEYS_EVAL)
            else:
                theta_cols = list(THETA_KEYS_EVAL)

        sector_medians: Dict[str, Dict[str, float]] = {}
        global_vals: Dict[str, List[float]] = {k: [] for k in theta_cols}
        for sector, values in sector_medians_raw.items():
            if isinstance(values, list):
                row = {theta_cols[i]: float(values[i]) for i in range(min(len(theta_cols), len(values)))}
            else:
                row = {k: float(values.get(k, np.nan)) for k in theta_cols}
            clean_row: Dict[str, float] = {}
            for k in theta_cols:
                value = float(row.get(k, np.nan))
                if np.isfinite(value):
                    clean_row[k] = value
                    global_vals.setdefault(k, []).append(value)
            sector_medians[str(sector)] = clean_row

        global_medians: Dict[str, float] = {}
        for k in theta_cols:
            vals = np.asarray(global_vals.get(k, []), dtype=float)
            global_medians[k] = float(np.nanmedian(vals)) if vals.size > 0 else 0.0

        return theta_cols, sector_medians, global_medians


class LLMForecastSanitizer:
    @staticmethod
    def _fallback_theta(name: str, priors: Dict[str, float]) -> float:
        if name in priors and np.isfinite(priors[name]):
            return float(priors[name])
        lo, hi, kind = THETA_BOUNDS[name]
        if kind == "signed":
            return 0.0
        return 0.5 * (lo + hi)

    @staticmethod
    def _coerce_float(value: Any, default: float) -> Tuple[float, bool]:
        try:
            fv = float(value)
        except Exception:
            return float(default), True
        if not np.isfinite(fv):
            return float(default), True
        return float(fv), False

    @classmethod
    def _sanitize_triplet_with_meta(
        cls,
        values: Sequence[Any],
        *,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        default: float = 0.0,
        field_name: str = "",
    ) -> Tuple[Tuple[float, float, float], Dict[str, Any]]:
        raw_vals: List[float] = []
        used_default = False
        for value in values:
            fv, fallback_used = cls._coerce_float(value, float(default))
            raw_vals.append(float(fv))
            used_default = used_default or bool(fallback_used)

        ordered_before = bool(raw_vals[0] <= raw_vals[1] <= raw_vals[2])
        arr = sorted(raw_vals)
        clipped = False
        if lo is not None:
            arr2 = [max(float(lo), v) for v in arr]
            clipped = clipped or any(abs(a - b) > 1e-12 for a, b in zip(arr, arr2))
            arr = arr2
        if hi is not None:
            arr2 = [min(float(hi), v) for v in arr]
            clipped = clipped or any(abs(a - b) > 1e-12 for a, b in zip(arr, arr2))
            arr = arr2
        arr = sorted(arr)
        degenerate = bool(abs(arr[0] - arr[1]) <= 1e-12 and abs(arr[1] - arr[2]) <= 1e-12)
        meta = {
            "field": str(field_name),
            "raw_triplet": [float(v) for v in raw_vals],
            "sanitized_triplet": [float(arr[0]), float(arr[1]), float(arr[2])],
            "used_default": bool(used_default),
            "reordered_quantiles": bool(not ordered_before),
            "clipped_to_bounds": bool(clipped),
            "degenerate_triplet": bool(degenerate),
        }
        return (float(arr[0]), float(arr[1]), float(arr[2])), meta

    @classmethod
    def sanitize_with_meta(
        cls,
        parsed: Dict[str, Any],
        *,
        sector: str,
        prior_theta: Dict[str, float],
        logS_baseline: float,
    ) -> Tuple[Dict[str, Dict[str, Tuple[float, float, float]]], Dict[str, Any]]:
        output = {"logS": {}, "theta": {}}
        meta: Dict[str, Any] = {
            "sector": str(sector),
            "logS_baseline": float(logS_baseline),
            "logS_source": "fallback",
            "theta_source": {},
            "missing_theta_keys": [],
            "extra_theta_keys": [],
            "missing_logS_keys": [],
            "degenerate_fields": [],
            "reordered_fields": [],
            "clipped_fields": [],
            "financial_overrides": [],
        }

        logS_data = parsed.get("logS", {}) if isinstance(parsed, dict) else {}
        if isinstance(logS_data, dict):
            missing_logS_keys = [q for q in ["q10", "q50", "q90"] if q not in logS_data]
            meta["missing_logS_keys"] = missing_logS_keys
            logS_triplet = [logS_data.get("q10", logS_baseline), logS_data.get("q50", logS_baseline), logS_data.get("q90", logS_baseline)]
            meta["logS_source"] = "model" if len(missing_logS_keys) == 0 else "mixed"
        else:
            logS_triplet = [logS_baseline, logS_baseline, logS_baseline]
            meta["missing_logS_keys"] = ["q10", "q50", "q90"]
            meta["logS_source"] = "fallback"
        logS_q, logS_meta = cls._sanitize_triplet_with_meta(logS_triplet, default=float(logS_baseline), field_name="logS")
        output["logS"] = {"q": logS_q}
        meta["logS_triplet_meta"] = logS_meta
        if logS_meta["degenerate_triplet"]:
            meta["degenerate_fields"].append("logS")
        if logS_meta["reordered_quantiles"]:
            meta["reordered_fields"].append("logS")

        theta_raw = parsed.get("theta", {}) if isinstance(parsed, dict) else {}
        theta_raw = theta_raw if isinstance(theta_raw, dict) else {}
        extra_theta_keys = sorted([str(k) for k in theta_raw.keys() if k not in THETA_KEYS_EVAL])
        meta["extra_theta_keys"] = extra_theta_keys
        is_financial = _is_financial_sector(sector)
        for name in THETA_KEYS_EVAL:
            if is_financial and name in ["m_gross", "DSO", "DIO", "DPO"]:
                output["theta"][name] = (0.0, 0.0, 0.0)
                meta["theta_source"][name] = "financial_override"
                meta["financial_overrides"].append(name)
                meta["degenerate_fields"].append(name)
                continue

            raw = theta_raw.get(name, {})
            fallback = cls._fallback_theta(name, prior_theta)
            lo, hi, _ = THETA_BOUNDS[name]
            if isinstance(raw, dict):
                provided = [q for q in ["q10", "q50", "q90"] if q in raw]
                if len(provided) == 3:
                    source = "model"
                elif len(provided) > 0:
                    source = "mixed"
                else:
                    source = "fallback"
                triplet = [raw.get("q10", fallback), raw.get("q50", fallback), raw.get("q90", fallback)]
            else:
                provided = []
                source = "fallback"
                triplet = [fallback, fallback, fallback]

            if len(provided) < 3:
                meta["missing_theta_keys"].append(name)
            theta_q, theta_meta = cls._sanitize_triplet_with_meta(
                triplet,
                lo=lo,
                hi=hi,
                default=fallback,
                field_name=name,
            )
            output["theta"][name] = theta_q
            meta["theta_source"][name] = source
            if theta_meta["degenerate_triplet"]:
                meta["degenerate_fields"].append(name)
            if theta_meta["reordered_quantiles"]:
                meta["reordered_fields"].append(name)
            if theta_meta["clipped_to_bounds"]:
                meta["clipped_fields"].append(name)

        source_counts: Dict[str, int] = {}
        for _, src_name in meta["theta_source"].items():
            source_counts[src_name] = int(source_counts.get(src_name, 0) + 1)
        meta["theta_source_counts"] = source_counts
        meta["used_any_theta_fallback"] = bool(any(src in {"fallback", "mixed"} for src in meta["theta_source"].values()))
        meta["all_theta_degenerate"] = bool(all(name in meta["degenerate_fields"] for name in THETA_KEYS_EVAL))
        return output, meta

    @classmethod
    def sanitize(
        cls,
        parsed: Dict[str, Any],
        *,
        sector: str,
        prior_theta: Dict[str, float],
        logS_baseline: float,
    ) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
        output, _ = cls.sanitize_with_meta(
            parsed,
            sector=sector,
            prior_theta=prior_theta,
            logS_baseline=logS_baseline,
        )
        return output



class LLMPromptBuilder:
    def __init__(self, history_window: int = 6, rollout_context_window: int = 4) -> None:
        self.history_window = max(2, int(history_window))
        self.rollout_context_window = max(1, int(rollout_context_window))

    @staticmethod
    def _compact_number(value: Any) -> Optional[float]:
        try:
            fv = float(value)
        except Exception:
            return None
        if not np.isfinite(fv):
            return None
        return float(round(fv, 6))

    def _state_map_from_array(self, state: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(state, dtype=float).reshape(-1)
        out: Dict[str, float] = {}
        for i, name in enumerate(STATE_COLS):
            if i < arr.size:
                val = self._compact_number(arr[i])
                if val is not None:
                    out[name] = val
        return out

    def _state_features_from_array(self, state: np.ndarray, *, logS_anchor: Optional[float] = None) -> Dict[str, float]:
        arr = np.asarray(state, dtype=float).reshape(-1)
        if arr.size < len(STATE_COLS):
            arr2 = np.zeros(len(STATE_COLS), dtype=float)
            arr2[:arr.size] = arr
            arr = arr2

        C, AR, Inv, OCA, K, ONCA, AP, OCL, STD, LTD, ONCL, E = [float(x) for x in arr[:12]]
        TA = C + AR + Inv + OCA + K + ONCA
        TL = AP + OCL + STD + LTD + ONCL
        feats: Dict[str, Optional[float]] = {
            "TA_current": self._compact_number(TA),
            "TL_current": self._compact_number(TL),
            "equity_identity_gap": self._compact_number(TA - TL - E),
            "cash_to_assets": self._compact_number(C / TA) if abs(TA) > 1e-12 else 0.0,
            "debt_to_assets": self._compact_number((STD + LTD) / TA) if abs(TA) > 1e-12 else 0.0,
            "working_capital_to_assets": self._compact_number((AR + Inv + OCA - AP - OCL) / TA) if abs(TA) > 1e-12 else 0.0,
        }
        if logS_anchor is not None and np.isfinite(logS_anchor):
            S_anchor = float(np.exp(np.clip(float(logS_anchor), -20.0, 50.0)))
            feats["sales_anchor"] = self._compact_number(S_anchor)
            feats["cash_to_sales_anchor"] = self._compact_number(C / S_anchor) if abs(S_anchor) > 1e-12 else 0.0
            feats["inventory_to_sales_anchor"] = self._compact_number(Inv / S_anchor) if abs(S_anchor) > 1e-12 else 0.0
            feats["short_debt_to_sales_anchor"] = self._compact_number(STD / S_anchor) if abs(S_anchor) > 1e-12 else 0.0
        return {k: float(v) for k, v in feats.items() if v is not None}

    def _history_rows(self, df: pd.DataFrame, idx: int) -> List[Dict[str, Any]]:
        cols = [
            "date", "period_days", "S", "logS",
            "C", "AR", "Inv", "OCA", "OCA_implied", "K", "ONCA", "ONCA_implied",
            "AP", "OCL", "OCL_implied", "STD", "LTD", "ONCL", "ONCL_implied",
            "TA", "TL",
            "COGS", "OPEX", "Tax", "NI", "Div", "I", "EquityIssues", "Buyback",
            "m_gross", "m_opex", "DSO", "DIO", "DPO",
            "alpha_OCA", "alpha_ONCA", "alpha_OCL", "alpha_ONCL",
            "kappa", "delta", "payout", "neteq_to_sales", "phi", "r_ST", "r_LT", "tau",
        ]
        start = max(0, idx - self.history_window + 1)
        out: List[Dict[str, Any]] = []
        for j in range(start, idx + 1):
            row = df.iloc[j]
            rec: Dict[str, Any] = {}
            for col in cols:
                if col not in row.index:
                    continue
                if col == "date":
                    rec[col] = str(pd.to_datetime(row[col]).date())
                else:
                    val = self._compact_number(row[col])
                    if val is not None:
                        rec[col] = val
            out.append(rec)
        return out

    def _compact_context_sequence(self, items: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not items:
            return []
        out: List[Dict[str, Any]] = []
        start = max(0, len(items) - self.rollout_context_window)
        for item in list(items)[start:]:
            if isinstance(item, dict):
                out.append(item)
        return out

    @staticmethod
    def _theta_semantics() -> Dict[str, str]:
        return {
            "m_gross": "gross-margin control; COGS=(1-m_gross)*S",
            "m_opex": "operating-expense ratio; OPEX=m_opex*S",
            "DSO": "days sales outstanding; AR=(DSO/dt)*S",
            "DIO": "days inventory outstanding; Inv=(DIO/dt)*COGS",
            "DPO": "days payables outstanding; AP=(DPO/dt)*COGS",
            "alpha_OCA": "other current assets as a fraction of sales; OCA=alpha_OCA*S",
            "alpha_ONCA": "other non-current assets as a fraction of sales; ONCA=alpha_ONCA*S",
            "alpha_OCL": "other current liabilities as a fraction of sales; OCL=alpha_OCL*S",
            "alpha_ONCL": "other non-current liabilities as a fraction of sales; ONCL=alpha_ONCL*S",
            "kappa": "capital-expenditure ratio; CapEx=kappa*S",
            "delta": "depreciation rate on lagged K; Dep=delta*K_prev",
            "payout": "dividend payout ratio applied to positive NI; Div=payout*max(NI,0)",
            "neteq_to_sales": "net equity issuance or buyback scaled by sales; NetEq=neteq_to_sales*S",
            "phi": "minimum cash buffer ratio; C_min=phi*S",
            "r_ST": "short-term debt rate; Int contribution from lagged STD",
            "r_LT": "long-term debt rate; Int contribution from lagged LTD",
            "tau": "effective tax rate; Tax=tau*max(EBIT-Int,0)",
        }

    @staticmethod
    def _state_semantics() -> Dict[str, str]:
        return {
            "C": "cash",
            "AR": "accounts receivable",
            "Inv": "inventory",
            "OCA": "other current assets",
            "K": "property plant and equipment / capital stock",
            "ONCA": "other non-current assets",
            "AP": "accounts payable",
            "OCL": "other current liabilities",
            "STD": "short-term debt",
            "LTD": "long-term debt",
            "ONCL": "other non-current liabilities",
            "E_flow": "equity stock carried by the accounting recursion",
        }

    @staticmethod
    def _formula_sheet() -> Dict[str, str]:
        return {
            "sales_driver": "S = exp(logS)",
            "operating_block": "COGS=(1-m_gross)*S; OPEX=m_opex*S",
            "working_capital": "AR=(DSO/dt)*S; Inv=(DIO/dt)*COGS; AP=(DPO/dt)*COGS",
            "residual_balance_sheet_blocks": "OCA=alpha_OCA*S; ONCA=alpha_ONCA*S; OCL=alpha_OCL*S; ONCL=alpha_ONCL*S",
            "capital_stock": "CapEx=kappa*S; Dep=delta*K_prev; K_next=K_prev+CapEx-Dep",
            "interest": "Int = 0 if disable_interest else r_ST*STD_prev + r_LT*LTD_prev",
            "earnings": "EBIT=(S-COGS-OPEX)-Dep; Tax=tau*max(EBIT-Int,0); NI=EBIT-Int-Tax",
            "equity_and_payout": "Div=payout*max(NI,0); NetEq=neteq_to_sales*S; E_next=E_prev+NI-Div+NetEq",
            "cash_flow_logic": "dNWC=dAR+dInv+dOCA-dAP-dOCL; CFO=NI+Dep-dNWC; CFI=-CapEx-dONCA",
            "liquidity_closure": "C_min=phi*S; borrow/repay debt to keep cash near the minimum buffer",
            "identity_goal": "local simulator rolls forward next state and checks TA = TL + E and cash reconciliation",
        }

    def build(
        self,
        *,
        df: pd.DataFrame,
        idx: int,
        ticker: str,
        sector: str,
        prior_theta: Dict[str, float],
        logS_baseline: float,
        current_state: np.ndarray,
        period_days_hint: float,
        disable_interest: bool,
        recent_rollout_context: Optional[Sequence[Dict[str, Any]]] = None,
        current_observed_context: Optional[Dict[str, Any]] = None,
        baseline_next_context: Optional[Dict[str, Any]] = None,
        theta_history_summary: Optional[Dict[str, Any]] = None,
        sales_history_summary: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        _ = baseline_next_context
        history = self._history_rows(df, idx)
        state_map = self._state_map_from_array(current_state)
        bounds = {
            name: {"lo": THETA_BOUNDS[name][0], "hi": THETA_BOUNDS[name][1], "kind": THETA_BOUNDS[name][2]}
            for name in THETA_KEYS_EVAL
        }
        financial_rule = _is_financial_sector(sector)

        system_prompt = (
            "You are a forecasting submodule for an accounting-driven one-step financial statement model. "
            "Infer the next-period logS and theta only from the provided company-specific history, the company sector, the raw state/flow values, and the formula sheet. "
            "Do NOT rely on any external theory constants, sector medians, TFT outputs, or unstated priors. "
            "Return EXACTLY one single-line minified JSON object. "
            "The first character must be '{' and the last character must be '}'. "
            "Use double-quoted keys and numeric values only. "
            "Use commas as separators; semicolons are forbidden. "
            "Do not return markdown, code fences, comments, prose, NaN, null, Infinity, trailing commas, or unquoted keys. "
            "Local code will take your logS and theta forecasts, run the accounting simulator, and enforce accounting identities. "
            "Your forecast should therefore be economically plausible given the company history and the formulas. "
            "Quantiles must satisfy q10 <= q50 <= q90 for every variable. "
            "All numbers must be raw decimals, not percentages. "
            "Every theta key in required_output_schema must appear exactly once. "
            "Do not omit keys. Do not rename keys. Do not wrap the JSON in any extra text."
        )

        user_payload = {
            "task": "Forecast the next observed period only.",
            "ticker": str(ticker),
            "sector": str(sector),
            "current_index": int(idx),
            "next_period_days_hint": float(round(float(period_days_hint), 6)),
            "disable_interest": bool(disable_interest),
            "forecast_objective": (
                "Predict next-period logS and theta so that, when the local simulator applies the accounting formulas, "
                "the resulting next-period statements and flows are realistic for this company."
            ),
            "financial_sector_rule": (
                "Because this company is in a financial/bank sector, set m_gross, DSO, DIO, and DPO to exactly 0.0 for q10/q50/q90."
                if financial_rule else
                "Not a financial-sector override."
            ),
            "task_background": {
                "two_stage_pipeline": "Stage 1 forecasts logS and theta; Stage 2 locally simulates the next state with accounting constraints.",
                "driver_definition": "logS is the log of sales; S=exp(logS).",
                "state_and_flow_note": "Stocks are end-of-period state variables; flows are within-period quantities accrued over the next period.",
                "identity_note": "The local simulator checks balance-sheet identity and cash reconciliation; the LLM should focus on economically plausible controls."
            },
            "formula_sheet": self._formula_sheet(),
            "theta_semantics": self._theta_semantics(),
            "state_variable_semantics": self._state_semantics(),
            "company_history_inference_summary": {
                "sales_history_summary": sales_history_summary or {},
                "theta_history_summary": theta_history_summary or {},
            },
            "current_identity_consistent_state": state_map,
            "current_state_features": self._state_features_from_array(current_state, logS_anchor=logS_baseline),
            "current_observed_context": current_observed_context or {},
            "recent_rollout_context": self._compact_context_sequence(recent_rollout_context),
            "historical_rows": history,
            "theta_bounds": bounds,
            "required_theta_key_order": list(THETA_KEYS_EVAL),
            "required_output_schema": {
                "logS": {"q10": 0.0, "q50": 0.0, "q90": 0.0},
                "theta": {
                    name: {"q10": 0.0, "q50": 0.0, "q90": 0.0}
                    for name in THETA_KEYS_EVAL
                },
            },
            "forecasting_guidance": [
                "Use the company's own observed history and current raw state/flow configuration as the primary evidence.",
                "Use the provided formulas to reason how theta affects AR, inventory, payables, expenses, cash, debt, equity, and the next-period statements.",
                "Prefer continuity with recent company behavior unless the recent raw state/flow signals indicate a regime change.",
                "Let q10 and q90 express uncertainty around q50; avoid degenerate triplets unless the variable is truly near-deterministic.",
                "Do not mechanically repeat the latest value for every field; update values when the current state/flow configuration implies pressure on liquidity, working capital, margins, or financing.",
                "Do not use any external sector median, theory constant, or TFT forecast. Infer from the provided company data only.",
            ],
            "hard_requirements": [
                "Return exactly one JSON object following required_output_schema.",
                "Keep q10 <= q50 <= q90 for every variable.",
                "Respect all theta_bounds.",
                "Every required theta key must appear exactly once.",
                "Return compact valid JSON only: no semicolons, no comments, no markdown, no explanatory text.",
            ],
            "pre_submission_checklist": [
                "The response is one valid JSON object.",
                "Top-level keys are exactly logS and theta, plus any optional ignored extras.",
                "theta contains every key in required_theta_key_order.",
                "All values are numeric decimals.",
                "No key is missing and no quantile order is violated.",
            ],
            "output_format_rules": {
                "single_line_json": True,
                "double_quoted_keys": True,
                "comma_separators_only": True,
                "no_trailing_commas": True,
            },
        }
        return system_prompt, json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))


def _normalize_ticker_selection(tickers: Optional[Sequence[str] | str]) -> Optional[List[str]]:
    if tickers is None:
        return None
    if isinstance(tickers, str):
        normalized = [tok.strip() for tok in tickers.split(",") if tok.strip()]
    else:
        normalized = [str(tok).strip() for tok in tickers if str(tok).strip()]
    return normalized or None


class LLMBacktestRunner:
    """Rolling one-step backtester driven by LLM forecasts.

    Architecture:
    1. Local code loads history and sector priors.
    2. The LLM predicts only ``logS`` and ``theta`` quantiles.
    3. Local code sanitizes those predictions and calls ``simulate_step``.
    4. Outputs are saved in a TFT-compatible CSV schema for downstream plotting.
    """

    def __init__(self, config: Optional[LLMBacktestConfig] = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either 'config' or keyword arguments, not both.")
        self.config = config if config is not None else LLMBacktestConfig(**kwargs)
        if self.config.mode != "backtest":
            raise ValueError("LLMBacktestRunner currently supports mode='backtest' only.")
        self.meta = JsonRepository.load(os.path.join(self.config.data_dir, "meta.json"))
        self.prompt_builder = LLMPromptBuilder(self.config.prompt_history_window, self.config.rollout_context_window)
        self.theta_cols = list(THETA_KEYS_EVAL)
        self.sector_theta_medians = {}
        self.global_theta_medians = {}
        self.client = OpenAICompatibleChatClient(self.config.api) if self.config.api.enabled else None
        self._call_counter = 0

    @staticmethod
    def load_test_tickers(data_dir: str) -> List[str]:
        meta = JsonRepository.load(os.path.join(data_dir, "meta.json"))
        return [str(t) for t in meta.get("test_tickers", [])]

    def selected_tickers(self) -> List[str]:
        explicit = _normalize_ticker_selection(self.config.tickers)
        return explicit if explicit is not None else [str(t) for t in self.meta.get("test_tickers", [])]

    def _history_theta_defaults(self, df: pd.DataFrame, idx: int, sector: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        upto = df.iloc[: idx + 1] if idx >= 0 else df.iloc[:0]
        for name in THETA_KEYS_EVAL:
            if _is_financial_sector(sector) and name in ["m_gross", "DSO", "DIO", "DPO"]:
                out[name] = 0.0
                continue
            vals: List[float] = []
            if name in upto.columns:
                for v in upto[name].tolist():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if np.isfinite(fv):
                        vals.append(fv)
            if vals:
                out[name] = float(vals[-1])
            else:
                lo, hi, kind = THETA_BOUNDS[name]
                out[name] = 0.0 if kind == "signed" else float(0.5 * (lo + hi))
        return out

    def _history_theta_summary(self, df: pd.DataFrame, idx: int, sector: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        upto = df.iloc[: idx + 1] if idx >= 0 else df.iloc[:0]
        for name in THETA_KEYS_EVAL:
            if _is_financial_sector(sector) and name in ["m_gross", "DSO", "DIO", "DPO"]:
                out[name] = {"count": 0, "financial_override": True, "last": 0.0, "median": 0.0}
                continue
            vals: List[float] = []
            if name in upto.columns:
                for v in upto[name].tolist():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if np.isfinite(fv):
                        vals.append(fv)
            if vals:
                arr = np.asarray(vals, dtype=float)
                rec: Dict[str, Any] = {
                    "count": int(arr.size),
                    "last": float(round(float(arr[-1]), 6)),
                    "median": float(round(float(np.nanmedian(arr)), 6)),
                    "min": float(round(float(np.nanmin(arr)), 6)),
                    "max": float(round(float(np.nanmax(arr)), 6)),
                }
                if arr.size >= 2:
                    rec["recent_delta"] = float(round(float(arr[-1] - arr[-2]), 6))
                out[name] = rec
            else:
                lo, hi, kind = THETA_BOUNDS[name]
                out[name] = {
                    "count": 0,
                    "default_if_missing": 0.0 if kind == "signed" else float(round(0.5 * (lo + hi), 6)),
                }
        return out

    def _fallback_logS_baseline(self, df: pd.DataFrame, idx: int) -> float:
        hist_logS = np.asarray([_get_logS(df.iloc[i]) for i in range(idx + 1)], dtype=float)
        hist_logS = hist_logS[np.isfinite(hist_logS)]
        if hist_logS.size == 0:
            return 0.0
        if hist_logS.size >= int(self.config.min_ar1_points):
            c, phi = fit_ar1(hist_logS)
        else:
            c, phi = 0.0, 1.0
        return float(ar1_next(float(hist_logS[-1]), c, phi))

    def _sales_history_summary(self, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        hist_logS = np.asarray([_get_logS(df.iloc[i]) for i in range(idx + 1)], dtype=float)
        hist_logS = hist_logS[np.isfinite(hist_logS)]
        rec: Dict[str, Any] = {
            "count": int(hist_logS.size),
            "history_default_next_logS": float(round(self._fallback_logS_baseline(df, idx), 6)),
        }
        if hist_logS.size > 0:
            rec["last_logS"] = float(round(float(hist_logS[-1]), 6))
            rec["median_logS"] = float(round(float(np.nanmedian(hist_logS)), 6))
            rec["min_logS"] = float(round(float(np.nanmin(hist_logS)), 6))
            rec["max_logS"] = float(round(float(np.nanmax(hist_logS)), 6))
        if hist_logS.size >= 2:
            deltas = np.diff(hist_logS)
            rec["last_delta_logS"] = float(round(float(deltas[-1]), 6))
            rec["median_delta_logS"] = float(round(float(np.nanmedian(deltas)), 6))
        return rec

    def _log(self, message: str) -> None:
        if bool(self.config.verbose):
            print(message, flush=True)

    def _fallback_sanitized_output(self, *, sector: str, idx: int, df: pd.DataFrame) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
        history_theta_defaults = self._history_theta_defaults(df, idx, sector)
        logS_baseline = self._fallback_logS_baseline(df, idx)
        return LLMForecastSanitizer.sanitize({}, sector=sector, prior_theta=history_theta_defaults, logS_baseline=logS_baseline)

    def _state_map_from_array(self, state: np.ndarray) -> Dict[str, float]:
        return self.prompt_builder._state_map_from_array(np.asarray(state, dtype=float))

    def _state_features_from_array(self, state: np.ndarray, *, logS_anchor: Optional[float] = None) -> Dict[str, float]:
        return self.prompt_builder._state_features_from_array(np.asarray(state, dtype=float), logS_anchor=logS_anchor)

    @staticmethod
    def _flow_truth_specs() -> Dict[str, Tuple[List[str], Optional[Any]]]:
        def abs_if_finite(x: float) -> float:
            return float(abs(x)) if np.isfinite(x) else float("nan")
        return {
            "COGS": (["COGS"], None),
            "OPEX": (["OPEX"], None),
            "Tax": (["Tax"], None),
            "NI": (["NI"], None),
            "Div": (["Div"], abs_if_finite),
            "Int": (["I"], abs_if_finite),
            "TA": (["TA"], None),
            "TL": (["TL"], None),
        }

    def _theta_truth_from_row(self, row: pd.Series, sector: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in THETA_KEYS_EVAL:
            value = _get_first_finite(row, [k])
            if _is_financial_sector(sector) and (k in ["m_gross", "DSO", "DIO", "DPO"]):
                value = 0.0
            compact = self.prompt_builder._compact_number(value)
            if compact is not None:
                out[k] = float(compact)
        return out

    def _flow_truth_from_row(self, row: pd.Series) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, (cols, fn) in self._flow_truth_specs().items():
            value = _get_first_finite(row, cols)
            if fn is not None:
                value = fn(value)
            compact = self.prompt_builder._compact_number(value)
            if compact is not None:
                out[name] = float(compact)
        eq = _get_first_finite(row, ["EquityIssues"])
        bb = _get_first_finite(row, ["Buyback"])
        if np.isfinite(eq) or np.isfinite(bb):
            eqv = float(eq) if np.isfinite(eq) else 0.0
            bbv = float(bb) if np.isfinite(bb) else 0.0
            neteq = self.prompt_builder._compact_number(eqv - bbv)
            if neteq is not None:
                out["NetEq"] = float(neteq)
        return out

    def _observed_context_from_row(self, *, df: pd.DataFrame, idx: int, sector: str, source: str = "observed_current") -> Dict[str, Any]:
        row = df.iloc[idx]
        state = _truth_state_from_row(row).astype(float)
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        if np.all(np.isfinite(state[:11])):
            state[11] = float(np.sum(state[:6]) - np.sum(state[6:11]))
        logS_val = _get_logS(row)
        S_val = _get_first_finite(row, ["S"])
        if not np.isfinite(S_val) and np.isfinite(logS_val):
            S_val = float(np.exp(np.clip(float(logS_val), -20.0, 50.0)))
        period_days = _get_first_finite(row, ["period_days"])
        if not np.isfinite(period_days):
            period_days = 365.0
        ctx: Dict[str, Any] = {
            "source": str(source),
            "idx": int(idx),
            "date": str(pd.to_datetime(row["date"]).date()),
            "period_days": float(round(float(period_days), 6)),
            "state": self._state_map_from_array(state),
            "state_features": self._state_features_from_array(state, logS_anchor=logS_val if np.isfinite(logS_val) else None),
            "theta_observed": self._theta_truth_from_row(row, sector),
            "flow_observed": self._flow_truth_from_row(row),
        }
        compact_logS = self.prompt_builder._compact_number(logS_val)
        if compact_logS is not None:
            ctx["logS_observed"] = float(compact_logS)
        compact_S = self.prompt_builder._compact_number(S_val)
        if compact_S is not None:
            ctx["S_observed"] = float(compact_S)
        return ctx

    def _diag_flow_map(self, diag: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in FLOW_KEYS_EVAL:
            if name in diag:
                compact = self.prompt_builder._compact_number(diag.get(name))
                if compact is not None:
                    out[name] = float(compact)
        return out

    def _rollout_context_entry(
        self,
        *,
        source: str,
        date: Any,
        idx: int,
        period_days: float,
        logS: float,
        theta: Dict[str, float],
        state: np.ndarray,
        diag: Dict[str, Any],
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "source": str(source),
            "idx": int(idx),
            "date": str(pd.to_datetime(date).date()) if date is not None else "",
            "period_days": float(round(float(period_days), 6)),
            "logS_q50_used": float(round(float(logS), 6)),
            "theta_q50_used": {k: float(round(float(v), 6)) for k, v in theta.items() if np.isfinite(v)},
            "next_state": self._state_map_from_array(state),
            "next_state_features": self._state_features_from_array(state, logS_anchor=logS),
            "next_flow": self._diag_flow_map(diag),
            "checks": {
                k: float(round(float(diag[k]), 6))
                for k in ["bs_resid", "cash_resid", "Borrow", "Repay_st", "Repay_lt", "C_min"]
                if k in diag and np.isfinite(diag[k])
            },
        }
        return entry

    def _baseline_next_context(
        self,
        *,
        df: pd.DataFrame,
        idx: int,
        sector: str,
        current_state: np.ndarray,
        period_days_hint: float,
        disable_interest: bool,
    ) -> Dict[str, Any]:
        prior_theta = self._prior_theta_for_sector(sector)
        logS_baseline = self._fallback_logS_baseline(df, idx)
        try:
            st_next, diag = simulate_step(
                np.asarray(current_state, dtype=float),
                float(logS_baseline),
                prior_theta,
                float(period_days_hint),
                is_financial=_is_financial_sector(sector),
                disable_interest=disable_interest,
            )
            return self._rollout_context_entry(
                source="baseline_from_prior",
                date=df.iloc[min(idx + 1, len(df) - 1)]["date"] if len(df) > 0 else "",
                idx=int(idx + 1),
                period_days=float(period_days_hint),
                logS=float(logS_baseline),
                theta=prior_theta,
                state=st_next,
                diag=diag,
            )
        except Exception as exc:
            return {
                "source": "baseline_from_prior",
                "warning": f"baseline simulation failed: {exc}",
                "logS_q50_used": float(round(float(logS_baseline), 6)),
                "theta_q50_used": {k: float(round(float(v), 6)) for k, v in prior_theta.items() if np.isfinite(v)},
            }


    def _build_storage_snapshot(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        call_info: Optional[Dict[str, Any]] = None,
        parsed: Optional[Dict[str, Any]] = None,
        sanitized: Optional[Dict[str, Any]] = None,
        parser_error: str = "",
        elapsed_s: Optional[float] = None,
        attempt_index: int = 0,
        logical_call_no: Optional[int] = None,
        api_call_no: Optional[int] = None,
        parser_meta: Optional[Dict[str, Any]] = None,
        sanitization_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        snap: Dict[str, Any] = {
            "attempt_index": int(attempt_index),
            "logical_call_no": int(logical_call_no) if logical_call_no is not None else None,
            "api_call_no": int(api_call_no) if api_call_no is not None else None,
            "elapsed_s": float(elapsed_s) if elapsed_s is not None else None,
            "parser_error": str(parser_error or ""),
        }
        if bool(self.config.save_raw_prompts):
            snap["system_prompt"] = system_prompt
            snap["user_prompt"] = user_prompt
        else:
            snap["system_prompt_preview"] = LLMResponseParser.preview(system_prompt, limit=int(self.config.debug_preview_chars))
            snap["user_prompt_preview"] = LLMResponseParser.preview(user_prompt, limit=int(self.config.debug_preview_chars))

        call_info = call_info or {}
        if call_info:
            snap["request_payload"] = call_info.get("request_payload")
            snap["raw_text"] = call_info.get("text", "")
            snap["raw_text_preview"] = LLMResponseParser.preview(call_info.get("text", ""), limit=int(self.config.debug_preview_chars))
            if bool(self.config.save_full_raw_response):
                snap["raw_response"] = call_info.get("raw_response")
            else:
                snap["raw_response_preview"] = LLMResponseParser.preview(call_info.get("raw_response", ""), limit=int(self.config.debug_preview_chars))
        if parsed is not None:
            snap["parsed"] = parsed
        if sanitized is not None:
            snap["sanitized"] = sanitized
        if parser_meta is not None:
            snap["parser_meta"] = parser_meta
        if bool(self.config.include_sanitizer_diagnostics) and sanitization_meta is not None:
            snap["sanitization_meta"] = sanitization_meta
        return snap

    @staticmethod
    def _approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
        return bool(np.isfinite(a) and np.isfinite(b) and abs(float(a) - float(b)) <= tol)

    def _diagnose_sanitized_output(
        self,
        *,
        sector: str,
        idx: int,
        df: pd.DataFrame,
        sanitized: Dict[str, Any],
        sanitization_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        history_theta_defaults = self._history_theta_defaults(df, idx, sector)
        logS_history_default = self._fallback_logS_baseline(df, idx)
        logS_q10, logS_q50, logS_q90 = sanitized["logS"]["q"]
        logS_is_degenerate = self._approx_equal(logS_q10, logS_q50) and self._approx_equal(logS_q50, logS_q90)
        logS_equals_default = self._approx_equal(logS_q50, logS_history_default)
        theta_default_like_keys: List[str] = []
        theta_degenerate_keys: List[str] = []
        for k in THETA_KEYS_EVAL:
            q10, q50, q90 = sanitized["theta"][k]
            if self._approx_equal(q10, q50) and self._approx_equal(q50, q90):
                theta_degenerate_keys.append(k)
            if self._approx_equal(q50, float(history_theta_defaults.get(k, 0.0))):
                theta_default_like_keys.append(k)
        fully_history_default_like = bool(
            logS_equals_default
            and logS_is_degenerate
            and len(theta_default_like_keys) == len(THETA_KEYS_EVAL)
            and len(theta_degenerate_keys) == len(THETA_KEYS_EVAL)
        )
        diag = {
            "logS_q50_equals_history_default": bool(logS_equals_default),
            "logS_triplet_degenerate": bool(logS_is_degenerate),
            "num_theta_q50_equal_history_defaults": int(len(theta_default_like_keys)),
            "num_theta_triplets_degenerate": int(len(theta_degenerate_keys)),
            "theta_q50_equal_history_default_keys": theta_default_like_keys,
            "theta_triplet_degenerate_keys": theta_degenerate_keys,
            "fully_history_default_like": bool(fully_history_default_like),
            "fully_baseline_like": bool(fully_history_default_like),
        }
        if sanitization_meta is not None:
            diag["sanitization_meta_summary"] = {
                "logS_source": sanitization_meta.get("logS_source"),
                "theta_source_counts": sanitization_meta.get("theta_source_counts"),
                "missing_theta_keys": sanitization_meta.get("missing_theta_keys"),
                "extra_theta_keys": sanitization_meta.get("extra_theta_keys"),
                "degenerate_fields": sanitization_meta.get("degenerate_fields"),
                "reordered_fields": sanitization_meta.get("reordered_fields"),
                "clipped_fields": sanitization_meta.get("clipped_fields"),
            }
        return diag

    def _summarize_call_for_record(
        self,
        *,
        sector: str,
        idx: int,
        df: pd.DataFrame,
        call: Dict[str, Any],
    ) -> Dict[str, Any]:
        sanitization_meta = call.get("sanitization_meta") or {}
        parser_meta = call.get("parser_meta") or {}
        sanitized = call.get("sanitized") or {}
        return {
            "logical_call_no": call.get("logical_call_no"),
            "api_call_no": call.get("api_call_no"),
            "num_attempts": int(len(call.get("attempts", []) or [])),
            "parser_repair_used": bool(parser_meta.get("repair_used", False)),
            "parser_repair_name": parser_meta.get("repair_name", "original"),
            "logS_source": sanitization_meta.get("logS_source", "unknown"),
            "theta_source_counts": sanitization_meta.get("theta_source_counts", {}),
            "missing_theta_keys": sanitization_meta.get("missing_theta_keys", []),
            "extra_theta_keys": sanitization_meta.get("extra_theta_keys", []),
            "used_any_theta_fallback": bool(sanitization_meta.get("used_any_theta_fallback", False)),
            "diagnostics": self._diagnose_sanitized_output(
                sector=sector,
                idx=idx,
                df=df,
                sanitized=sanitized,
                sanitization_meta=sanitization_meta,
            ) if sanitized else {},
        }

    def _build_retry_prompts(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        parser_error: str,
        raw_text: str,
    ) -> Tuple[str, str]:
        retry_system = (
            system_prompt
            + " Your previous reply was invalid JSON. Reissue the full answer as strict valid minified JSON only."
        )
        repair_payload = {
            "repair_request": {
                "reason": str(parser_error),
                "previous_output_preview": LLMResponseParser.preview(raw_text, limit=int(self.config.debug_preview_chars)),
                "instructions": [
                    "Return the full object again, not a patch.",
                    "Use commas, never semicolons.",
                    "Do not include markdown fences or explanations.",
                    "Make sure the output is a single valid JSON object.",
                ],
            }
        }
        retry_user = user_prompt + "\n" + json.dumps(repair_payload, ensure_ascii=False, separators=(",", ":"))
        return retry_system, retry_user

    def _predict_step1_quantiles(
        self,
        *,
        df: pd.DataFrame,
        idx: int,
        ticker: str,
        sector: str,
        current_state: np.ndarray,
        period_days_hint: float,
        disable_interest: bool,
        recent_rollout_context: Optional[Sequence[Dict[str, Any]]] = None,
        current_observed_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError("LLM API is not enabled. Set config.api.enabled=True and provide API credentials.")

        history_theta_defaults = self._history_theta_defaults(df, idx, sector)
        logS_baseline = self._fallback_logS_baseline(df, idx)
        theta_history_summary = self._history_theta_summary(df, idx, sector)
        sales_history_summary = self._sales_history_summary(df, idx)
        base_system_prompt, base_user_prompt = self.prompt_builder.build(
            df=df,
            idx=idx,
            ticker=ticker,
            sector=sector,
            prior_theta=history_theta_defaults,
            logS_baseline=logS_baseline,
            current_state=current_state,
            period_days_hint=period_days_hint,
            disable_interest=disable_interest,
            recent_rollout_context=recent_rollout_context,
            current_observed_context=current_observed_context,
            baseline_next_context=None,
            theta_history_summary=theta_history_summary,
            sales_history_summary=sales_history_summary,
        )

        logical_call_no = int(self._call_counter) + 1
        max_attempts = max(1, int(self.config.retry_on_invalid_json) + 1)
        attempt_snapshots: List[Dict[str, Any]] = []
        parser_error = ""

        for attempt_index in range(max_attempts):
            if attempt_index == 0:
                system_prompt = base_system_prompt
                user_prompt = base_user_prompt
            else:
                system_prompt, user_prompt = self._build_retry_prompts(
                    system_prompt=base_system_prompt,
                    user_prompt=base_user_prompt,
                    parser_error=parser_error,
                    raw_text=attempt_snapshots[-1].get("raw_text", ""),
                )

            self._call_counter += 1
            api_call_no = int(self._call_counter)
            if int(self.config.progress_every_calls) <= 1 or (api_call_no % int(self.config.progress_every_calls) == 0):
                self._log(
                    f"[LLM] call#{api_call_no} ticker={ticker} idx={idx} date={str(pd.to_datetime(df.iloc[idx]['date']).date())} period_days={float(period_days_hint):.1f} attempt={attempt_index + 1}/{max_attempts}"
                )
            t0 = time.time()
            call_info = self.client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
            elapsed = time.time() - t0
            self._log(f"[LLM] call#{api_call_no} completed in {elapsed:.2f}s")

            try:
                parsed, parser_meta = LLMResponseParser.parse_with_meta(call_info["text"])
                sanitized, sanitization_meta = LLMForecastSanitizer.sanitize_with_meta(
                    parsed,
                    sector=sector,
                    prior_theta=history_theta_defaults,
                    logS_baseline=logS_baseline,
                )
                attempt_snapshots.append(
                    self._build_storage_snapshot(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        call_info=call_info,
                        parsed=parsed,
                        sanitized=sanitized,
                        elapsed_s=elapsed,
                        attempt_index=attempt_index,
                        logical_call_no=logical_call_no,
                        api_call_no=api_call_no,
                        parser_meta=parser_meta,
                        sanitization_meta=sanitization_meta,
                    )
                )
                return {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_text": call_info["text"],
                    "request_payload": call_info["request_payload"],
                    "raw_response": call_info["raw_response"],
                    "parsed": parsed,
                    "sanitized": sanitized,
                    "parser_meta": parser_meta,
                    "sanitization_meta": sanitization_meta,
                    "attempts": attempt_snapshots,
                    "logical_call_no": logical_call_no,
                    "api_call_no": api_call_no,
                }
            except Exception as exc:
                parser_error = f"JSON parse failed: {exc}"
                attempt_snapshots.append(
                    self._build_storage_snapshot(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        call_info=call_info,
                        parsed=None,
                        sanitized=None,
                        parser_error=parser_error,
                        elapsed_s=elapsed,
                        attempt_index=attempt_index,
                        logical_call_no=logical_call_no,
                        api_call_no=api_call_no,
                    )
                )

        raise LLMCallRuntimeError(
            parser_error or "LLM call failed for an unknown reason.",
            snapshot={
                "logical_call_no": logical_call_no,
                "attempts": attempt_snapshots,
                "history_theta_defaults": history_theta_defaults,
                "logS_history_default": float(logS_baseline),
                "theta_history_summary": theta_history_summary,
                "sales_history_summary": sales_history_summary,
                "current_observed_context": current_observed_context,
                "recent_rollout_context": list(recent_rollout_context or []),
            },
        )

    def _warmup_record(
        self,
        *,
        df: pd.DataFrame,
        i: int,
        idx0: int,
        ticker: str,
        sector: str,
        disable_interest: bool,
        raw_records: List[Dict[str, Any]],
        rollout_context_history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        r = df.iloc[i]
        logS_true_i = _get_logS(r)
        logS_true_val_i, mask_logS_i = _nan_to_zero_with_mask(logS_true_i)
        s_true_i = _get_first_finite(r, ["S"])
        s_true_val_i, mask_s_i = _nan_to_zero_with_mask(s_true_i)

        st_true_i = _truth_state_from_row(r)
        st_true_vals_i: List[float] = []
        st_true_masks_i: List[int] = []
        for value in st_true_i.tolist():
            vv, mm = _nan_to_zero_with_mask(float(value) if np.isfinite(value) else float("nan"))
            st_true_vals_i.append(vv)
            st_true_masks_i.append(mm)

        period_days_i = _get_first_finite(r, ["period_days"])
        if not np.isfinite(period_days_i):
            period_days_i = 365.0

        theta_q10 = {k: float("nan") for k in THETA_KEYS_EVAL}
        theta_q50 = {k: float("nan") for k in THETA_KEYS_EVAL}
        theta_q90 = {k: float("nan") for k in THETA_KEYS_EVAL}
        logS_q10 = logS_q50 = logS_q90 = float("nan")
        try:
            current_state = _truth_state_from_row(r).astype(float)
            current_state = np.nan_to_num(current_state, nan=0.0, posinf=0.0, neginf=0.0)
            if np.all(np.isfinite(current_state[:11])):
                current_state[11] = float(np.sum(current_state[:6]) - np.sum(current_state[6:11]))
            call = self._predict_step1_quantiles(
                df=df,
                idx=i,
                ticker=ticker,
                sector=sector,
                current_state=current_state,
                period_days_hint=float(period_days_i),
                disable_interest=disable_interest,
                recent_rollout_context=list(rollout_context_history or []),
                current_observed_context=self._observed_context_from_row(
                    df=df,
                    idx=i,
                    sector=sector,
                    source="warmup_current_observed",
                ),
            )
            sanitized = call["sanitized"]
            logS_q10, logS_q50, logS_q90 = sanitized["logS"]["q"]
            for k in THETA_KEYS_EVAL:
                q10, q50, q90 = sanitized["theta"][k]
                theta_q10[k] = float(q10)
                theta_q50[k] = float(q50)
                theta_q90[k] = float(q90)
            try:
                raw_records.append(
                    {
                        "phase": "warmup",
                        "ticker": ticker,
                        "sector": sector,
                        "idx": int(i),
                        "date": str(pd.to_datetime(r["date"]).date()),
                        "summary": self._summarize_call_for_record(
                            sector=sector,
                            idx=i,
                            df=df,
                            call=call,
                        ),
                        "call": self._mask_raw_call_for_storage(call),
                    }
                )
            except Exception as storage_exc:
                raw_records.append(
                    {
                        "phase": "warmup",
                        "ticker": ticker,
                        "sector": sector,
                        "idx": int(i),
                        "date": str(pd.to_datetime(r["date"]).date()),
                        "warning": f"raw call storage failed but warmup prediction was kept: {storage_exc}",
                    }
                )
                self._log(f"[WARN] {ticker}: warmup idx={i} raw call storage failed ({storage_exc})")
        except Exception as exc:
            rec_err = {
                "phase": "warmup",
                "ticker": ticker,
                "sector": sector,
                "idx": int(i),
                "date": str(pd.to_datetime(r["date"]).date()),
                "error": str(exc),
            }
            if getattr(exc, "snapshot", None):
                rec_err["call"] = getattr(exc, "snapshot")
            raw_records.append(rec_err)

        theta_true = {}
        theta_mask = {}
        for k in THETA_KEYS_EVAL:
            v = _get_first_finite(r, [k])
            vv, mm = _nan_to_zero_with_mask(v)
            if _is_financial_sector(sector) and (k in ["m_gross", "DSO", "DIO", "DPO"]):
                vv, mm = 0.0, 0
            theta_true[k] = vv
            theta_mask[k] = mm

        def abs_if_finite(x: float) -> float:
            return float(abs(x)) if np.isfinite(x) else float("nan")

        flow_specs = {
            "COGS": (["COGS"], None),
            "OPEX": (["OPEX"], None),
            "Tax": (["Tax"], None),
            "NI": (["NI"], None),
            "Div": (["Div"], abs_if_finite),
            "Int": (["I"], abs_if_finite),
            "TA": (["TA"], None),
            "TL": (["TL"], None),
        }
        flow_true = {}
        flow_mask = {}
        for name, (cols, fn) in flow_specs.items():
            value = _get_first_finite(r, cols)
            if fn is not None:
                value = fn(value)
            vv, mm = _nan_to_zero_with_mask(value)
            flow_true[name] = vv
            flow_mask[name] = mm

        eq = _get_first_finite(r, ["EquityIssues"])
        bb = _get_first_finite(r, ["Buyback"])
        mm = int(np.isfinite(eq) or np.isfinite(bb))
        eqv = float(eq) if np.isfinite(eq) else 0.0
        bbv = float(bb) if np.isfinite(bb) else 0.0
        neteq_true = float(eqv - bbv) if mm else 0.0
        neteq_mask = int(mm)

        rec_warm: Dict[str, Any] = dict(
            ticker=str(ticker),
            sector=str(sector),
            date=str(pd.to_datetime(r["date"]).date()),
            idx=int(i),
            step=int(i - idx0),
            period_days=float(period_days_i),
            logS_pred=float("nan"),
            logS_pred_q10=float(logS_q10),
            logS_pred_q50=float(logS_q50),
            logS_pred_q90=float(logS_q90),
            logS_true=float(logS_true_val_i),
            mask_logS=int(mask_logS_i),
            S_true=float(s_true_val_i),
            mask_S=int(mask_s_i),
            **{f"theta_{k}": float(theta_q50.get(k, np.nan)) for k in THETA_KEYS_EVAL},
            **{f"theta_{k}_q10": float(theta_q10.get(k, np.nan)) for k in THETA_KEYS_EVAL},
            **{f"theta_{k}_q50": float(theta_q50.get(k, np.nan)) for k in THETA_KEYS_EVAL},
            **{f"theta_{k}_q90": float(theta_q90.get(k, np.nan)) for k in THETA_KEYS_EVAL},
        )
        rec_warm.update({f"theta_true_{k}": float(v) for k, v in theta_true.items()})
        rec_warm.update({f"mask_theta_{k}": int(m) for k, m in theta_mask.items()})
        for name in flow_specs.keys():
            rec_warm[f"{name}_true"] = float(flow_true[name])
            rec_warm[f"mask_{name}"] = int(flow_mask[name])
        rec_warm["NetEq_true"] = float(neteq_true)
        rec_warm["mask_NetEq"] = int(neteq_mask)
        rec_warm.update({f"state_true_{STATE_COLS[j]}": float(st_true_vals_i[j]) for j in range(len(STATE_COLS))})
        rec_warm.update({f"mask_state_{STATE_COLS[j]}": int(st_true_masks_i[j]) for j in range(len(STATE_COLS))})
        return rec_warm

    def _mask_raw_call_for_storage(self, call: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(call)
        if not bool(self.config.save_raw_prompts):
            payload.pop("system_prompt", None)
            payload.pop("user_prompt", None)
        if not bool(self.config.save_full_raw_response):
            raw_response = payload.pop("raw_response", None)
            if raw_response is not None:
                payload["raw_response_preview"] = LLMResponseParser.preview(raw_response, limit=int(self.config.debug_preview_chars))
        return payload

    def run_backtest_one_ticker(
        self,
        df: pd.DataFrame,
        ticker: str,
        sector: str,
        warmup: int = 3,
        disable_interest: bool = False,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        df = df.sort_values("date").reset_index(drop=True).copy()
        n = len(df)
        if n <= warmup:
            raise ValueError(f"{ticker}: need > warmup rows; got n={n}, warmup={warmup}")

        idx0 = warmup - 1
        row0 = df.iloc[idx0]
        st = _truth_state_from_row(row0)
        st = np.nan_to_num(st, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        ta0 = float(st[0] + st[1] + st[2] + st[3] + st[4] + st[5])
        tl0 = float(st[6] + st[7] + st[8] + st[9] + st[10])
        st[11] = float(ta0 - tl0)

        logS_t = _get_logS(row0)
        if not np.isfinite(logS_t):
            hist_logS = np.asarray([_get_logS(df.iloc[i]) for i in range(idx0 + 1)], dtype=float)
            hist_logS = hist_logS[np.isfinite(hist_logS)]
            logS_t = float(hist_logS[-1]) if hist_logS.size > 0 else 0.0

        rows: List[Dict[str, Any]] = []
        raw_records: List[Dict[str, Any]] = []
        rollout_context_history: List[Dict[str, Any]] = []

        self._log(f"[TICKER] start {ticker} sector={sector} rows={n} warmup={warmup}")
        for i in range(0, idx0 + 1):
            rows.append(
                self._warmup_record(
                    df=df,
                    i=i,
                    idx0=idx0,
                    ticker=ticker,
                    sector=sector,
                    disable_interest=disable_interest,
                    raw_records=raw_records,
                    rollout_context_history=rollout_context_history,
                )
            )
            rollout_context_history.append(
                self._observed_context_from_row(
                    df=df,
                    idx=i,
                    sector=sector,
                    source="warmup_observed_history",
                )
            )

        for idx in range(idx0, n - 1):
            row_cur = df.iloc[idx]
            row_nxt = df.iloc[idx + 1]

            logS_cur_truth = _get_logS(row_cur)
            if np.isfinite(logS_cur_truth):
                logS_t = float(logS_cur_truth)
            if _has_full_truth_state(row_cur):
                st = _truth_state_from_row(row_cur).astype(float)

            period_days_panel = _get_first_finite(row_nxt, ["period_days"])
            if not np.isfinite(period_days_panel):
                period_days_panel = _get_first_finite(row_cur, ["period_days"])
            if not np.isfinite(period_days_panel):
                period_days_panel = 365.0
            try:
                dcur = pd.to_datetime(row_cur["date"])
                dnxt = pd.to_datetime(row_nxt["date"])
                period_days_date = float((dnxt - dcur).days)
            except Exception:
                period_days_date = float("nan")
            if np.isfinite(period_days_date) and period_days_date > 0:
                period_days = float(period_days_date)
            else:
                period_days = float(period_days_panel)
            period_days = float(np.clip(period_days, DT_MIN, DT_MAX))

            try:
                call = self._predict_step1_quantiles(
                    df=df,
                    idx=idx,
                    ticker=ticker,
                    sector=sector,
                    current_state=np.asarray(st, dtype=float),
                    period_days_hint=float(period_days),
                    disable_interest=disable_interest,
                    recent_rollout_context=rollout_context_history,
                    current_observed_context=self._observed_context_from_row(
                        df=df,
                        idx=idx,
                        sector=sector,
                        source="forecast_current_observed",
                    ),
                )
                sanitized = call["sanitized"]
                try:
                    raw_records.append(
                        {
                            "phase": "forecast",
                            "ticker": ticker,
                            "sector": sector,
                            "idx": int(idx),
                            "date_current": str(pd.to_datetime(row_cur["date"]).date()),
                            "date_target": str(pd.to_datetime(row_nxt["date"]).date()),
                            "summary": self._summarize_call_for_record(
                                sector=sector,
                                idx=idx,
                                df=df,
                                call=call,
                            ),
                            "call": self._mask_raw_call_for_storage(call),
                        }
                    )
                except Exception as storage_exc:
                    raw_records.append(
                        {
                            "phase": "forecast",
                            "ticker": ticker,
                            "sector": sector,
                            "idx": int(idx),
                            "date_current": str(pd.to_datetime(row_cur["date"]).date()),
                            "date_target": str(pd.to_datetime(row_nxt["date"]).date()),
                            "warning": f"raw call storage failed but forecast prediction was kept: {storage_exc}",
                        }
                    )
                    self._log(f"[WARN] {ticker}: step idx={idx} raw call storage failed ({storage_exc})")
            except Exception as exc:
                sanitized = self._fallback_sanitized_output(sector=sector, idx=idx, df=df)
                rec_err = {
                    "phase": "forecast",
                    "ticker": ticker,
                    "sector": sector,
                    "idx": int(idx),
                    "date_current": str(pd.to_datetime(row_cur["date"]).date()),
                    "date_target": str(pd.to_datetime(row_nxt["date"]).date()),
                    "error": str(exc),
                    "used_fallback": True,
                    "fallback_sanitized": sanitized,
                    "fallback_diagnostics": self._diagnose_sanitized_output(
                        sector=sector,
                        idx=idx,
                        df=df,
                        sanitized=sanitized,
                        sanitization_meta=None,
                    ),
                }
                if getattr(exc, "snapshot", None):
                    rec_err["call"] = getattr(exc, "snapshot")
                raw_records.append(rec_err)
                self._log(f"[WARN] {ticker}: step idx={idx} fallback used ({exc})")
                if not bool(self.config.continue_on_step_error):
                    raise
            logS_q10, logS_q50, logS_q90 = sanitized["logS"]["q"]
            logS_pred = float(logS_q50)

            theta_pred_q10 = {k: float(sanitized["theta"][k][0]) for k in THETA_KEYS_EVAL}
            theta_pred_q50 = {k: float(sanitized["theta"][k][1]) for k in THETA_KEYS_EVAL}
            theta_pred_q90 = {k: float(sanitized["theta"][k][2]) for k in THETA_KEYS_EVAL}
            theta_pred = dict(theta_pred_q50)

            st_pred_next, diag = simulate_step(
                st,
                float(logS_pred),
                theta_pred,
                float(period_days),
                is_financial=_is_financial_sector(sector),
                disable_interest=disable_interest,
            )
            try:
                st_pred_q10, diag_q10 = simulate_step(
                    st,
                    float(logS_q10),
                    {k: float(theta_pred_q10[k]) for k in THETA_KEYS_EVAL},
                    float(period_days),
                    is_financial=_is_financial_sector(sector),
                    disable_interest=disable_interest,
                )
            except Exception:
                st_pred_q10, diag_q10 = st_pred_next, dict(diag)
            try:
                st_pred_q90, diag_q90 = simulate_step(
                    st,
                    float(logS_q90),
                    {k: float(theta_pred_q90[k]) for k in THETA_KEYS_EVAL},
                    float(period_days),
                    is_financial=_is_financial_sector(sector),
                    disable_interest=disable_interest,
                )
            except Exception:
                st_pred_q90, diag_q90 = st_pred_next, dict(diag)
            _ = st_pred_q10, st_pred_q90, diag_q10, diag_q90

            logS_true = _get_logS(row_nxt)
            logS_true_val, mask_logS = _nan_to_zero_with_mask(logS_true)
            s_true = _get_first_finite(row_nxt, ["S"])
            s_true_val, mask_s = _nan_to_zero_with_mask(s_true)

            st_true = _truth_state_from_row(row_nxt)
            st_true_vals = []
            st_true_masks = []
            for value in st_true.tolist():
                vv, mm = _nan_to_zero_with_mask(float(value) if np.isfinite(value) else float("nan"))
                st_true_vals.append(vv)
                st_true_masks.append(mm)

            rec: Dict[str, Any] = dict(
                ticker=str(ticker),
                sector=str(sector),
                date=str(pd.to_datetime(row_nxt["date"]).date()),
                idx=int(idx + 1),
                step=int((idx + 1) - idx0),
                period_days=float(period_days),
                period_days_panel=float(period_days_panel),
                period_days_date=float(period_days_date) if np.isfinite(period_days_date) else float("nan"),
                logS_pred=float(logS_pred),
                logS_pred_q10=float(logS_q10),
                logS_pred_q50=float(logS_q50),
                logS_pred_q90=float(logS_q90),
                logS_true=float(logS_true_val),
                mask_logS=int(mask_logS),
                S_true=float(s_true_val),
                mask_S=int(mask_s),
                **{f"theta_{k}": float(theta_pred.get(k, 0.0)) for k in THETA_KEYS_EVAL},
                **{f"theta_{k}_q10": float(theta_pred_q10.get(k, 0.0)) for k in THETA_KEYS_EVAL},
                **{f"theta_{k}_q50": float(theta_pred_q50.get(k, 0.0)) for k in THETA_KEYS_EVAL},
                **{f"theta_{k}_q90": float(theta_pred_q90.get(k, 0.0)) for k in THETA_KEYS_EVAL},
            )

            theta_true = {}
            theta_mask = {}
            for k in THETA_KEYS_EVAL:
                value = _get_first_finite(row_nxt, [k])
                vv, mm = _nan_to_zero_with_mask(value)
                if _is_financial_sector(sector) and (k in ["m_gross", "DSO", "DIO", "DPO"]):
                    vv, mm = 0.0, 0
                theta_true[k] = vv
                theta_mask[k] = mm
            rec.update({f"theta_true_{k}": float(v) for k, v in theta_true.items()})
            rec.update({f"mask_theta_{k}": int(m) for k, m in theta_mask.items()})

            def abs_if_finite(x: float) -> float:
                return float(abs(x)) if np.isfinite(x) else float("nan")

            flow_specs = {
                "COGS": (["COGS"], None),
                "OPEX": (["OPEX"], None),
                "Tax": (["Tax"], None),
                "NI": (["NI"], None),
                "Div": (["Div"], abs_if_finite),
                "Int": (["I"], abs_if_finite),
                "TA": (["TA"], None),
                "TL": (["TL"], None),
            }
            for name, (cols, fn) in flow_specs.items():
                value = _get_first_finite(row_nxt, cols)
                if fn is not None:
                    value = fn(value)
                vv, mm = _nan_to_zero_with_mask(value)
                rec[f"{name}_true"] = float(vv)
                rec[f"mask_{name}"] = int(mm)

            eq = _get_first_finite(row_nxt, ["EquityIssues"])
            bb = _get_first_finite(row_nxt, ["Buyback"])
            mm = int(np.isfinite(eq) or np.isfinite(bb))
            eqv = float(eq) if np.isfinite(eq) else 0.0
            bbv = float(bb) if np.isfinite(bb) else 0.0
            rec["NetEq_true"] = float(eqv - bbv) if mm else 0.0
            rec["mask_NetEq"] = int(mm)

            rec.update({f"pred_{k}": float(v) for k, v in diag.items()})
            rec.update({f"state_pred_{STATE_COLS[i]}": float(st_pred_next[i]) for i in range(len(STATE_COLS))})
            rec.update({f"state_true_{STATE_COLS[i]}": float(st_true_vals[i]) for i in range(len(STATE_COLS))})
            rec.update({f"mask_state_{STATE_COLS[i]}": int(st_true_masks[i]) for i in range(len(STATE_COLS))})
            rows.append(rec)

            rollout_context_history.append(
                self._rollout_context_entry(
                    source="forecast_predicted_q50",
                    date=row_nxt["date"],
                    idx=int(idx + 1),
                    period_days=float(period_days),
                    logS=float(logS_pred),
                    theta=theta_pred,
                    state=st_pred_next,
                    diag=diag,
                )
            )

            st = st_pred_next
            logS_t = float(logS_pred)

        return pd.DataFrame(rows), raw_records

    def _save_one_ticker_outputs(self, ticker: str, out: pd.DataFrame, raw_records: List[Dict[str, Any]]) -> None:
        os.makedirs(self.config.out_dir, exist_ok=True)
        safe_ticker = _safe_name(ticker)
        primary_csv = os.path.join(self.config.out_dir, f"{safe_ticker}_llm_backtest.csv")
        out.to_csv(primary_csv, index=False)
        if self.config.duplicate_tft_filename:
            compat_csv = os.path.join(self.config.out_dir, f"{safe_ticker}_tft_backtest.csv")
            out.to_csv(compat_csv, index=False)
        json_path = os.path.join(self.config.out_dir, str(self.config.raw_json_subdir), f"{safe_ticker}_llm_outputs.json")
        payload = {
            "ticker": ticker,
            "out_csv": os.path.basename(primary_csv),
            "compat_csv": os.path.basename(compat_csv) if self.config.duplicate_tft_filename else None,
            "config_snapshot": {
                "model": self.config.api.model,
                "base_url": self.config.api.base_url,
                "endpoint": self.config.api.endpoint,
                "temperature": float(self.config.api.temperature),
                "max_tokens": int(self.config.api.max_tokens),
                "prompt_history_window": int(self.config.prompt_history_window),
                "rollout_context_window": int(self.config.rollout_context_window),
                "llm_prompt_mode": "company_history_and_formula_grounded_no_theory_prior",
                "retry_on_invalid_json": int(self.config.retry_on_invalid_json),
                "save_raw_prompts": bool(self.config.save_raw_prompts),
                "save_full_raw_response": bool(self.config.save_full_raw_response),
                "include_sanitizer_diagnostics": bool(self.config.include_sanitizer_diagnostics),
            },
            "records": raw_records,
        }
        JsonRepository.dump(json_path, payload)
        print(f"[OK] {ticker}: {primary_csv}")

    def run(self) -> None:
        os.makedirs(self.config.out_dir, exist_ok=True)
        selected = self.selected_tickers()
        all_frames: List[pd.DataFrame] = []
        summary: List[Dict[str, Any]] = []

        for ticker in selected:
            self._log(f"[RUN] loading {ticker}")
            df = load_panel(self.config.data_dir, ticker)
            sector = str(df["sector"].iloc[-1]) if ("sector" in df.columns and len(df) > 0) else ""
            disable_interest = bool(self.config.disable_interest_for_banks and _is_financial_sector(sector))
            try:
                out, raw_records = self.run_backtest_one_ticker(
                    df=df,
                    ticker=ticker,
                    sector=sector,
                    warmup=int(self.config.warmup),
                    disable_interest=disable_interest,
                )
            except Exception as exc:
                print(f"[WARN] {ticker}: backtest skipped ({exc})")
                summary.append({"ticker": ticker, "status": "skipped", "error": str(exc)})
                continue

            self._save_one_ticker_outputs(ticker, out, raw_records)
            all_frames.append(out)
            self._log(f"[RUN] finished {ticker}: rows={len(out)} raw_records={len(raw_records)}")
            model_forecast_records = [r for r in raw_records if r.get("phase") == "forecast" and not bool(r.get("used_fallback", False))]
            fallback_forecast_records = [r for r in raw_records if r.get("phase") == "forecast" and bool(r.get("used_fallback", False))]
            parser_repaired_calls = sum(int(bool((r.get("summary") or {}).get("parser_repair_used", False))) for r in model_forecast_records)
            history_default_like_calls = sum(int(bool(((r.get("summary") or {}).get("diagnostics") or {}).get("fully_history_default_like", False))) for r in model_forecast_records)
            summary.append({
                "ticker": ticker,
                "status": "ok",
                "rows": int(len(out)),
                "raw_records": int(len(raw_records)),
                "forecast_steps": int(len(model_forecast_records) + len(fallback_forecast_records)),
                "forecast_steps_model": int(len(model_forecast_records)),
                "forecast_steps_fallback": int(len(fallback_forecast_records)),
                "parser_repaired_calls": int(parser_repaired_calls),
                "history_default_like_model_calls": int(history_default_like_calls),
                "baseline_like_model_calls": int(history_default_like_calls),
            })

        if self.config.save_one_file and all_frames:
            big = pd.concat(all_frames, ignore_index=True)
            big.to_csv(os.path.join(self.config.out_dir, "backtest_all_llm.csv"), index=False)
            if self.config.duplicate_tft_filename:
                big.to_csv(os.path.join(self.config.out_dir, "backtest_all.csv"), index=False)

        JsonRepository.dump(os.path.join(self.config.out_dir, "run_summary.json"), summary)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--endpoint", type=str, default="/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api_key", type=str, default="")
    ap.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    ap.add_argument("--timeout_s", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--min_ar1_points", type=int, default=3)
    ap.add_argument("--save_one_file", action="store_true")
    ap.add_argument("--disable_interest_for_banks", action="store_true")
    ap.add_argument("--tickers", type=str, default="")
    ap.add_argument("--prompt_history_window", type=int, default=6)
    ap.add_argument("--rollout_context_window", type=int, default=4)
    ap.add_argument("--no_compat_tft_name", action="store_true")
    ap.add_argument("--no_response_format_json", action="store_true")
    ap.add_argument("--stop_on_step_error", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--progress_every_calls", type=int, default=1)
    ap.add_argument("--debug_preview_chars", type=int, default=400)
    ap.add_argument("--retry_on_invalid_json", type=int, default=1)
    ap.add_argument("--no_full_raw_response", action="store_true")
    ap.add_argument("--no_sanitizer_diagnostics", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    api_cfg = LLMApiConfig(
        enabled=True,
        base_url=args.base_url,
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        timeout_s=args.timeout_s,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_response_format_json=not bool(args.no_response_format_json),
    )
    runner = LLMBacktestRunner(
        LLMBacktestConfig(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            api=api_cfg,
            warmup=args.warmup,
            min_ar1_points=args.min_ar1_points,
            save_one_file=bool(args.save_one_file),
            disable_interest_for_banks=bool(args.disable_interest_for_banks),
            tickers=args.tickers,
            prompt_history_window=args.prompt_history_window,
            rollout_context_window=args.rollout_context_window,
            duplicate_tft_filename=not bool(args.no_compat_tft_name),
            continue_on_step_error=not bool(args.stop_on_step_error),
            verbose=not bool(args.quiet),
            progress_every_calls=args.progress_every_calls,
            debug_preview_chars=args.debug_preview_chars,
            retry_on_invalid_json=args.retry_on_invalid_json,
            save_full_raw_response=not bool(args.no_full_raw_response),
            include_sanitizer_diagnostics=not bool(args.no_sanitizer_diagnostics),
        )
    )
    runner.run()


__all__ = [
    "LLMApiConfig",
    "LLMBacktestConfig",
    "LLMBacktestRunner",
    "main",
]


if __name__ == "__main__":
    main()
