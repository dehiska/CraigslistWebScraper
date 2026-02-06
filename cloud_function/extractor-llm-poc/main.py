# main.py
# Purpose: Robust LLM-based vehicle extractor with strict validation
# Strategy:
#   - LLM for probabilistic extraction
#   - Deterministic validation layer
#   - Never output NaN / None for unresolved make-model
#   - Preserve rejected guesses for audit + reprocessing

import os
import re
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple

from flask import Request, jsonify
from google.api_core import retry as gax_retry
from google.cloud import storage

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.api_core.exceptions import (
    ResourceExhausted,
    InternalServerError,
    Aborted,
    DeadlineExceeded,
)

# -------------------- ENV --------------------
PROJECT_ID = os.getenv("PROJECT_ID", "craigslist-scraper-4849")
REGION = os.getenv("REGION", "us-central1")
BUCKET_NAME = os.getenv("GCS_BUCKET", "craigslist-scraper-4849")
STRUCTURED_PREFIX = os.getenv("STRUCTURED_PREFIX", "structured")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
VEHICLE_MODELS_PATH = os.getenv("VEHICLE_MODELS_PATH", "vehicle_models.json")

UNRESOLVED = "__UNRESOLVED__"

READ_RETRY = gax_retry.Retry(
    predicate=gax_retry.if_transient_error,
    initial=1.0,
    maximum=10.0,
    multiplier=2.0,
    deadline=120.0,
)

def _if_llm_retryable(exc):
    return isinstance(exc, (ResourceExhausted, InternalServerError, Aborted, DeadlineExceeded))

LLM_RETRY = gax_retry.Retry(
    predicate=_if_llm_retryable,
    initial=5.0,
    maximum=30.0,
    multiplier=2.0,
    deadline=180.0,
)

storage_client = storage.Client()
_CACHED_MODEL = None
_VEHICLE_LOOKUP = None

RUN_ID_ISO_RE = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")

# -------------------- NORMALIZATION --------------------
def _norm_str(s):
    if not s:
        return None
    s = str(s).strip()
    return s if s else None

def _safe_int(x):
    try:
        if x is None or x == "":
            return None
        return int(str(x).replace(",", "").strip())
    except Exception:
        return None

def _norm_model(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())

# -------------------- VEHICLE DATA --------------------
def _load_vehicle_models() -> Dict[str, List[str]]:
    global _VEHICLE_LOOKUP
    if _VEHICLE_LOOKUP is not None:
        return _VEHICLE_LOOKUP

    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(VEHICLE_MODELS_PATH)
        data = json.loads(blob.download_as_text(retry=READ_RETRY))
    except Exception:
        with open(VEHICLE_MODELS_PATH, "r") as f:
            data = json.load(f)

    lookup = {}
    for entry in data:
        make = entry["Make"].strip().upper()
        models = [_norm_model(m) for m in entry["Models"]]
        lookup.setdefault(make, []).extend(models)

    _VEHICLE_LOOKUP = lookup
    return lookup

def _validate_make_model(make: Optional[str], model: Optional[str]) -> Tuple[bool, str]:
    lookup = _load_vehicle_models()

    if not make or not model:
        return False, "missing_make_or_model"

    make_u = make.upper().strip()
    model_n = _norm_model(model)

    bad_keywords = {
        "CONTACT", "INFORMATION", "PHONE", "EMAIL", "CALL", "TEXT",
        "REPLY", "POSTED", "NORTH", "SOUTH", "EAST", "WEST", "HAVEN"
    }

    if make_u in bad_keywords and model_n in bad_keywords:
        return False, "geographic_or_header_noise"

    if make_u not in lookup:
        return False, f"make_not_found:{make}"

    if model_n in lookup[make_u]:
        return True, "exact_match"

    return False, f"model_not_found:{model}"

# -------------------- LLM --------------------
def _get_vertex_model() -> GenerativeModel:
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        vertexai.init(project=PROJECT_ID, location=REGION)
        _CACHED_MODEL = GenerativeModel(LLM_MODEL)
    return _CACHED_MODEL

# -------------------- EXTRACTION --------------------
def _cheap_regex_hints(text: str) -> dict:
    return {
        "year": _safe_int(re.search(r"(19\d{2}|20\d{2})", text).group(1)) if re.search(r"(19\d{2}|20\d{2})", text) else None,
        "price": _safe_int(re.search(r"\$?\s?(\d{3,6})", text).group(1)) if re.search(r"\$?\s?(\d{3,6})", text) else None,
    }

def _vertex_extract_fields(raw_text: str) -> dict:
    model = _get_vertex_model()
    schema = {
        "type": "object",
        "properties": {
            "price": {"type": "integer", "nullable": True},
            "year": {"type": "integer", "nullable": True},
            "make": {"type": "string", "nullable": True},
            "model": {"type": "string", "nullable": True},
            "mileage": {"type": "integer", "nullable": True},
        },
        "required": ["price", "year", "make", "model", "mileage"],
    }

    prompt = f"""
Extract vehicle info.
Rules:
- MAKE and MODEL must be real manufacturer names.
- Ignore contact headers and locations.
- Do NOT guess.

TEXT:
{raw_text}
"""

    cfg = GenerationConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=schema,
    )

    resp = model.generate_content(prompt, generation_config=cfg)
    parsed = json.loads(resp.text)

    parsed["price"] = _safe_int(parsed.get("price"))
    parsed["year"] = _safe_int(parsed.get("year"))
    parsed["mileage"] = _safe_int(parsed.get("mileage"))
    parsed["make"] = _norm_str(parsed.get("make"))
    parsed["model"] = _norm_str(parsed.get("model"))

    return parsed

def _extract_with_validation(raw_text: str) -> dict:
    for attempt in range(1, 5):
        parsed = _vertex_extract_fields(raw_text)
        valid, reason = _validate_make_model(parsed.get("make"), parsed.get("model"))

        if valid:
            return {
                **parsed,
                "validation_status": "valid",
                "validation_reason": reason,
            }

        time.sleep(1.5 * attempt)

    return {
        "price": parsed.get("price"),
        "year": parsed.get("year"),
        "mileage": parsed.get("mileage"),
        "make": UNRESOLVED,
        "model": UNRESOLVED,
        "raw_make_guess": parsed.get("make"),
        "raw_model_guess": parsed.get("model"),
        "validation_status": "rejected",
        "validation_reason": reason,
    }

# -------------------- HTTP ENTRY --------------------
def llm_extract_http(request: Request):
    logging.getLogger().setLevel(logging.INFO)

    body = request.get_json(silent=True) or {}
    run_id = body.get("run_id")

    if not run_id:
        return jsonify({"ok": False, "error": "run_id required"}), 400

    inputs = _list_per_listing_jsonl_for_run(BUCKET_NAME, run_id)
    written = valid = 0

    for in_key in inputs:
        raw_line = storage_client.bucket(BUCKET_NAME).blob(in_key).download_as_text()
        base = json.loads(raw_line)

        listing_text = storage_client.bucket(BUCKET_NAME).blob(
            base["source_txt"]
        ).download_as_text()

        extracted = _extract_with_validation(listing_text)
        if extracted["validation_status"] == "valid":
            valid += 1

        out = {
            "post_id": base["post_id"],
            "run_id": run_id,
            **extracted,
        }

        out_key = in_key.rsplit("/", 2)[0] + f"/jsonl_llm/{base['post_id']}_llm.jsonl"
        storage_client.bucket(BUCKET_NAME).blob(out_key).upload_from_string(
            json.dumps(out) + "\n",
            content_type="application/x-ndjson",
        )

        written += 1

    return jsonify({"ok": True, "written": written, "valid": valid}), 200
