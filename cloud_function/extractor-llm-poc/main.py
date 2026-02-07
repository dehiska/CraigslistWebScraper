# main.py
# Purpose: Robust LLM-based vehicle extractor with strict validation
# Strategy:
#   - Gemini (Vertex AI) for probabilistic extraction
#   - Deterministic validation using vehicle_models.json
#   - Reject contact/location noise (e.g. New Haven, Contact Info)
#   - Trim stripping + base-model fallback (CRV EX -> CRV; Sierra 1500HD -> Sierra)
#   - Make aliases (Chevy -> Chevrolet; Mini Cooper -> MINI)
#   - Small make inference (E-250 -> Ford)
#   - Never emit bogus make/model; unresolved is explicit

import os
import re
import json
import logging
import time
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

# -------------------- RETRIES --------------------
READ_RETRY = gax_retry.Retry(
    predicate=gax_retry.if_transient_error,
    initial=1.0, maximum=10.0, multiplier=2.0, deadline=120.0
)

def _if_llm_retryable(exc):
    return isinstance(exc, (ResourceExhausted, InternalServerError, Aborted, DeadlineExceeded))

LLM_RETRY = gax_retry.Retry(
    predicate=_if_llm_retryable,
    initial=5.0, maximum=30.0, multiplier=2.0, deadline=180.0
)

# -------------------- GLOBALS --------------------
storage_client = storage.Client()
_CACHED_MODEL = None
_VEHICLE_LOOKUP = None

RUN_ID_ISO_RE = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")

# -------------------- NORMALIZATION --------------------
def _norm_str(s):
    if s is None:
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
    # Remove punctuation/spaces to match Kaggle normalization
    return re.sub(r"[^A-Z0-9]", "", s.upper())

# -------------------- MAKE ALIASES / INFERENCE --------------------
MAKE_ALIASES = {
    "CHEVY": "CHEVROLET",
    "CHEVROLET": "CHEVROLET",
    "MINI COOPER": "MINI",
    "MINI": "MINI",
}

def _canonical_make(make: Optional[str]) -> Optional[str]:
    if not make:
        return None
    m = make.strip().upper()
    return MAKE_ALIASES.get(m, m)

def _infer_make_from_model(make: Optional[str], model: Optional[str]) -> Optional[str]:
    """
    If make missing but model implies it (E-250, F-150, etc.), fill it.
    Keep this conservative.
    """
    if make:
        return make
    if not model:
        return None

    m = model.strip().upper()
    # Ford E-series + F-series
    if re.search(r"\bE[\-\s]?(150|250|350)\b", m) or re.search(r"\bF[\-\s]?(150|250|350)\b", m):
        return "FORD"

    return None

# -------------------- TRIM STRIPPING + BASE MODEL --------------------
BAD_KEYWORDS = {
    "CONTACT", "INFORMATION", "PHONE", "EMAIL", "CALL", "TEXT",
    "REPLY", "POSTED", "NEW", "NORTH", "SOUTH", "EAST", "WEST", "HAVEN"
}

TRIM_TOKENS = {
    "EX","LX","DX","SE","LE","XLE","XSE","SR","SR5","S","SV","SL","SLT","LT","LS","LTD","LIMITED",
    "SPORT","TOURING","PREMIUM","PLATINUM","BASE","SEL","ST","RS","SS",
    "AWD","4WD","FWD","RWD","4X4","2WD",
    "SEDAN","COUPE","HATCH","HATCHBACK","WAGON","CONVERTIBLE","TRUCK","VAN","SUV","CROSSOVER",
    "CARGO",
    "HYBRID","EV","ELECTRIC","TURBO","DIESEL",
    "HD"  # helps with "1500HD" splitting cases
}

def _is_engine_or_package_token(tok: str) -> bool:
    """
    Catch tokens like 2.5I, 3.6R, 5.7L, 2500HD, 1500HD etc.
    """
    t = tok.upper()
    t = re.sub(r"[^A-Z0-9\.]", "", t)
    if re.fullmatch(r"\d+(\.\d+)?[A-Z]*", t):  # 2.5I, 36R, 57L
        return True
    if re.fullmatch(r"\d{3,4}(HD)?", t):       # 1500, 2500, 2500HD
        return True
    return False

def _strip_trims(model: str) -> str:
    """
    Keep left-most base model words and drop trim/drivetrain/body/engine tokens.
    """
    if not model:
        return model

    parts = re.split(r"\s+", model.strip().upper())
    kept: List[str] = []

    for p in parts:
        p_clean = re.sub(r"[^A-Z0-9\-\.]", "", p)  # keep - and . for tokens
        if not p_clean:
            continue

        if p_clean in TRIM_TOKENS:
            break
        if _is_engine_or_package_token(p_clean):
            break

        kept.append(p_clean)

    return " ".join(kept) if kept else model

def _base_model_token(model: str) -> str:
    """
    Get just the first word token (best-effort base), e.g.
    "SAVANA 2500HD CARGO" -> "SAVANA"
    "FORESTER 2.5I AWD" -> "FORESTER"
    """
    if not model:
        return model
    first = model.strip().split()[0]
    return first

def _apply_make_specific_model_aliases(make_u: str, model_raw: str) -> str:
    """
    Fix known dataset quirks / naming differences.
    """
    mnorm = _norm_model(model_raw)

    # Mazda: dataset has model "5" not "MAZDA5"
    if make_u == "MAZDA" and mnorm == "MAZDA5":
        return "5"

    # If you want to add more, do it here.

    return model_raw

# -------------------- VEHICLE DATA --------------------
def _load_vehicle_models() -> Dict[str, List[str]]:
    global _VEHICLE_LOOKUP
    if _VEHICLE_LOOKUP is not None:
        return _VEHICLE_LOOKUP

    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(VEHICLE_MODELS_PATH)
        data = json.loads(blob.download_as_text(retry=READ_RETRY))
        logging.info("Loaded vehicle_models.json from GCS")
    except Exception:
        with open(VEHICLE_MODELS_PATH, "r") as f:
            data = json.load(f)
        logging.info("Loaded vehicle_models.json from local file")

    lookup: Dict[str, List[str]] = {}
    for entry in data:
        make = str(entry["Make"]).strip().upper()
        models = [_norm_model(m) for m in entry.get("Models", []) if m]
        lookup.setdefault(make, []).extend(models)

    # de-dupe
    lookup = {k: sorted(set(v)) for k, v in lookup.items()}

    _VEHICLE_LOOKUP = lookup
    logging.info(f"Vehicle lookup loaded: {len(lookup)} makes")
    return lookup

def _validate_make_model(make: Optional[str], model: Optional[str]) -> Tuple[bool, str]:
    lookup = _load_vehicle_models()

    # Infer + canonicalize make
    make2 = _infer_make_from_model(make, model)
    make_u = _canonical_make(make2)

    if not make_u or not model:
        return False, "missing_make_or_model"

    # Quick noise reject
    if make_u in BAD_KEYWORDS:
        return False, "geographic_or_header_noise"

    # Make not in dataset? (Freightliner/Workhorse are real but not in this Kaggle list)
    if make_u not in lookup:
        return False, f"make_not_found:{make2 or make_u}"

    # Apply make-specific model aliasing before stripping
    model_raw = model.strip()
    model_raw = _apply_make_specific_model_aliases(make_u, model_raw)

    # Prepare candidate model forms
    stripped = _strip_trims(model_raw)
    base = _base_model_token(stripped)

    model_n = _norm_model(model_raw)
    stripped_n = _norm_model(stripped)
    base_n = _norm_model(base)

    # Noise reject on model variants
    if model_n in BAD_KEYWORDS or stripped_n in BAD_KEYWORDS or base_n in BAD_KEYWORDS:
        return False, "geographic_or_header_noise"

    models = lookup[make_u]

    # 1) exact matches
    if model_n in models:
        return True, "exact_match"
    if stripped_n in models:
        return True, "exact_match_stripped"
    if base_n in models:
        return True, "exact_match_base"

    # 2) prefix / fuzzy matches
    if any(m.startswith(model_n) or model_n.startswith(m) for m in models):
        return True, "prefix_match"
    if any(m.startswith(stripped_n) or stripped_n.startswith(m) for m in models):
        return True, "prefix_match_stripped"
    if any(m.startswith(base_n) for m in models) and len(base_n) >= 3:
        # base token should match things like:
        # GMC SAVANA* , GMC SIERRA*, CHEVROLET BLAZER*
        return True, "prefix_match_base"

    return False, f"model_not_found:{model_raw}"

# -------------------- VERTEX MODEL --------------------
def _get_vertex_model() -> GenerativeModel:
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        vertexai.init(project=PROJECT_ID, location=REGION)
        _CACHED_MODEL = GenerativeModel(LLM_MODEL)
    return _CACHED_MODEL

# -------------------- GCS HELPERS --------------------
def _list_per_listing_jsonl_for_run(bucket: str, run_id: str) -> List[str]:
    prefix = f"{STRUCTURED_PREFIX}/run_id={run_id}/jsonl/"
    b = storage_client.bucket(bucket)
    return sorted(blob.name for blob in b.list_blobs(prefix=prefix) if blob.name.endswith(".jsonl"))

def _download_text(blob_name: str) -> str:
    return storage_client.bucket(BUCKET_NAME).blob(blob_name).download_as_text(
        retry=READ_RETRY, timeout=120
    ).replace("\u00a0", " ")

def _upload_jsonl(blob_name: str, record: dict):
    storage_client.bucket(BUCKET_NAME).blob(blob_name).upload_from_string(
        json.dumps(record) + "\n",
        content_type="application/x-ndjson",
    )

# -------------------- LLM EXTRACTION --------------------
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
You are extracting vehicle information from a Craigslist car listing.

Rules:
- Extract ONLY the vehicle MAKE (manufacturer) and MODEL.
- Ignore contact info, addresses, location headers, and seller names.
- NEVER use words like Contact, Information, New, Haven, North, West as make/model.
- If unsure, return null — do NOT guess.

Return JSON with: price, year, make, model, mileage.

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

    return {
        "price": _safe_int(parsed.get("price")),
        "year": _safe_int(parsed.get("year")),
        "mileage": _safe_int(parsed.get("mileage")),
        "make": _norm_str(parsed.get("make")),
        "model": _norm_str(parsed.get("model")),
    }

def _extract_with_validation(raw_text: str) -> dict:
    parsed = None
    reason = "unknown"

    for attempt in range(1, 5):
        parsed = _vertex_extract_fields(raw_text)
        valid, reason = _validate_make_model(parsed.get("make"), parsed.get("model"))

        if valid:
            # Canonicalize make/model on output for consistency
            make_out = _canonical_make(_infer_make_from_model(parsed.get("make"), parsed.get("model")))
            model_out = parsed.get("model")
            return {
                **parsed,
                "make": make_out if make_out else parsed.get("make"),
                "model": model_out,
                "validation_status": "valid",
                "validation_reason": reason,
                "extraction_attempts": attempt,
            }

        time.sleep(1.5 * attempt)

    return {
        "price": parsed.get("price") if parsed else None,
        "year": parsed.get("year") if parsed else None,
        "mileage": parsed.get("mileage") if parsed else None,
        "make": UNRESOLVED,
        "model": UNRESOLVED,
        "raw_make_guess": parsed.get("make") if parsed else None,
        "raw_model_guess": parsed.get("model") if parsed else None,
        "validation_status": "rejected",
        "validation_reason": reason,
        "extraction_attempts": 4,
    }

# -------------------- HTTP ENTRY --------------------
def llm_extract_http(request: Request):
    logging.getLogger().setLevel(logging.INFO)

    body = request.get_json(silent=True) or {}
    run_id = body.get("run_id")
    max_files = int(body.get("max_files") or 0)

    if not run_id:
        return jsonify({"ok": False, "error": "run_id required"}), 400

    inputs = _list_per_listing_jsonl_for_run(BUCKET_NAME, run_id)
    if not inputs:
        return jsonify({"ok": True, "written": 0, "valid": 0, "run_id": run_id, "note": "no input jsonl found"}), 200

    if max_files > 0:
        inputs = inputs[:max_files]

    written = valid = 0

    for in_key in inputs:
        base = json.loads(_download_text(in_key))
        post_id = base.get("post_id")
        if not post_id:
            continue

        listing_text = _download_text(base["source_txt"])
        extracted = _extract_with_validation(listing_text)

        if extracted.get("validation_status") == "valid":
            valid += 1

        out = {"post_id": post_id, "run_id": run_id, **extracted}
        out_key = f"{STRUCTURED_PREFIX}/run_id={run_id}/jsonl_llm/{post_id}_llm.jsonl"
        _upload_jsonl(out_key, out)

        written += 1

    return jsonify({"ok": True, "written": written, "valid": valid}), 200
