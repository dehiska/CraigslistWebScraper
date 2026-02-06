# main.py
# Purpose: PoC LLM extractor that reads your existing per-listing JSONL records,
# fetches the original TXT, asks an LLM (Vertex AI) to extract fields, and writes
# a sibling "<post_id>_llm.jsonl" to the NEW 'jsonl_llm/' sub-directory.
#
# FINAL FIXES INCLUDED:
# 1. Schema updated to use "type": "string" + "nullable": True.
# 2. system_instruction removed from GenerationConfig and merged into prompt.
# 3. LLM_MODEL set to 'gemini-2.5-flash' (Fixes 404/NotFound error).
# 4. "additionalProperties": False removed from schema (Fixes internal ParseError).
# 5. Non-breaking spaces (U+00A0) replaced with standard spaces (U+0020). <--- FIX FOR THIS ERROR
# main.py
# Purpose: Improved LLM extractor with robust vehicle make/model extraction
# 
# KEY IMPROVEMENTS:
# 1. Loads Kaggle vehicle_models.json for validation
# 2. Enhanced prompt engineering to avoid extracting "Contact Information"
# 3. Retry logic: 3 additional attempts with validation
# 4. Better field extraction and normalization
# 5. Non-breaking space fixes (U+00A0 → U+0020)

import os
import re
import json
import logging
import traceback
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple

from flask import Request, jsonify
from google.api_core import retry as gax_retry
from google.cloud import storage

# ---- REQUIRED VERTEX AI IMPORTS ----
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.api_core.exceptions import ResourceExhausted, InternalServerError, Aborted, DeadlineExceeded

# -------------------- ENV --------------------
PROJECT_ID           = os.getenv("PROJECT_ID", "")
REGION               = os.getenv("REGION", "us-central1")
BUCKET_NAME          = os.getenv("GCS_BUCKET", "")
STRUCTURED_PREFIX    = os.getenv("STRUCTURED_PREFIX", "structured")
LLM_PROVIDER         = os.getenv("LLM_PROVIDER", "vertex").lower()
LLM_MODEL            = os.getenv("LLM_MODEL", "gemini-2.5-flash")
OVERWRITE_DEFAULT    = os.getenv("OVERWRITE", "false").lower() == "true"
MAX_FILES_DEFAULT    = int(os.getenv("MAX_FILES", "0") or 0)
VEHICLE_MODELS_PATH  = os.getenv("VEHICLE_MODELS_PATH", "vehicle_models.json")

# GCS READ RETRY
READ_RETRY = gax_retry.Retry(
    predicate=gax_retry.if_transient_error,
    initial=1.0, maximum=10.0, multiplier=2.0, deadline=120.0
)

# LLM API RETRY PREDICATE
def _if_llm_retryable(exception):
    """Checks if the exception is transient and should trigger a retry."""
    return isinstance(exception, (ResourceExhausted, InternalServerError, Aborted, DeadlineExceeded))

# LLM CALL RETRY
LLM_RETRY = gax_retry.Retry(
    predicate=_if_llm_retryable,
    initial=5.0, maximum=30.0, multiplier=2.0, deadline=180.0,
)

storage_client = storage.Client()
_CACHED_MODEL_OBJ = None
_VEHICLE_LOOKUP = None

# Accept BOTH run id styles
RUN_ID_ISO_RE    = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")


# -------------------- VEHICLE DATA LOADER --------------------
def _load_vehicle_models() -> Dict[str, List[str]]:
    """
    Load and normalize the Kaggle vehicle models JSON.
    Returns dict: {MAKE_UPPER: [MODEL_UPPER, ...]}
    """
    global _VEHICLE_LOOKUP
    if _VEHICLE_LOOKUP is not None:
        return _VEHICLE_LOOKUP
    
    try:
        # Try to download from GCS first
        if BUCKET_NAME:
            try:
                bucket = storage_client.bucket(BUCKET_NAME)
                blob = bucket.blob(VEHICLE_MODELS_PATH)
                data_str = blob.download_as_text(retry=READ_RETRY, timeout=60)
                data = json.loads(data_str)
                logging.info(f"Loaded vehicle_models.json from GCS: {BUCKET_NAME}/{VEHICLE_MODELS_PATH}")
            except Exception as e:
                logging.warning(f"Could not load from GCS: {e}")
                # Fall back to local file
                with open(VEHICLE_MODELS_PATH, 'r') as f:
                    data = json.load(f)
                logging.info(f"Loaded vehicle_models.json from local path: {VEHICLE_MODELS_PATH}")
        else:
            with open(VEHICLE_MODELS_PATH, 'r') as f:
                data = json.load(f)
            logging.info(f"Loaded vehicle_models.json from local path: {VEHICLE_MODELS_PATH}")
        
        # Build normalized lookup: MAKE -> [MODELS]
        lookup = {}
        for entry in data:
            make = entry['Make'].strip().upper()
            models = [m.strip().upper() for m in entry['Models']]
            # Merge if duplicate makes exist
            if make in lookup:
                lookup[make].extend(models)
            else:
                lookup[make] = models
        
        _VEHICLE_LOOKUP = lookup
        logging.info(f"Vehicle lookup loaded: {len(lookup)} makes, "
                     f"{sum(len(models) for models in lookup.values())} total models")
        return lookup
    
    except Exception as e:
        logging.error(f"Failed to load vehicle_models.json: {e}")
        # Return empty dict to allow operation without validation
        _VEHICLE_LOOKUP = {}
        return {}


def _validate_make_model(make: Optional[str], model: Optional[str]) -> Tuple[bool, str]:
    """
    Validate extracted make/model against Kaggle dataset.
    Returns: (is_valid: bool, reason: str)
    """
    lookup = _load_vehicle_models()
    
    if not lookup:
        # No validation data available
        return True, "no_validation_data"
    
    if not make or not model:
        return False, "missing_make_or_model"
    
    # Normalize
    make_upper = make.strip().upper()
    model_upper = model.strip().upper()
    
    # Check for known bad extractions
    bad_makes = ["CONTACT", "INFORMATION", "PHONE", "EMAIL", "CALL", "TEXT", "WEBSITE"]
    bad_models = ["CONTACT", "INFORMATION", "PHONE", "EMAIL", "CALL", "TEXT"]
    
    if make_upper in bad_makes or model_upper in bad_models:
        return False, "extracted_contact_info"
    
    # Check if make exists
    if make_upper not in lookup:
        return False, f"make_not_found:{make}"
    
    # Check if model exists for this make (fuzzy match)
    valid_models = lookup[make_upper]
    
    # Exact match
    if model_upper in valid_models:
        return True, "exact_match"
    
    # Partial match (model contains or is contained in a valid model)
    for valid_model in valid_models:
        if model_upper in valid_model or valid_model in model_upper:
            return True, f"partial_match:{valid_model}"
    
    return False, f"model_not_found:{model}"


# -------------------- HELPERS --------------------
def _get_vertex_model() -> GenerativeModel:
    """Initializes and returns the cached Vertex AI model object."""
    global _CACHED_MODEL_OBJ
    if _CACHED_MODEL_OBJ is None:
        if not PROJECT_ID:
            raise RuntimeError("PROJECT_ID environment variable is missing.")
        
        vertexai.init(project=PROJECT_ID, location=REGION)
        _CACHED_MODEL_OBJ = GenerativeModel(LLM_MODEL)
        logging.info(f"Initialized Vertex AI model: {LLM_MODEL} in {REGION}")
    return _CACHED_MODEL_OBJ


def _list_structured_run_ids(bucket: str, structured_prefix: str) -> list[str]:
    """List 'structured/run_id=*/' directories and return normalized run_ids."""
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:
        pass

    runs = []
    for pref in getattr(it, "prefixes", []):
        tail = pref.rstrip("/").split("/")[-1]
        if tail.startswith("run_id="):
            cand = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(cand) or RUN_ID_PLAIN_RE.match(cand):
                runs.append(cand)
    return sorted(runs)


def _normalize_run_id_iso(run_id: str) -> str:
    """Normalize run_id to ISO8601 Z string for provenance."""
    try:
        if RUN_ID_ISO_RE.match(run_id):
            dt = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        elif RUN_ID_PLAIN_RE.match(run_id):
            dt = datetime.strptime(run_id, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        else:
            raise ValueError("unsupported run_id")
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _list_per_listing_jsonl_for_run(bucket: str, run_id: str) -> list[str]:
    """Return *input* per-listing JSONL object names for a given run_id."""
    prefix = f"{STRUCTURED_PREFIX}/run_id={run_id}/jsonl/"
    bucket_obj = storage_client.bucket(bucket)
    names = []
    for b in bucket_obj.list_blobs(prefix=prefix):
        if not b.name.endswith(".jsonl"):
            continue
        names.append(b.name)
    return names


def _download_text(blob_name: str) -> str:
    """Download text from GCS with retry."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    text = blob.download_as_text(retry=READ_RETRY, timeout=120)
    # Fix non-breaking spaces
    return text.replace('\u00a0', ' ')


def _upload_jsonl_line(blob_name: str, record: dict):
    """Upload a single JSONL record to GCS."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    blob.upload_from_string(line, content_type="application/x-ndjson")


def _blob_exists(blob_name: str) -> bool:
    """Check if blob exists in GCS."""
    bucket = storage_client.bucket(BUCKET_NAME)
    return bucket.blob(blob_name).exists()


def _safe_int(x):
    """Safely convert to int, handling None, empty strings, and commas."""
    try:
        if x is None or x == "":
            return None
        return int(str(x).replace(",", "").strip())
    except Exception:
        return None


def _norm_str(s):
    """Normalize string: strip, handle None, return None for empty."""
    if s is None:
        return None
    s = str(s).strip()
    return s if s else None


# -------------------- IMPROVED VERTEX AI EXTRACTION --------------------
def _vertex_extract_fields(raw_text: str, attempt: int = 1) -> dict:
    """
    Extract vehicle fields from raw listing text using Vertex AI.
    
    Args:
        raw_text: The raw Craigslist listing text
        attempt: Current attempt number (1-4 for retry logic)
    
    Returns:
        Dict with extracted fields: price, year, make, model, mileage
    """
    model = _get_vertex_model()

    # Strict JSON schema
    schema = {
        "type": "object",
        "properties": {
            "price": {"type": "integer", "nullable": True},
            "year": {"type": "integer", "nullable": True},
            "make": {"type": "string", "nullable": True},
            "model": {"type": "string", "nullable": True},
            "mileage": {"type": "integer", "nullable": True},
        },
        "required": ["price", "year", "make", "model", "mileage"]
    }

    # Enhanced system instruction with specific rules
    sys_instr = """You are extracting vehicle information from a Craigslist car listing.

CRITICAL RULES:
1. MAKE and MODEL must be the VEHICLE MANUFACTURER and VEHICLE MODEL ONLY
2. NEVER extract contact information (phone, email, address) as make or model
3. IGNORE any text that says "Contact Information:" or similar headers
4. Look for the actual vehicle details in the listing body, VIN info, or title
5. Common patterns: "YEAR MAKE MODEL" (e.g., "2021 BMW X3")
6. If you see "Contact Information:" followed by text, SKIP that section entirely
7. Price should be the listing price in USD (remove $ and commas)
8. Year should be 4-digit year (1900-2030 range)
9. Mileage should be in miles (remove commas)
10. If a field is genuinely not present or unclear, use null - DO NOT GUESS

EXAMPLES OF CORRECT EXTRACTION:
- Text: "2021 BMW X3 xDrive30i - Call/Text 718-578-4337"
  → make: "BMW", model: "X3" (NOT "Contact" or "Information")
  
- Text: "Contact Information: John Doe..."
  → SKIP this section, look elsewhere for vehicle info

Extract ONLY these fields as a JSON object: price, year, make, model, mileage."""

    # Combine instruction and text
    if attempt > 1:
        prompt = f"{sys_instr}\n\n[RETRY ATTEMPT {attempt}] Previous attempts failed validation. Be MORE CAREFUL about make and model.\n\nTEXT:\n{raw_text}"
    else:
        prompt = f"{sys_instr}\n\nTEXT:\n{raw_text}"

    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=40,
        candidate_count=1,
        response_mime_type="application/json",
        response_schema=schema,
    )

    # LLM call with retry for transient errors
    max_api_attempts = 3
    resp = None
    for api_attempt in range(max_api_attempts):
        try:
            resp = model.generate_content(prompt, generation_config=gen_cfg)
            break
        except Exception as e:
            if not _if_llm_retryable(e) or api_attempt == max_api_attempts - 1:
                logging.error(f"Fatal/non-retryable LLM error: {e}")
                raise
            
            sleep_time = 5.0 * (2 ** api_attempt)
            logging.warning(f"Transient LLM error on API attempt {api_attempt+1}/{max_api_attempts}. "
                          f"Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)

    if resp is None:
        raise RuntimeError("LLM call failed after all API retries.")

    # Parse response
    parsed = json.loads(resp.text)

    # Normalize fields
    parsed["price"] = _safe_int(parsed.get("price"))
    parsed["year"] = _safe_int(parsed.get("year"))
    parsed["mileage"] = _safe_int(parsed.get("mileage"))
    parsed["make"] = _norm_str(parsed.get("make"))
    parsed["model"] = _norm_str(parsed.get("model"))

    return parsed


def _extract_with_validation(raw_text: str, post_id: str) -> dict:
    """
    Extract fields with validation and retry logic.
    Attempts up to 4 times (1 initial + 3 retries) if validation fails.
    
    Args:
        raw_text: The raw listing text
        post_id: The post ID for logging
    
    Returns:
        Dict with extracted fields and metadata
    """
    max_attempts = 4
    
    for attempt in range(1, max_attempts + 1):
        try:
            logging.info(f"[{post_id}] Extraction attempt {attempt}/{max_attempts}")
            
            # Extract fields
            parsed = _vertex_extract_fields(raw_text, attempt=attempt)
            
            # Validate make and model
            is_valid, reason = _validate_make_model(parsed.get("make"), parsed.get("model"))
            
            if is_valid:
                logging.info(f"[{post_id}] ✓ Validation passed on attempt {attempt}: {reason}")
                return {
                    **parsed,
                    "validation_status": "valid",
                    "validation_reason": reason,
                    "extraction_attempts": attempt
                }
            else:
                logging.warning(f"[{post_id}] ✗ Validation failed on attempt {attempt}: {reason}")
                
                if attempt == max_attempts:
                    # Final attempt failed - return anyway but mark as invalid
                    logging.warning(f"[{post_id}] Max attempts reached. Keeping unvalidated result.")
                    return {
                        **parsed,
                        "validation_status": "invalid",
                        "validation_reason": reason,
                        "extraction_attempts": attempt
                    }
                
                # Wait before retry
                time.sleep(2.0 * attempt)
                
        except Exception as e:
            logging.error(f"[{post_id}] Extraction error on attempt {attempt}: {e}")
            if attempt == max_attempts:
                # Return null values on complete failure
                return {
                    "price": None,
                    "year": None,
                    "make": None,
                    "model": None,
                    "mileage": None,
                    "validation_status": "error",
                    "validation_reason": str(e),
                    "extraction_attempts": attempt
                }
            time.sleep(2.0 * attempt)
    
    # Should never reach here, but just in case
    return {
        "price": None,
        "year": None,
        "make": None,
        "model": None,
        "mileage": None,
        "validation_status": "error",
        "validation_reason": "unknown_error",
        "extraction_attempts": max_attempts
    }


# -------------------- HTTP ENTRY --------------------
def llm_extract_http(request: Request):
    """
    HTTP Cloud Function entry point.
    Reads latest run's per-listing JSONL inputs and writes LLM outputs with validation.
    """
    logging.getLogger().setLevel(logging.INFO)

    if not BUCKET_NAME:
        return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500
    if not PROJECT_ID:
        return jsonify({"ok": False, "error": "missing PROJECT_ID env"}), 500
    if LLM_PROVIDER != "vertex":
        return jsonify({"ok": False, "error": "PoC supports LLM_PROVIDER='vertex' only"}), 400

    # Load vehicle models at startup
    try:
        _load_vehicle_models()
    except Exception as e:
        logging.warning(f"Could not load vehicle models, validation will be skipped: {e}")

    # Parse request body
    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    run_id = body.get("run_id")
    max_files = int(body.get("max_files") or MAX_FILES_DEFAULT or 0)
    overwrite = bool(body.get("overwrite")) if "overwrite" in body else OVERWRITE_DEFAULT

    # Pick newest run if not provided
    if not run_id:
        runs = _list_structured_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        if not runs:
            return jsonify({"ok": False, "error": f"no run_ids found under {STRUCTURED_PREFIX}/"}), 200
        run_id = runs[-1]

    structured_iso = _normalize_run_id_iso(run_id)

    # Get input files
    inputs = _list_per_listing_jsonl_for_run(BUCKET_NAME, run_id)
    if not inputs:
        return jsonify({
            "ok": True,
            "run_id": run_id,
            "processed": 0,
            "written": 0,
            "skipped": 0,
            "errors": 0,
            "valid": 0,
            "invalid": 0
        }), 200
    
    if max_files > 0:
        inputs = inputs[:max_files]

    logging.info(f"Starting LLM extraction for run_id={run_id} ({len(inputs)} files to process)")

    # Counters
    processed = written = skipped = errors = 0
    valid_extractions = invalid_extractions = 0

    for in_key in inputs:
        processed += 1
        try:
            # Read input record
            raw_line = _download_text(in_key).strip()
            if not raw_line:
                raise ValueError("empty input jsonl")
            base_rec = json.loads(raw_line)

            post_id = base_rec.get("post_id")
            if not post_id:
                raise ValueError("missing post_id in input record")

            source_txt_key = base_rec.get("source_txt")
            if not source_txt_key:
                raise ValueError("missing source_txt in input record")

            # Output path: 'jsonl_llm/' folder
            out_prefix = in_key.rsplit("/", 2)[0] + "/jsonl_llm"
            out_key = out_prefix + f"/{post_id}_llm.jsonl"

            # Check if already exists
            if not overwrite and _blob_exists(out_key):
                logging.info(f"[{post_id}] Skipping (already exists)")
                skipped += 1
                continue

            # Fetch raw listing text
            raw_listing = _download_text(source_txt_key)

            # Extract with validation and retry
            extracted = _extract_with_validation(raw_listing, post_id)

            # Track validation stats
            if extracted.get("validation_status") == "valid":
                valid_extractions += 1
            elif extracted.get("validation_status") == "invalid":
                invalid_extractions += 1

            # Compose output record
            out_record = {
                "post_id": post_id,
                "run_id": base_rec.get("run_id", run_id),
                "scraped_at": base_rec.get("scraped_at", structured_iso),
                "source_txt": source_txt_key,
                "price": extracted.get("price"),
                "year": extracted.get("year"),
                "make": extracted.get("make"),
                "model": extracted.get("model"),
                "mileage": extracted.get("mileage"),
                "validation_status": extracted.get("validation_status"),
                "validation_reason": extracted.get("validation_reason"),
                "extraction_attempts": extracted.get("extraction_attempts"),
                "llm_provider": "vertex",
                "llm_model": LLM_MODEL,
                "llm_ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

            # Write output
            _upload_jsonl_line(out_key, out_record)
            written += 1
            
            logging.info(f"[{post_id}] ✓ Written: {extracted.get('make')} {extracted.get('model')} "
                        f"({extracted.get('validation_status')})")

        except Exception as e:
            errors += 1
            logging.error(f"Failed processing {in_key}: {e}\n{traceback.format_exc()}")

    # Final summary
    result = {
        "ok": True,
        "version": "extractor-llm-robust-v2",
        "run_id": run_id,
        "processed": processed,
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "valid_extractions": valid_extractions,
        "invalid_extractions": invalid_extractions,
        "validation_rate": f"{valid_extractions}/{written}" if written > 0 else "0/0"
    }
    
    logging.info(json.dumps(result))
    return jsonify(result), 200
