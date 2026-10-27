import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import prepare_property_features  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
REQUIRED_FIELDS = {
    "POSTED_BY", "UNDER_CONSTRUCTION", "RERA", "BHK_NO.", "BHK_OR_RK",
    "SQUARE_FT", "READY_TO_MOVE", "RESALE", "ADDRESS", "LONGITUDE", "LATITUDE",
}
FLAG_FIELDS = ("UNDER_CONSTRUCTION", "RERA", "READY_TO_MOVE", "RESALE")
NUMERIC_BOUNDS = {
    "BHK_NO.": (1, 20),
    "SQUARE_FT": (1, 1_000_000),
    "LONGITUDE": (-180, 180),
    "LATITUDE": (-180, 180),
}


@lru_cache(maxsize=1)
def get_model():
    """Load the trained pipeline on first prediction, not during Django startup."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Place a trusted trained pipeline there or run the documented training command."
        )
    return joblib.load(MODEL_PATH)


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    missing = sorted(REQUIRED_FIELDS - payload.keys())
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))

    if payload["POSTED_BY"] not in {"Owner", "Dealer", "Builder"}:
        raise ValueError("POSTED_BY must be Owner, Dealer, or Builder.")
    if payload["BHK_OR_RK"] not in {"BHK", "RK"}:
        raise ValueError("BHK_OR_RK must be BHK or RK.")
    if not isinstance(payload["ADDRESS"], str) or not payload["ADDRESS"].strip():
        raise ValueError("ADDRESS must be a non-empty string.")
    if len(payload["ADDRESS"]) > 300:
        raise ValueError("ADDRESS must be 300 characters or fewer.")

    validated = payload.copy()
    for field in FLAG_FIELDS:
        try:
            value = int(payload[field])
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be 0 or 1.") from None
        if str(payload[field]).strip() not in {"0", "1"} and payload[field] not in (0, 1, False, True):
            raise ValueError(f"{field} must be 0 or 1.")
        validated[field] = value

    for field, (minimum, maximum) in NUMERIC_BOUNDS.items():
        try:
            value = float(payload[field])
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be numeric.") from None
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{field} must be between {minimum} and {maximum}.")
        validated[field] = int(value) if field == "BHK_NO." else value

    return validated


@require_GET
def health(request):
    """Return lightweight service and model-artifact readiness information."""
    model_exists = MODEL_PATH.is_file()
    return JsonResponse({
        "status": "ok" if model_exists else "degraded",
        "service": "real-estate-price-estimator",
        "model_available": model_exists,
    }, status=200 if model_exists else 503)


@ensure_csrf_cookie
def home(request):
    # Browser-facing MapTiler key is public by design; restrict it to your site origin in MapTiler.
    return render(request, "predictions/index.html", {
        "maptiler_api_key": os.environ.get("MAPTILER_API_KEY", "").strip(),
    })


@require_POST
def predict_price(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
        payload = validate_payload(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse({"error": "Request body must contain valid JSON."}, status=400)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    try:
        features = prepare_property_features(pd.DataFrame([payload]))
        prediction = float(get_model().predict(features)[0])
        if not math.isfinite(prediction) or prediction < 0:
            return JsonResponse({"error": "The model returned an invalid prediction."}, status=500)
        return JsonResponse({
            "predicted_price_lakhs": round(prediction, 2),
            "predicted_price_rupees": round(prediction * 100_000),
            "currency": "INR",
        })
    except FileNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=503)
    except Exception:
        # Avoid returning implementation details to clients; inspect server logs for traceback.
        import logging
        logging.getLogger(__name__).exception("Property price prediction failed")
        return JsonResponse({"error": "Prediction failed. Check the server logs."}, status=500)
