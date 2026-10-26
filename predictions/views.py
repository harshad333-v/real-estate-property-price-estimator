
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import prepare_property_features

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
model = joblib.load(MODEL_PATH)


@require_POST
def predict_price(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))

        required_fields = [
            "POSTED_BY",
            "UNDER_CONSTRUCTION",
            "RERA",
            "BHK_NO.",
            "BHK_OR_RK",
            "SQUARE_FT",
            "READY_TO_MOVE",
            "RESALE",
            "ADDRESS",
            "LONGITUDE",
            "LATITUDE",
        ]

        missing = [
            field for field in required_fields
            if field not in payload
        ]
        if missing:
            return JsonResponse(
                {"error": "Missing fields", "fields": missing},
                status=400,
            )

        raw = pd.DataFrame([payload])
        features = prepare_property_features(raw)
        prediction = float(model.predict(features)[0])

        if not pd.notna(prediction) or prediction < 0:
            return JsonResponse(
                {"error": "The model returned an invalid prediction."},
                status=500,
            )

        return JsonResponse({
            "predicted_price_lakhs": round(prediction, 2),
            "predicted_price_rupees": round(prediction * 100000),
            "currency": "INR",
        })

    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception:
        return JsonResponse(
            {"error": "Prediction failed. Check the server console."},
            status=500,
        )
@ensure_csrf_cookie
def home(request):
    return render(request, "predictions/index.html")