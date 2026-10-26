import json
from unittest.mock import patch

import numpy as np
import pandas as pd
from django.test import SimpleTestCase, Client

from predictions.views import validate_payload


VALID_PAYLOAD = {
    "POSTED_BY": "Owner",
    "UNDER_CONSTRUCTION": 0,
    "RERA": 1,
    "BHK_NO.": 2,
    "BHK_OR_RK": "BHK",
    "SQUARE_FT": 1000,
    "READY_TO_MOVE": 1,
    "RESALE": 1,
    "ADDRESS": "Example Locality, Bengaluru",
    # Dataset-specific preprocessing currently normalizes these columns.
    "LONGITUDE": 12.9716,
    "LATITUDE": 77.5946,
}


class PayloadValidationTests(SimpleTestCase):
    def test_valid_payload_is_accepted(self):
        self.assertEqual(validate_payload(VALID_PAYLOAD)["BHK_NO."], 2)

    def test_missing_field_is_rejected(self):
        payload = VALID_PAYLOAD.copy()
        del payload["ADDRESS"]
        with self.assertRaisesRegex(ValueError, "ADDRESS"):
            validate_payload(payload)

    def test_invalid_flag_is_rejected(self):
        payload = VALID_PAYLOAD.copy()
        payload["RERA"] = 3
        with self.assertRaisesRegex(ValueError, "RERA"):
            validate_payload(payload)

    def test_non_finite_area_is_rejected(self):
        payload = VALID_PAYLOAD.copy()
        payload["SQUARE_FT"] = float("nan")
        with self.assertRaisesRegex(ValueError, "SQUARE_FT"):
            validate_payload(payload)

    def test_empty_address_is_rejected(self):
        payload = VALID_PAYLOAD.copy()
        payload["ADDRESS"] = "  "
        with self.assertRaisesRegex(ValueError, "ADDRESS"):
            validate_payload(payload)


class PredictionEndpointTests(SimpleTestCase):
    def setUp(self):
        self.client = Client(enforce_csrf_checks=False)

    def test_health_reports_missing_model_as_degraded(self):
        with patch("predictions.views.MODEL_PATH.is_file", return_value=False):
            response = self.client.get("/api/health/")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertFalse(response.json()["model_available"])

    def test_health_reports_available_model(self):
        with patch("predictions.views.MODEL_PATH.is_file", return_value=True):
            response = self.client.get("/api/health/")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["model_available"])

    def test_home_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Real Estate Price Estimator")

    def test_invalid_json_returns_400(self):
        response = self.client.post(
            "/api/predict/", data="{not-json", content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    def test_missing_fields_returns_400(self):
        response = self.client.post(
            "/api/predict/", data=json.dumps({}), content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    @patch("predictions.views.get_model")
    def test_valid_request_returns_prediction(self, mocked_get_model):
        mocked_get_model.return_value.predict.return_value = np.array([17.68513])
        response = self.client.post(
            "/api/predict/",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["currency"], "INR")
        self.assertEqual(response.json()["predicted_price_lakhs"], 17.69)
        self.assertEqual(response.json()["predicted_price_rupees"], 1_768_513)

    def test_missing_model_returns_503(self):
        with patch("predictions.views.get_model", side_effect=FileNotFoundError("model missing")):
            response = self.client.post(
                "/api/predict/",
                data=json.dumps(VALID_PAYLOAD),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 503)
