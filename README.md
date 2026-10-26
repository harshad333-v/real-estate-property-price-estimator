# Real Estate Property Price Estimator

A Django web app and scikit-learn pipeline for estimating residential property listing prices in India. It includes data preprocessing, model comparison (Linear Regression, Random Forest, XGBoost), a browser form, and a JSON prediction endpoint.

> Educational project only. These are model estimates from listing data, not certified valuations or verified sale prices.

## Stack
- Python, Django, Pandas, NumPy, scikit-learn, XGBoost, Joblib
- Django template, HTML, Tailwind CSS CDN, vanilla JavaScript
- Jupyter, Matplotlib, Seaborn

## Project structure
```text
backend/                 Django settings and manage.py
predictions/             HTML template, JSON API, tests
src/                     preprocessing, split, model pipeline, training
notebooks/               EDA and project workflow
models/                  generated model artifacts (ignored by Git)
data/README.md            dataset acquisition and placement instructions
audit_dataset.py          dataset audit script
requirements.txt          runtime and analysis dependencies
.env.example              local environment variable template
```

## Local setup (Windows PowerShell)

1. Clone and create the virtual environment:
```powershell
git clone https://github.com/harshad333-v/real-estate-property-price-estimator.git
cd real-estate-property-price-estimator
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

2. Edit `.env`. For local use, `DJANGO_DEBUG=true` and `DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost` are suitable. Set a unique `DJANGO_SECRET_KEY` even in local development. Never commit `.env`.

3. Obtain the dataset from an authorized source and confirm its licence. Put the training file at `data/raw/train.csv`. Raw data is intentionally not tracked. The target is `TARGET(PRICE_IN_LACS)`; the project notebook records 29,451 training rows and 68,720 test rows.

4. Train the pipeline from the repository root:
```powershell
python audit_dataset.py
python -m src.train_models
```
This should create `models/best_model.joblib` and reports under `reports/`. The model artifact is ignored by Git because it can be larger than GitHub's normal file-size limit. A fresh clone must train a model or obtain a trusted compatible artifact before predictions can run.

5. Check and start Django:
```powershell
python backend\manage.py check
python backend\manage.py test
python backend\manage.py runserver
```
Open http://127.0.0.1:8000/.

## Prediction API
`POST /api/predict/` with `Content-Type: application/json`. The endpoint uses Django CSRF protection; browser requests should first load the home page to receive the CSRF cookie.

Example body:
```json
{
  "POSTED_BY": "Owner",
  "UNDER_CONSTRUCTION": 0,
  "RERA": 1,
  "BHK_NO.": 2,
  "BHK_OR_RK": "BHK",
  "SQUARE_FT": 1000,
  "READY_TO_MOVE": 1,
  "RESALE": 1,
  "ADDRESS": "Example Locality, Bengaluru",
  "LONGITUDE": 12.9716,
  "LATITUDE": 77.5946
}
```
A successful response includes `predicted_price_lakhs`, `predicted_price_rupees`, and `currency`. Values are examples of the request schema, not an expected market price.

## Recorded validation metrics
These are historical results from a local run and should be reproduced before being treated as current:

| Model | MAE (lakhs) | RMSE (lakhs) | R² |
|---|---:|---:|---:|
| Linear Regression | 1,362.60 | 58,414.74 | -6,261.59 |
| Random Forest | 37.82 | 394.24 | 0.7148 |
| XGBoost | 42.74 | 446.33 | 0.6344 |

Random Forest had the lowest MAE in that run. The high RMSE relative to MAE indicates extreme prediction errors that need analysis by price range, location, and property type.

## Security and limitations
- Settings read `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, and `DJANGO_ALLOWED_HOSTS` from environment variables; `.env` is loaded locally and ignored by Git. Production must use a unique secret, `DJANGO_DEBUG=false`, correct hostnames, HTTPS, and Django's deployment checklist.
- The model loads on the first prediction. If the artifact is missing, the API returns a clear service-unavailable response rather than preventing Django from starting.
- Server-side validation rejects missing fields, invalid categories, non-finite numbers, and out-of-range values.
- `src/preprocessing.py` currently swaps the raw coordinate column interpretation to normalize the dataset. Verify the source dataset's schema and known coordinates before relying on geo features; don't change it without checking the actual data.
- Listing prices may differ from sale prices; dataset provenance and usage rights must be verified before public redistribution.
- The dataset and trained model are not stored in this repository.

## Roadmap
1. Reproduce clean install, training, and tests.
2. Verify coordinate mapping against source records.
3. Add evaluation by geography and price band, cross-validation, and a simple baseline.
4. Add API schema docs, structured logging, rate limiting, and deployment configuration.
5. Deploy with a versioned model artifact, CI, monitoring, and drift checks.

## Disclaimer
For educational and informational use only; not legal, tax, lending, investment, or certified valuation advice.
