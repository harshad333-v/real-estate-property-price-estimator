# Repository benchmark and implementation decisions

This document records useful engineering patterns reviewed from public GitHub examples and how they are being adapted to this project. The goal is to adopt practices, not copy code or compare metrics across incompatible datasets.

## References reviewed

| Reference | Relevant documented patterns | Decision for this project |
|---|---|---|
| [alikhalill/house-price-app](https://github.com/alikhalill/house-price-app) | Indian listing data, model comparison, trained estimator API, React/TypeScript frontend | Keep Django for now; prioritize data validation, evaluation, and reliable inference before a frontend rewrite |
| [mohamedalaah5ss-oss/house-price-ml-web-app](https://github.com/mohamedalaah5ss-oss/house-price-ml-web-app) | Reproducible training notebook, preprocessing packaged with the model, API validation, testable web app | Keep preprocessing in the sklearn pipeline and add automated endpoint checks |
| [delmore13/house-price-mlops-pipeline](https://github.com/delmore13/house-price-mlops-pipeline) | Model comparison reports, experiment metadata, API tests, Docker-ready layout | Add machine-readable evaluation metrics, a model card, and price-segment diagnostics first; defer MLflow/Docker until data and deployment requirements are verified |
| [prakhar-bip/real-estate-valuation-engine](https://git.mitgai.net/prakhar-bip/real-estate-valuation-engine) | Geospatial feature engineering, explainability, prediction logging, containerized serving | Treat coordinate verification as a release blocker; add locality-aware validation before SHAP or an ensemble |
| [MoKenany/house-price-prediction](https://github.com/MoKenany/house-price-prediction) | Form validation, model artifact workflow, endpoint-driven prediction UI | Keep explicit input validation and document missing-model behavior |

Repository descriptions and READMEs are self-reported. Their scores are not directly comparable with this project unless dataset, target units, cleaning, split strategy, and evaluation methodology match.

## Implemented on the fix/reproducible-secure-mvp branch

- Environment-based Django configuration and a .env.example.
- Lazy model loading and clear service-unavailable behavior when the artifact is absent.
- Request validation and Django endpoint tests.
- GitHub Actions workflow for Django system checks and tests.
- GET /api/health/ reports whether the model artifact exists.
- Training now writes:
  - reports/model_comparison.csv — metrics for candidate models.
  - reports/model_metrics.json — split details, metrics, selected model, and price-band performance.
  - reports/price_segment_metrics.csv — MAE/RMSE and R² where defined, by actual-price band.
  - reports/model_card.json — intended use, selected model, validation metrics, and known limitations.

## Important interpretation rules

- Prices and errors are measured in INR lakhs unless the field says otherwise.
- The training script selects a model by validation MAE. RMSE is still reported to expose large errors.
- Price bands are descriptive diagnostics, not guaranteed confidence intervals.
- The current random holdout does not prove generalization to new cities or future listings.
- Coordinate mapping must be verified against the actual authorized source dataset before making geographic claims.
- Do not present any benchmark as a certified valuation or guarantee of market price.

## Remaining roadmap

1. Verify the raw dataset schema, provenance, licence, and coordinate interpretation.
2. Re-run training and inspect the new price-segment report.
3. Add cross-validation and a final untouched test set.
4. Add geography-based holdout evaluation and locality-level error analysis.
5. Add feature explanations only after input features and model behavior are validated.
6. Add Docker and deployment/monitoring after choosing a hosting target and artifact-delivery strategy.