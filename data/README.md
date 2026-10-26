# Dataset setup

Raw datasets are intentionally excluded from Git. Obtain the dataset from its original, authorized source and review its licence/terms before using it.

Place the training CSV at `data/raw/train.csv`. The current training pipeline reads that path by default. If you use the separate test CSV, place it at `data/raw/test.csv`.

Expected training target: `TARGET(PRICE_IN_LACS)`.

Do not commit raw data, personally identifiable information, or datasets you are not authorized to redistribute.
