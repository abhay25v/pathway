# Pathway & Bitcoin Direction Projects

This repository collects two small projects built around data-driven workflows:

- **Task 1 – Pathway Demo**: minimal example demonstrating the Pathway stream-processing API.
- **Task 2 – Bitcoin Direction Predictor**: machine-learning pipeline that forecasts whether Bitcoin closes higher on the next day.

## Repository Structure

- `task1/app.py` – Pathway demo script that prints debug table rows.
- `task1/requirements.txt` – Dependencies for the demo app.
- `task1/Dockerfile` – Container recipe for Task 1.
- `task2/src/predictor.py` – Feature engineering, model training, and plotting for Task 2.
- `task2/data/bitcoin.csv` – Semicolon-delimited BTC OHLCV sample.
- `task2/plots/` – Auto-generated analytics figures.

## Getting Started

```bash
git clone <repo-url>
cd <repo-directory>
```

### Task 1 – Pathway Demo

```bash
cd task1
pip install -r requirements.txt
python app.py
```

Expected result: the Pathway runtime prints the rows defined in `app.table`.

Docker workflow:

```bash
cd task1
docker build -t pathway-app .
docker run --rm pathway-app
```

### Task 2 – Bitcoin Direction Predictor

```bash
cd task2
pip install -r requirements.txt
python src/predictor.py
```

Pipeline steps:

1. Load `data/bitcoin.csv` (semicolon separator) into pandas.
2. Engineer lag-based indicators: returns, moving averages, rolling volatility, RSI, and volume z-scores via `compute_rsi`.
3. Train a tuned `RandomForestClassifier` on an 80 / 20 time split with balanced class weights.
4. Print directional accuracy and a full classification report.
5. Generate plots saved to `plots/prediction_plot.png` and `plots/direction_plot.png`.

## Data Requirements

- The predictor expects the CSV columns shipped by CoinMarketCap (`timestamp`, `open`, `high`, `low`, `close`, `volume`, ...).
- Replace `task2/data/bitcoin.csv` with any one-year BTC dataset following the same schema.

## Outputs

- Console metrics summarizing model performance.
- Predicted vs. actual price trajectory plot.
- Predicted vs. actual direction plot.

