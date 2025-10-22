import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
	"""Return the classically-defined Relative Strength Index."""
	delta = series.diff()
	gain = delta.clip(lower=0.0)
	loss = -delta.clip(upper=0.0)
	avg_gain = gain.rolling(window=window, min_periods=window).mean()
	avg_loss = loss.rolling(window=window, min_periods=window).mean()
	rs = avg_gain / avg_loss.replace(0, np.nan)
	rsi = 100 - (100 / (1 + rs))
	return rsi.fillna(method="bfill")


# Load and preprocess data (CSV uses semicolon separator)
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "bitcoin.csv")
df = pd.read_csv(data_path, sep=";")

# Ensure chronological order and create canonical columns
df["Date"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("Date").reset_index(drop=True)
df["Close"] = df["close"].astype(float)
df["Open"] = df["open"].astype(float)
df["High"] = df["high"].astype(float)
df["Low"] = df["low"].astype(float)
df["Volume"] = df["volume"].astype(float)

# Core features based only on past information
df["return"] = df["Close"].pct_change()
df["log_return"] = np.log(df["Close"]).diff()
df["range"] = (df["High"] - df["Low"]) / df["Close"]

for window in (3, 5, 10, 20):
	df[f"sma_{window}"] = df["Close"].rolling(window).mean()
	df[f"ema_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
	df[f"volatility_{window}"] = df["return"].rolling(window).std()

df["volume_zscore"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
df["rsi_14"] = compute_rsi(df["Close"], window=14)

# Forward-looking target: does the next close go up?
df["future_close"] = df["Close"].shift(-1)
df["Direction"] = (df["future_close"] > df["Close"]).astype(int)

# Drop rows that contain NaNs from rolling calculations or target shift
df = df.dropna().reset_index(drop=True)

feature_columns = [
	"Close",
	"Open",
	"High",
	"Low",
	"return",
	"log_return",
	"range",
	"sma_3",
	"sma_5",
	"sma_10",
	"sma_20",
	"ema_3",
	"ema_5",
	"ema_10",
	"ema_20",
	"volatility_3",
	"volatility_5",
	"volatility_10",
	"volatility_20",
	"volume_zscore",
	"rsi_14",
]

train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size].set_index("Date")
test_df = df.iloc[train_size:].set_index("Date")

X_train = train_df[feature_columns]
y_train = train_df["Direction"]
X_test = test_df[feature_columns]
y_test = test_df["Direction"]

# Random forest handles non-scaled features and nonlinear interactions
model = RandomForestClassifier(
	n_estimators=500,
	max_depth=6,
	min_samples_leaf=10,
	random_state=42,
	class_weight="balanced_subsample",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Directional Accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))

# Simulated price path using average absolute return as move magnitude proxy
avg_abs_return = train_df["return"].abs().mean()
predicted_returns = np.where(y_pred == 1, avg_abs_return, -avg_abs_return)
predicted_prices = [test_df["Close"].iloc[0]]
for ret in predicted_returns:
	predicted_prices.append(predicted_prices[-1] * (1 + ret))
predicted_prices = pd.Series(predicted_prices[1:], index=test_df.index[: len(predicted_returns)])

actual_prices = test_df["Close"].iloc[: len(predicted_returns)]

plt.figure(figsize=(12, 5))
plt.plot(actual_prices.index, actual_prices.values, label="Actual Close", linewidth=2)
plt.plot(predicted_prices.index, predicted_prices.values, label="Predicted Price Path", linestyle="--")
plt.title("Predicted vs Actual Bitcoin Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, linestyle=":")

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "plots"), exist_ok=True)
plt.savefig(os.path.join(os.path.dirname(__file__), "..", "plots", "prediction_plot.png"), dpi=200)
plt.show()

# Direction comparison plot for quick visual check
plt.figure(figsize=(12, 3))
plt.plot(y_test.reset_index(drop=True).values, label="Actual Direction")
plt.plot(pd.Series(y_pred).values, label="Predicted Direction", linestyle="--")
plt.title("Predicted vs Actual Direction (1 = Up)")
plt.xlabel("Test Sample Index")
plt.ylabel("Direction")
plt.legend()
plt.grid(True, linestyle=":")

plt.savefig(os.path.join(os.path.dirname(__file__), "..", "plots", "direction_plot.png"), dpi=200)
plt.show()