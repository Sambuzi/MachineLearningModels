import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def create_dataset(series, n_lags):
	X, target = [], []
	for t in range(n_lags, len(series)):
		X.append(series[t - n_lags:t])
		target.append(series[t])
	return np.array(X), np.array(target)


def sliding_forecast(model, last_window, horizon):
	window = last_window.copy()
	preds = []
	for _ in range(horizon):
		next_pred = model.predict(window.reshape(1, -1))[0]
		preds.append(next_pred)
		window = np.roll(window, -1)
		window[-1] = next_pred
	return np.array(preds)


def main(csv_path='M3C_monthly.CSV', lookback=12, nfore=12):
	df = pd.read_csv(
		csv_path,
		sep=';',
		decimal=',',
		engine='python',
		on_bad_lines='skip',
		encoding='latin1',
	)

	# select the same series as forecast_mlp: row 505, columns from index 6 onwards
	rawdata = df.iloc[505, 6:].values.astype(float)
	ds = pd.Series(rawdata)
	ds.index = pd.RangeIndex(start=0, stop=len(ds))
	y_series = ds.values.astype(float)

	# Log-transform
	log_y = np.log(y_series)

	# Prepare lagged dataset
	nlags = lookback  # use past `lookback` months to predict next month
	X_all, y_all = create_dataset(log_y, nlags)

	# Train/validation split
	split_idx = int(0.8 * len(X_all))
	X_train, X_valid = X_all[:split_idx], X_all[split_idx:]
	y_train, y_valid = y_all[:split_idx], y_all[split_idx:]

	# Optuna objective
	def objective(trial):
		params = {
			"objective": "reg:squarederror",
			"eval_metric": "rmse",
			"booster": "gbtree",
			"n_estimators": trial.suggest_int("n_estimators", 100, 800),
			"max_depth": trial.suggest_int("max_depth", 2, 8),
			"learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
			"min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
			"subsample": trial.suggest_float("subsample", 0.6, 1.0),
			"colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
			"gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
			"reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
			"reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
			"random_state": 666,
		}
        #using XGBoost as model
		model = xgb.XGBRegressor(**params)
		model.fit(
			X_train,
			y_train,
			eval_set=[(X_valid, y_valid)],
			verbose=False,
		)

		pred_valid = model.predict(X_valid)
		rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
		return rmse

	# Run Optuna for searching optimal hyperparameters
	study = optuna.create_study(direction="minimize")
	study.optimize(objective, n_trials=50)

	print("Best RMSE:", study.best_value)
	print("Best params:")
	for k, v in study.best_params.items():
		print(f"  {k}: {v}")

	# Fit final model on all data
	best_model = xgb.XGBRegressor(random_state=42, **study.best_params)
	best_model.fit(X_all, y_all, verbose=False)

	# Forecast
	H = 12  # forecast next 12 months
	last_window = log_y[-nlags:]
	forecast_log = sliding_forecast(best_model, last_window, H)

	# Undo log transform
	forecast = np.exp(forecast_log)
	print("Forecast:", np.round(forecast, 1))

	# Plot
	time = np.arange(len(y_series))
	future_time = np.arange(len(y_series), len(y_series) + H) - 12
	plt.figure(figsize=(10, 5))
	plt.plot(time, y_series, label="Observed")
	plt.plot(future_time, forecast, label="Forecast")
	plt.axvline(len(y_series) - 1, linestyle="--")
	plt.legend()
	plt.title("Airline Passengers: XGBoost + Optuna")
	plt.show()


if __name__ == "__main__":
	main()