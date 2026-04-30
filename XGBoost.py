import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


def create_dataset(arr, lookback=12):
	X, y = [], []
	arr = np.asarray(arr).flatten()
	for i in range(len(arr) - lookback):
		X.append(arr[i:i + lookback])
		y.append(arr[i + lookback])
	return np.array(X), np.array(y)


def main(csv_path='M3C_monthly.CSV', lookback=12, nfore=12):
	df = pd.read_csv(
		csv_path,
		sep=';',
		decimal=',',
		engine='python',
		on_bad_lines='skip',
		encoding='latin1',
	)

	# select the same series as other scripts: row 505, columns from index 6 onwards
	rawdata = df.iloc[505, 6:].values.astype(float)
	ds = pd.Series(rawdata)
	ds.index = pd.RangeIndex(start=0, stop=len(ds))
	vals = ds.values.astype(float)

	X, y = create_dataset(vals, lookback)

	x_train = X[:-nfore]
	y_train = y[:-nfore]

	# fit model
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	model.fit(x_train, y_train)

	# rolling nfore-step prediction
	xinput = x_train[-1].astype(float).copy()
	yfore = []
	for i in range(nfore):
		pred = model.predict(xinput.reshape(1, -1))[0]
		yfore.append(pred)
		xinput = np.roll(xinput, -1)
		xinput[-1] = pred

	# Plot
	plt.plot(vals, label="Actual series")
	plt.plot(range(len(vals) - nfore, len(vals)), yfore, "-o", label="12-forecast")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()
